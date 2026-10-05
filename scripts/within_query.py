"""The answer axis: format sensitivity inside one query.

Scores the gold passage and its BM25 negatives under each of the five
metadata formats. It then measures how far the ranking inside a candidate
list moves with the format. Every model in the roster is measured under the
two conditions the score axis uses:

  1. `with_body`: the passage is rendered metadata followed by budgeted body
     prose. This is the primary condition.
  2. `metadata_only`: the passage is the rendered metadata block alone.

The two conditions share one candidate list per query, so they cover
identical queries.

A model that fails to load or to score is recorded in `models_failed` and does
not stop the other models. Results are written after each model completes.

Writes `{split}_within_{mode}.json` to the output directory.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

# Environment settings to reduce parallelism overhead in tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import torch

from fsr.candidates import (
    DEFAULT_NEGATIVES,
    build_candidate_lists,
    measure_overflow,
)
from fsr.cli import add_common_args, take
from fsr.corpus.files import write_atomic
from fsr.corpus.layout import split_dir
from fsr.corpus.splitting import SPLIT_CHOICES, load_records, strict_gated
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.models.loading import load_model, load_tokenizer
from fsr.models.registry import BASE_MODEL_IDS, BASE_MODELS, by_slug
from fsr.passages import MAX_TOKENS
from fsr.reporting import (
    NAME_WIDTH,
    ProgressCounter,
    failures_block,
    format_duration,
    heading,
    model_heading,
    report_elapsed,
    report_saved,
    shorten,
    table,
)
from fsr.scoring import score_batch
from fsr.within_query import (
    conditional_inconsistency_from_matrices,
    gold_leads,
    gold_top1_from_matrices,
    score_scale_diagnostic,
    within_query_mrr,
    within_query_rank_stability,
)

AXIS = "within"
OUT_SUBDIR = "h1"
MODES = ("metadata_only", "with_body")
PARSED_NAME = "parsed_train.json"


def resolve_tanh_heads(names: list[str], override: bool | None) -> list[bool]:
    """Return whether each named model wants the two-layer tanh head.

    Args:
        names: Registry slugs or HF identifiers.
        override: The value the caller asked for, or None for the registry.

    Returns:
        One flag per name, in the order given.
    """
    if override is not None:
        return [override] * len(names)
    return [False if "/" in name else by_slug(name).tanh_head for name in names]


def resolve_models(names: list[str]) -> list[str]:
    """Return the Hugging Face identifier of each named model.

    Args:
        names: Registry slugs or Hugging Face identifiers.

    Returns:
        The identifiers, in the order given.

    Raises:
        ValueError: If a name is neither a slug nor an identifier.
    """
    return [name if "/" in name else by_slug(name).model_id for name in names]


def load_split(split_path: Path, name: str) -> list[dict]:
    """Return the records of one split."""
    return json.loads((split_path / f"{name}.json").read_text())["records"]


def load_negatives(split_path: Path) -> dict[str, list[str]]:
    """Return the mined negative ids of each query, by query id."""
    return json.loads((split_path / "bm25_negatives.json").read_text())["negatives"]


def load_corpus_index(data_root: Path, split_path: Path) -> dict[str, dict]:
    """Return every record a negative may cite, by id.

    Args:
        data_root: The directory holding the parsed corpus.
        split_path: The directory holding the split files.

    Returns:
        The quality-passed training records, plus the validation split.
    """
    parsed = json.loads((data_root / PARSED_NAME).read_text())
    by_id = {r["id"]: r for r in strict_gated(parsed["records"])}
    for record in load_split(split_path, "nq_val"):
        by_id[record["id"]] = record
    return by_id


def score_all_formats(
    model: Any,
    tokenizer: Any,
    prepared: list[dict],
    batch_size: int,
    device: str,
    with_body: bool = True,
    verbose: bool = True,
    counter: ProgressCounter | None = None,
    label: str = "",
) -> dict[str, np.ndarray]:
    """Score every candidate of every query, once per format.

    Args:
        model: The reranker.
        tokenizer: The tokenizer of that reranker.
        prepared: The queries with their candidate lists.
        batch_size: The scoring batch size.
        device: The device to score on.
        with_body: Whether a candidate carries its body text.
        verbose: Whether to report each format as it completes.
        counter: The progress counter of the job, if one is running.
        label: What to prefix each counted format with.

    Returns:
        One (n_queries, n_candidates) array per format.
    """
    n_queries = len(prepared)
    n_candidates = len(prepared[0]["candidates"]) if prepared else 0
    matrices = {}
    for name in FORMAT_NAMES:
        render = FORMATS[name]
        pairs = [
            (
                query["question"],
                render(c["pairs"], c["truncated_body"] if with_body else ""),
            )
            for query in prepared
            for c in query["candidates"]
        ]
        flat = score_batch(model, tokenizer, pairs, batch_size, device)
        matrix = np.asarray(flat, dtype=float).reshape(n_queries, n_candidates)
        matrices[name] = matrix
        if verbose:
            print(
                f"  scored {name:>10}  "
                f"gold mean={matrix[:, 0].mean():+.3f}  "
                f"neg mean={matrix[:, 1:].mean():+.3f}"
            )
        if counter is not None:
            counter.step(f"{label}/{name}" if label else name)
    return matrices


def summarise_model(
    matrices: dict[str, np.ndarray], store_matrices: bool = False
) -> dict[str, Any]:
    """Compute every within-query statistic for one model.

    Args:
        matrices: The candidate scores of each format.
        store_matrices: Whether to keep the raw score matrices.

    Returns:
        The statistics, with the gold column kept so the cross-query flip rate
        can be recomputed on exactly these queries.
    """
    per_pair, summary, per_query = within_query_rank_stability(matrices)
    mrr = within_query_mrr(matrices)
    entry = {
        "within_query": {"pairwise": per_pair, "summary": summary},
        "per_query": per_query,
        "mrr_per_format": {f: v["mrr"] for f, v in mrr.items()},
        "reciprocal_ranks": {f: v["reciprocal_ranks"] for f, v in mrr.items()},
        "gold_top1": gold_top1_from_matrices(matrices),
        "conditional_inconsistency": conditional_inconsistency_from_matrices(matrices),
        "gold_scores_per_fmt": {f: m[:, 0].tolist() for f, m in matrices.items()},
        "gold_leads_per_fmt": {f: v.tolist() for f, v in gold_leads(matrices).items()},
        "scale_diagnostic": score_scale_diagnostic(matrices),
    }
    if store_matrices:
        entry["matrices"] = {f: m.tolist() for f, m in matrices.items()}
    return entry


def report_model(entry: dict[str, Any]) -> None:
    """Print the statistics of one model."""
    summary = entry["within_query"]["summary"]
    gold_top1 = entry["gold_top1"]
    print(
        f"\n  max within-query flip %: {summary['max_flip_rate_pct']:.1f}%  "
        f"({summary['max_flip_pair']})"
    )
    print(
        f"  min Kendall tau:         {summary['min_kendall_tau']:.3f}  "
        f"({summary['min_tau_pair']})"
    )
    print(
        f"  max top-1 change %:      {summary['max_top1_changed_pct']:.1f}%  "
        f"({summary['max_top1_pair']})"
    )
    print(f"  delta/S_within:          {entry['scale_diagnostic']['ratio']:.3f}")
    print(
        f"  gold top-1 all formats:  {gold_top1['gold_top1_all_formats_pct']:.1f}%  |  "
        f"format-dependent: {gold_top1['gold_top1_format_dependent_pct']:.1f}%  |  "
        f"never: {gold_top1['gold_top1_never_pct']:.1f}%"
    )
    print(
        "  MRR per format:          "
        + "  ".join(f"{f}={v:.3f}" for f, v in entry["mrr_per_format"].items())
    )


def report_summary(
    results: dict[str, dict],
    failures: list[dict],
    n_candidates: int,
    split: str,
    n_queries: int,
) -> None:
    """Print the cross-model table.

    Args:
        results: The statistics of each model.
        failures: The models that did not complete.
        n_candidates: The candidates each query carries.
        split: The split measured.
        n_queries: The queries measured.
    """
    heading(
        f"WITHIN-QUERY SUMMARY - split={split}  "
        f"n_queries={n_queries}  candidates={n_candidates}"
    )
    table(
        (
            ("model", NAME_WIDTH),
            ("max flip%", 10),
            ("min tau", 9),
            ("top1 chg%", 10),
            ("fmt-dep%", 9),
            ("delta/S", 9),
        ),
        [
            [
                shorten(model_name),
                f"{summary['max_flip_rate_pct']:.1f}%",
                f"{summary['min_kendall_tau']:.3f}",
                f"{summary['max_top1_changed_pct']:.1f}%",
                f"{entry['gold_top1']['gold_top1_format_dependent_pct']:.1f}%",
                f"{entry['scale_diagnostic']['ratio']:.3f}",
            ]
            for model_name, entry in results.items()
            for summary in [entry["within_query"]["summary"]]
        ],
    )
    print(
        "\n  fmt-dep% = queries whose gold ranks first under some formats "
        "but not others."
    )
    failures_block(failures)


def load_tokenizers(model_ids: list[str]) -> dict[str, Any]:
    """Load every tokenizer, or stop.

    Budgets come from the whole set, so a partial set would truncate
    candidates differently and break comparability between runs.

    Args:
        model_ids: The models whose tokenizers to load.

    Returns:
        The tokenizers, by model identifier.

    Raises:
        SystemExit: If any tokenizer fails to load.
    """
    print(f"\nLoading tokenizers for {len(model_ids)} budget models...")
    tokenizers = {}
    for model_id in model_ids:
        try:
            tokenizers[model_id] = load_tokenizer(model_id)
            print(f"  ok  {model_id}")
        except Exception as error:
            print(f"  FAILED  {model_id}: {type(error).__name__}: {error}")
    if len(tokenizers) != len(model_ids):
        raise SystemExit("Not every tokenizer loaded.")
    return tokenizers


def prepare_queries(
    args: argparse.Namespace, split_path: Path, tokenizers: dict[str, Any]
) -> tuple[list[dict], dict[str, Any]]:
    """Build the candidate lists, or read the cache.

    Args:
        args: The parsed arguments.
        split_path: The directory holding the split files.
        tokenizers: The tokenizers to budget across.

    Returns:
        The prepared queries and the build statistics.

    Raises:
        SystemExit: If no query survives.
    """
    records = take(load_records(args.data_root, args.split), args.limit)
    print(f"\nSplit '{args.split}': {len(records)} queries")

    suffix = f"_limit{args.limit}" if args.limit else ""
    cache_path = args.out_dir / f"{args.split}_candidates{suffix}.json"
    if cache_path.exists() and not args.force:
        print(f"\nReusing candidate lists from {cache_path}")
        cached = json.loads(cache_path.read_text())
        prepared, stats = cached["prepared"], cached["prep_stats"]
        print(f"  {stats['n_kept']} queries")
    else:
        print("\nBuilding candidate lists, one body budget per candidate...")
        started = time.time()
        prepared, stats = build_candidate_lists(
            records,
            load_negatives(split_path),
            load_corpus_index(args.data_root, split_path),
            tokenizers,
            args.negatives,
        )
        print(f"  {stats['n_kept']} queries kept")
        report_elapsed("built candidate lists", started)
        write_atomic(
            cache_path, json.dumps({"prepared": prepared, "prep_stats": stats})
        )
        print(f"  cached -> {cache_path}")

    for key in (
        "dropped_gold_budget",
        "dropped_no_bm25_negs",
        "dropped_too_few_eligible_negs",
        "negatives_skipped_for_budget",
    ):
        print(f"  {key}: {stats[key]}")
    if not prepared:
        raise SystemExit("No queries survived candidate-list construction.")
    return prepared, stats


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="test", choices=list(SPLIT_CHOICES))
    ap.add_argument(
        "--mode",
        choices=[*MODES, "both"],
        default="both",
        help="Render the candidates with their body, with metadata alone, or both",
    )
    ap.add_argument("--models", nargs="+", default=list(BASE_MODEL_IDS))
    ap.add_argument(
        "--budget-models",
        nargs="+",
        default=list(BASE_MODEL_IDS),
        help="Tokenizers the per-candidate body budgets come from.",
    )
    ap.add_argument(
        "--lora-adapter",
        default=None,
        help="PEFT adapter to load on the base model. Needs exactly one --models entry",
    )
    ap.add_argument(
        "--tanh-head",
        action="store_true",
        default=None,
        help="Replace the classifier head. The default is the registry entry",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help="Request the eager attention kernel in place of the fused one",
    )
    ap.add_argument("--negatives", type=int, default=DEFAULT_NEGATIVES)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="File to keep the live progress of this job in",
    )
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help="Write the result to this file. Needs a single --mode",
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help="Appended to the file name, to keep parallel runs apart",
    )
    ap.add_argument(
        "--overflow-sample",
        type=int,
        default=100,
        help=f"Queries sampled to measure the share of renderings over "
        f"{MAX_TOKENS} tokens (0 disables)",
    )
    ap.add_argument(
        "--list-models",
        action="store_true",
        help="Print the registry slug of each model in the roster, then stop",
    )
    ap.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build and cache the candidate lists, then stop before scoring",
    )
    ap.add_argument(
        "--store-matrices",
        action="store_true",
        help="Keep the raw score matrices as well as the statistics",
    )
    add_common_args(ap, "Cap on queries measured (0 = no cap)")
    return ap


def main() -> None:
    """Measure within-query rank disturbance for each requested model.

    Raises:
        SystemExit: If an adapter is given for more than one model.
    """
    args = build_parser().parse_args()
    if args.list_models:
        for model in BASE_MODELS:
            print(model.slug)
        return
    args.out_dir = args.out_dir or args.data_root / OUT_SUBDIR
    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = list(MODES) if args.mode == "both" else [args.mode]
    if args.out_path is not None:
        if len(modes) != 1:
            raise SystemExit("--out-path needs a single --mode, not both.")
        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        out_paths = {modes[0]: args.out_path}
    else:
        suffix = f"_{args.out_tag}" if args.out_tag else ""
        out_paths = {
            mode: args.out_dir / f"{args.split}_{AXIS}_{mode}{suffix}.json"
            for mode in modes
        }
    split_path = split_dir(args.data_root)

    if args.lora_adapter and len(args.models) != 1:
        raise SystemExit(
            "--lora-adapter applies to a single model; pass one --models entry."
        )

    models = resolve_models(args.models)
    tanh_heads = resolve_tanh_heads(args.models, args.tanh_head)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    tokenizers = load_tokenizers(resolve_models(args.budget_models))
    prepared, prep_stats = prepare_queries(args, split_path, tokenizers)
    if args.prepare_only:
        print("Prepared the candidate lists; stopping before scoring.")
        return

    overflow = measure_overflow(
        prepared, tokenizers, args.overflow_sample, seed=args.seed
    )
    prep_stats["overflow"] = overflow
    if overflow:
        worst = max(overflow["over_max_tokens_pct"].values())
        print(
            f"  >{MAX_TOKENS}-token renderings: worst model {worst:.2f}% "
            f"(sampled {overflow['queries_sampled']} queries)"
        )

    results: dict[str, dict[str, dict]] = {mode: {} for mode in modes}
    failures: list[dict] = []
    n_candidates = len(prepared[0]["candidates"])

    def save(mode: str) -> None:
        out_paths[mode].write_text(
            json.dumps(
                {
                    "split": args.split,
                    "mode": mode,
                    "lora_adapter": args.lora_adapter,
                    "tanh_head": dict(zip(args.models, tanh_heads, strict=True)),
                    "budget_models": resolve_models(args.budget_models),
                    "neg_count_per_query": args.negatives,
                    "n_queries": len(prepared),
                    "query_ids": [q["id"] for q in prepared],
                    "formats": list(FORMAT_NAMES),
                    "prep_stats": prep_stats,
                    "models_probed": list(results[mode]),
                    "models_failed": failures,
                    "results": results[mode],
                }
            )
        )

    counter = ProgressCounter(
        len(models) * len(modes) * len(FORMAT_NAMES), args.progress_file
    )
    for model_name, tanh_head in zip(models, tanh_heads, strict=True):
        model_heading(model_name)
        try:
            model = load_model(
                model_name,
                device,
                lora_adapter_path=args.lora_adapter,
                eager_attn=args.eager_attn,
                tanh_head=tanh_head,
            )
            if tanh_head:
                print("  classifier head: dense->tanh->out_proj")
            if args.lora_adapter:
                print(f"  adapter: {args.lora_adapter}")
        except Exception as error:
            print(f"  model load failed: {type(error).__name__}: {error}")
            counter.fail()
            failures.append(
                {"model": model_name, "stage": "model_load", "error": str(error)}
            )
            for mode in modes:
                save(mode)
            continue

        try:
            started = time.time()
            tokenizer = tokenizers.get(model_name) or load_tokenizer(model_name)
            for mode in modes:
                print(f"  {mode}")
                matrices = score_all_formats(
                    model,
                    tokenizer,
                    prepared,
                    args.batch_size,
                    device,
                    with_body=mode == "with_body",
                    counter=counter,
                    label=mode,
                )
                entry = summarise_model(matrices, args.store_matrices)
                results[mode][model_name] = entry
                report_model(entry)
                save(mode)
            report_elapsed("scored", started)
        except Exception as error:
            print(f"  scoring failed: {type(error).__name__}: {error}")
            counter.fail()
            failures.append(
                {"model": model_name, "stage": "scoring", "error": str(error)}
            )
            for mode in modes:
                save(mode)

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    for mode in modes:
        report_summary(
            results[mode], failures, n_candidates, f"{args.split}/{mode}", len(prepared)
        )
        save(mode)
        report_saved(out_paths[mode], mode)
    print(f"\nPass complete in {format_duration(counter.elapsed)}", flush=True)


if __name__ == "__main__":
    main()
