"""The score axis: cross-query format sensitivity of one arm, with a guardrail.

Scores every record of a split in each of the five metadata formats, under a
shared body budget. It reports the format sensitivity of those scores. It then
scores the gold passage against its BM25 negatives in each format.

Writes `h2/{split}_cross_{model}_{arm}.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.corpus.splitting import load_records
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.h2_layout import BASE_ARM, adapter_dir, arm, result_path
from fsr.metrics import (
    DEFAULT_N_BOOT,
    bootstrap_ci_of_max_abs_d,
    bootstrap_ci_of_mean,
    format_sensitivity_summary,
    reciprocal_ranks,
)
from fsr.models.loading import load_model, load_tokenizer
from fsr.models.registry import BASE_MODEL_IDS
from fsr.passages import prepare_records_with_body
from fsr.reporting import ProgressCounter, heading, report_elapsed, report_saved
from fsr.scoring import score_batch
from fsr.training.data import load_corpus_index, load_negatives, truncate_to_budget

MRR_NEG_COUNT = 15
EVAL_SPLITS = ("dev", "test")


def render_positive(record: dict[str, Any], format_name: str) -> str:
    """Render the budgeted passage of a record in one format."""
    return FORMATS[format_name](record["pairs"], record["truncated_body"])


def render_negative(
    record: dict[str, Any], format_name: str, budget: int, tokenizer: Any
) -> str:
    """Render a negative passage, cut to the budget of the query it answers.

    Args:
        record: The negative record.
        format_name: The format to render in.
        budget: The token budget of the query.
        tokenizer: The tokenizer that measures the body.

    Returns:
        The rendered passage.
    """
    return FORMATS[format_name](
        record["pairs"], truncate_to_budget(record["body"], budget, tokenizer)
    )


def score_formats(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    device: str,
    counter: ProgressCounter | None = None,
) -> dict[str, list[float]]:
    """Score every record in every format.

    Args:
        model: The model to score with.
        tokenizer: The tokenizer that matches the model.
        records: The prepared records.
        batch_size: The pairs scored at one time.
        device: The device to score on.
        counter: The progress counter of the job, if one is running.

    Returns:
        The per-record scores, by format name.
    """
    scores_per_fmt = {}
    for name in FORMAT_NAMES:
        pairs = [(r["question"], render_positive(r, name)) for r in records]
        scores = score_batch(model, tokenizer, pairs, batch_size, device)
        scores_per_fmt[name] = scores
        print(f"  scored {name:>10}  mean={np.mean(scores):+.3f}  n={len(scores)}")
        if counter is not None:
            counter.step(f"scores/{name}")
    return scores_per_fmt


def score_guardrail(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    negatives: dict[str, list[str]],
    corpus: dict[str, dict[str, Any]],
    batch_size: int,
    device: str,
    counter: ProgressCounter | None = None,
) -> tuple[dict[str, float], dict[str, list[float]]]:
    """Score the gold passage against its negatives, in every format.

    Args:
        model: The model to score with.
        tokenizer: The tokenizer that matches the model.
        records: The prepared records.
        negatives: The mined negative identifiers, by query identifier.
        corpus: The records a negative identifier names, by identifier.
        batch_size: The pairs scored at one time.
        device: The device to score on.
        counter: The progress counter of the job, if one is running.

    Returns:
        The mean and per-record reciprocal ranks of each format.
    """
    per_format_mrr = {}
    per_format_rr = {}
    for name in FORMAT_NAMES:
        gold_pairs = []
        negative_pairs = []
        for r in records:
            gold_pairs.append((r["question"], render_positive(r, name)))
            for neg_id in negatives[r["id"]][:MRR_NEG_COUNT]:
                negative_pairs.append(
                    (
                        r["question"],
                        render_negative(
                            corpus[neg_id], name, r["body_budget_tokens"], tokenizer
                        ),
                    )
                )
        gold = score_batch(model, tokenizer, gold_pairs, batch_size, device)
        against = score_batch(model, tokenizer, negative_pairs, batch_size, device)
        matrix = np.array(against).reshape(len(records), MRR_NEG_COUNT)
        rr = reciprocal_ranks(gold, matrix)
        per_format_rr[name] = rr.tolist()
        per_format_mrr[name] = float(rr.mean())
        print(f"  MRR {name:>10}: {per_format_mrr[name]:.4f}")
        if counter is not None:
            counter.step(f"mrr/{name}")
    return per_format_mrr, per_format_rr


def guardrail_payload(
    per_format_mrr: dict[str, float],
    per_format_rr: dict[str, list[float]],
    record_ids: list[str],
    seed: int,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """Summarise the ranking guardrail.

    Args:
        per_format_mrr: The mean reciprocal rank of each format.
        per_format_rr: The per-record reciprocal ranks of each format.
        record_ids: The records the guardrail covered.
        seed: The seed for the interval resample.
        n_boot: Resamples behind the interval.

    Returns:
        The guardrail section of the output.
    """
    pooled = [rr for name in FORMAT_NAMES for rr in per_format_rr[name]]
    return {
        "per_format_mrr": per_format_mrr,
        "per_format_reciprocal_ranks": per_format_rr,
        "mean_mrr": float(np.mean(list(per_format_mrr.values()))),
        "min_mrr": float(min(per_format_mrr.values())),
        "all_format_rr_mean_ci95": bootstrap_ci_of_mean(
            pooled, n_boot=n_boot, seed=seed
        ),
        "neg_count_per_query": MRR_NEG_COUNT,
        "record_ids": record_ids,
    }


def arm_of(args: argparse.Namespace) -> str:
    """Return the arm an invocation evaluates.

    Args:
        args: The parsed arguments.

    Returns:
        The arm name. A baseline run evaluates the untrained model.
    """
    if args.baseline:
        return BASE_ARM
    held_out = None if args.held_out_format == "none" else args.held_out_format
    return arm(held_out, args.lambda_inv, args.rank_tag)


def load_budget_tokenizers(names: list[str]) -> dict[str, Any]:
    """Load the tokenizers the body budget is measured across.

    The budget is the tightest across the whole roster, so every model reads
    identical text and the arms stay comparable. A missing tokenizer would
    loosen the budget silently, so the load must be complete.

    Args:
        names: Registry slugs or Hugging Face identifiers.

    Returns:
        The tokenizers, by identifier.

    Raises:
        SystemExit: If any tokenizer fails to load.
    """
    loaded = {}
    for name in names:
        model_id = resolve_model(name).model_id
        if model_id in loaded:
            continue
        try:
            loaded[model_id] = load_tokenizer(model_id)
        except Exception as error:
            raise SystemExit(
                f"the budget tokenizer {model_id} did not load ({error}). The "
                "body budget comes from the whole roster, so fix the load and "
                "run again."
            ) from error
    return loaded


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Evaluate the untrained model instead of a trained arm",
    )
    ap.add_argument(
        "--held-out-format",
        default="none",
        choices=[*FORMAT_NAMES, "none"],
        help="The format the arm withheld from training",
    )
    ap.add_argument("--lambda-inv", type=float, default=1.0)
    ap.add_argument(
        "--rank-tag",
        type=int,
        default=None,
        help="The adapter rank the arm name carries, for a rank sweep",
    )
    ap.add_argument("--split", default="test", choices=list(EVAL_SPLITS))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--budget-models",
        nargs="+",
        default=list(BASE_MODEL_IDS),
        help="Tokenizers the body budget comes from. The default is the roster",
    )
    ap.add_argument(
        "--tanh-head",
        action="store_true",
        default=None,
        help="Replace the classifier head. The default is the registry entry",
    )
    ap.add_argument("--no-mrr", action="store_true", help="Skip the ranking guardrail")
    ap.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Resamples behind each interval",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="File to keep the live progress of this job in",
    )
    ap.add_argument("--limit", type=int, default=None, help="Cap records for a check")
    ap.add_argument(
        "--force", action="store_true", help="Evaluate again over a finished run"
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Evaluate one arm and write its results."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    arm_name = arm_of(args)
    out_path = result_path(args.data_root, args.split, "cross", entry.slug, arm_name)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {arm_name}: {out_path} is present.")
        return

    adapter = (
        None if args.baseline else adapter_dir(args.data_root, entry.slug, arm_name)
    )
    if adapter is not None and not adapter.exists():
        raise SystemExit(f"no adapter at {adapter}; train the arm first.")

    batch_size = args.batch_size or entry.eval_batch
    tanh_head = entry.tanh_head if args.tanh_head is None else args.tanh_head

    heading(f"EVAL  {entry.label}  split={args.split}  arm={arm_name}", width=70)
    print(f"  identifier:  {entry.model_id}")
    print(f"  adapter:     {adapter or 'none, untrained baseline'}")
    print(f"  batch:       {batch_size}")
    print(f"  output:      {out_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device:      {device}")

    tokenizer = load_tokenizer(entry.model_id)
    model = load_model(
        entry.model_id,
        device,
        lora_adapter_path=str(adapter) if adapter else None,
        tanh_head=tanh_head,
    )

    records = load_records(args.data_root, args.split, verbose=False)
    if args.limit:
        records = records[: args.limit]
    print(f"\n{len(records):,} records in {args.split}")
    budget_tokenizers = load_budget_tokenizers(args.budget_models)
    prepared, budget_stats = prepare_records_with_body(records, budget_tokenizers)
    print(f"  {len(prepared):,} kept, {budget_stats['dropped']:,} dropped")
    if not prepared:
        raise SystemExit("no records survived the body budget; nothing to score.")

    units = len(FORMAT_NAMES) * (1 if args.no_mrr else 2)
    counter = ProgressCounter(units, args.progress_file)

    started = time.time()
    print("\nScoring the formats...")
    scores_per_fmt = score_formats(
        model, tokenizer, prepared, batch_size, device, counter
    )
    summary = format_sensitivity_summary(scores_per_fmt)
    reported = summary["summary"]
    print(f"\n  max |d|:    {reported['max_abs_cohen_d']:.3f}")
    print(f"  mean |d|:   {reported['mean_abs_cohen_d']:.3f}")
    print(f"  min rho:    {reported['min_spearman_rho']:.3f}")
    print(f"  max flip:   {reported['max_flip_rate_pct']:.1f}%")
    low, high = bootstrap_ci_of_max_abs_d(
        scores_per_fmt, n_boot=args.n_boot, seed=args.seed
    )
    print(f"  max |d| 95% interval: [{low:.3f}, {high:.3f}]")
    report_elapsed("format scoring", started)

    guardrail = None
    if not args.no_mrr:
        started = time.time()
        print("\nScoring the ranking guardrail...")
        negatives = load_negatives(args.data_root)
        corpus = load_corpus_index(args.data_root, include_nq_val=True, verbose=False)
        covered = [r for r in prepared if r["id"] in negatives]
        if len(covered) < len(prepared):
            print(f"  {len(prepared) - len(covered)} records have no negatives")
        per_format_mrr, per_format_rr = score_guardrail(
            model, tokenizer, covered, negatives, corpus, batch_size, device, counter
        )
        guardrail = guardrail_payload(
            per_format_mrr,
            per_format_rr,
            [r["id"] for r in covered],
            args.seed,
            args.n_boot,
        )
        print(f"\n  mean MRR:   {guardrail['mean_mrr']:.4f}")
        print(f"  worst MRR:  {guardrail['min_mrr']:.4f}")
        report_elapsed("guardrail scoring", started)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": entry.slug,
                "model_id": entry.model_id,
                "arm": arm_name,
                "adapter": str(adapter) if adapter else None,
                "split": args.split,
                "limit": args.limit,
                "n_records_input": len(records),
                "n_records_kept": len(prepared),
                "body_budget_stats": budget_stats,
                "format_sensitivity": summary,
                "max_abs_d_ci95": [low, high],
                "n_boot": args.n_boot,
                "mrr_guardrail": guardrail,
                "scores_per_fmt": scores_per_fmt,
                "record_ids": [r["id"] for r in prepared],
            }
        )
    )
    report_saved(out_path, arm_name)


if __name__ == "__main__":
    main()
