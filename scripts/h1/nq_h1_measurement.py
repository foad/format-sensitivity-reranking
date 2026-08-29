"""Measure format sensitivity across every reranker and every format.

Runs each model in the roster over each format renderer, under two conditions:

  1. `with_body`: the passage is rendered metadata followed by budgeted body
     prose. This is the primary condition.
  2. `metadata_only`: the passage is the rendered metadata block alone.

The body budget for `with_body` uses the tightest tokenizer across every model
in the run, so each model scores identical body content on each record.

A model that fails to load or to score is recorded in `models_failed` and does
not stop the other models. Results are written after each model completes.

Writes `{split}_{mode}.json` to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fsr.common.metrics import format_sensitivity_summary
from fsr.common.rendering import (
    FORMATS,
    MIN_BODY_TOKENS,
    prepare_records_with_body,
)
from fsr.common.scoring import score_batch

DEFAULT_IN_DIR = Path("data") / "nq"
DEFAULT_OUT_DIR = Path("data") / "nq" / "h1_measurement"
SMOKE_TEST_RECORDS = 10
SMOKE_TEST_BATCH = 4

DEFAULT_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "cross-encoder/ms-marco-MiniLM-L12-v2",
    "BAAI/bge-reranker-base",
    "BAAI/bge-reranker-v2-m3",
    "mixedbread-ai/mxbai-rerank-base-v1",
    "jinaai/jina-reranker-v2-base-multilingual",
]


def try_load_tokenizer(model_name: str, trust_remote_code: bool) -> Any | None:
    """Load a tokenizer, or return None if the load fails.

    Args:
        model_name: The model identifier.
        trust_remote_code: Whether to run the vendor's remote modeling code.

    Returns:
        The tokenizer, or None.
    """
    try:
        return AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
    except Exception as e:
        print(f"  x tokenizer load failed for {model_name}: {type(e).__name__}: {e}")
        return None


def try_load_model(
    model_name: str,
    device: str,
    trust_remote_code: bool,
    force_eager_attn: bool = False,
) -> Any | None:
    """Load a model onto a device in evaluation mode, or return None.

    A model that rejects the eager-attention argument is retried without it.

    Args:
        model_name: The model identifier.
        device: The device to move the model to.
        trust_remote_code: Whether to run the vendor's remote modeling code.
        force_eager_attn: Whether to request the eager attention kernel.

    Returns:
        The model, or None.
    """
    load_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    if force_eager_attn:
        load_kwargs["attn_implementation"] = "eager"
    try:
        return (
            AutoModelForSequenceClassification.from_pretrained(
                model_name, **load_kwargs
            )
            .to(device)
            .eval()
        )
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        try:
            return (
                AutoModelForSequenceClassification.from_pretrained(
                    model_name, **load_kwargs
                )
                .to(device)
                .eval()
            )
        except Exception as e:
            print(f"  x model load failed for {model_name}: {type(e).__name__}: {e}")
            return None
    except Exception as e:
        print(f"  x model load failed for {model_name}: {type(e).__name__}: {e}")
        return None


def budget_summary(records: list[dict]) -> dict[str, int]:
    """Summarise the body-token budgets of the kept records.

    Args:
        records: Records carrying body_budget_tokens.

    Returns:
        The smallest, median, and largest budget. Each is 0 for no records.
    """
    budgets = [r["body_budget_tokens"] for r in records]
    if not budgets:
        return {"budget_min": 0, "budget_median": 0, "budget_max": 0}
    return {
        "budget_min": min(budgets),
        "budget_median": int(np.median(budgets)),
        "budget_max": max(budgets),
    }


def build_pairs(records: list[dict], renderer: Any, mode: str) -> list[tuple[str, str]]:
    """Render every record into a query and passage pair.

    Args:
        records: The records to render.
        renderer: The format renderer to apply.
        mode: Either with_body or metadata_only.

    Returns:
        One pair per record.
    """
    return [
        (
            r["question"],
            renderer(
                r["pairs"], r.get("truncated_body", "") if mode == "with_body" else ""
            ),
        )
        for r in records
    ]


def run_mode(
    mode: str,
    records: list[dict],
    models: list[str],
    tokenizers: dict[str, Any],
    args: argparse.Namespace,
    device: str,
    tightest_counts: dict[str, int] | None,
    drop_stats: dict[str, int] | None,
) -> None:
    """Score every model for one mode and write the results.

    Args:
        mode: Either with_body or metadata_only.
        records: The records to score.
        models: The model identifiers to run.
        tokenizers: The loaded tokenizers, by model identifier.
        args: The parsed command-line arguments.
        device: The device to score on.
        tightest_counts: The tightest-tokenizer counts, for with_body only.
        drop_stats: The drop and budget statistics, for with_body only.
    """
    print(f"\n{'=' * 70}")
    print(f"MODE: {mode}   n_records = {len(records)}")
    print(f"{'=' * 70}")

    out_path = args.out_dir / f"{args.split}_{mode}.json"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, Any] = {}
    failures: list[dict[str, str]] = []

    def save_partial() -> None:
        payload: dict[str, Any] = {
            "mode": mode,
            "split": args.split,
            "n_records": len(records),
            "source_cache": f"{args.in_dir.name}/parsed_{args.split}.json",
            "models_probed": list(all_results),
            "models_failed": failures,
            "formats": list(FORMATS),
            "results": all_results,
        }
        if mode == "with_body":
            payload["drop_stats"] = drop_stats
            payload["tightest_tokeniser_counts"] = tightest_counts
        out_path.write_text(json.dumps(payload))

    for model_name in models:
        if model_name not in tokenizers:
            print(f"\n  Skipping {model_name} (tokenizer missing)")
            failures.append(
                {
                    "model": model_name,
                    "stage": "tokenizer_load",
                    "error": "tokenizer missing",
                }
            )
            save_partial()
            continue

        print(f"\n{'-' * 70}\n{model_name}\n{'-' * 70}")
        model = try_load_model(
            model_name,
            device,
            not args.no_trust_remote_code,
            force_eager_attn=args.eager_attn,
        )
        if model is None:
            failures.append(
                {
                    "model": model_name,
                    "stage": "model_load",
                    "error": "load failed (see stdout)",
                }
            )
            save_partial()
            continue

        try:
            scores_per_fmt = {}
            for fmt_name, renderer in FORMATS.items():
                pairs = build_pairs(records, renderer, mode)
                scores = score_batch(
                    model, tokenizers[model_name], pairs, args.batch_size, device
                )
                scores_per_fmt[fmt_name] = scores
                print(f"  scored {fmt_name}  (mean={np.mean(scores):+.3f})")

            stats_out = format_sensitivity_summary(scores_per_fmt, seed=args.seed)
            all_results[model_name] = {"scores": scores_per_fmt, "stats": stats_out}

            s = stats_out["summary"]
            print(f"\n  max |d|:     {s['max_abs_cohen_d']:.3f}  ({s['max_d_pair']})")
            print(f"  min rho:     {s['min_spearman_rho']:.3f}  ({s['min_rho_pair']})")
            print(
                f"  max flip %:  {s['max_flip_rate_pct']:.1f}%  ({s['max_flip_pair']})"
            )
            print(f"  ranking:     {' > '.join(s['format_ranking_by_mean'])}")
            save_partial()
        except Exception as e:
            print(f"  x scoring failed for {model_name}: {type(e).__name__}: {e}")
            failures.append(
                {
                    "model": model_name,
                    "stage": "scoring",
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            save_partial()

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n\n{'=' * 100}")
    print(f"CROSS-MODEL SUMMARY - {mode}  (n={len(records)})")
    print(f"{'=' * 100}")
    header = f"{'model':<48} {'max |d|':>10} {'min rho':>8} {'max flip%':>10}  ranking"
    print(header)
    print("-" * 130)
    for model_name, res in all_results.items():
        s = res["stats"]["summary"]
        short = model_name if len(model_name) <= 48 else "..." + model_name[-45:]
        print(
            f"{short:<48} {s['max_abs_cohen_d']:>10.3f} "
            f"{s['min_spearman_rho']:>8.3f} {s['max_flip_rate_pct']:>9.1f}%  "
            f"{' > '.join(s['format_ranking_by_mean'])}"
        )

    if failures:
        print(f"\n{'=' * 100}\nFAILURES  ({len(failures)})\n{'=' * 100}")
        for f in failures:
            print(f"  {f['model']:<48} [{f['stage']}] {f['error']}")

    save_partial()
    size_mb = out_path.stat().st_size / 1e6
    print(f"\nSaved {mode} results -> {out_path}  ({size_mb:.1f} MB)")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default="train", choices=["train", "validation"])
    ap.add_argument(
        "--mode", choices=["metadata_only", "with_body", "both"], default="both"
    )
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--limit", type=int, default=None, help="Cap records for testing")
    ap.add_argument(
        "--no-trust-remote-code",
        action="store_true",
        help="Disable trust_remote_code (some vendor models require it)",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help='Force attn_implementation="eager" on model load',
    )
    ap.add_argument(
        "--smoke-test",
        action="store_true",
        help="Fail-fast mode: 10 records, batch size 4. Validates the pipeline "
        "for every model without producing meaningful statistics.",
    )
    return ap


def main() -> None:
    """Run the requested modes over the requested models."""
    args = build_parser().parse_args()

    if args.smoke_test:
        args.limit = SMOKE_TEST_RECORDS
        args.batch_size = SMOKE_TEST_BATCH
        print(f"\n{'=' * 70}")
        print(
            f"SMOKE TEST MODE  -  {SMOKE_TEST_RECORDS} records, "
            f"batch size {SMOKE_TEST_BATCH}"
        )
        print("Results are not statistically meaningful.")
        print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    trust = not args.no_trust_remote_code
    print(f"Device: {device}")
    print(f"trust_remote_code: {trust}")

    parsed_path = args.in_dir / f"parsed_{args.split}.json"
    print(f"\nLoading {parsed_path}...")
    data = json.loads(parsed_path.read_text())
    records = [r for r in data["records"] if not r["quality_flags"]]
    if args.limit:
        records = records[: args.limit]
    print(f"  {len(records):,} quality-passed records")

    print(f"\nLoading tokenizers for {len(args.models)} models...")
    tokenizers = {}
    for m in args.models:
        tok = try_load_tokenizer(m, trust)
        if tok is not None:
            tokenizers[m] = tok
            print(f"  ok {m}")
    if not tokenizers:
        raise SystemExit("No tokenizers loaded, cannot proceed.")

    modes = ["metadata_only", "with_body"] if args.mode == "both" else [args.mode]

    for mode in modes:
        if mode == "with_body":
            print(
                "\nPreparing with_body records (body budget across all tokenizers)..."
            )
            eligible, stats = prepare_records_with_body(records, tokenizers)
            tightest_counts = stats["tightest_counts"]
            drop_stats = {"dropped": stats["dropped"], **budget_summary(eligible)}
            print(
                f"  {len(eligible):,} / {len(records):,} kept  "
                f"({drop_stats['dropped']} dropped, budget < {MIN_BODY_TOKENS})"
            )
            print(
                f"  body budget tokens: min={drop_stats['budget_min']} "
                f"median={drop_stats['budget_median']} max={drop_stats['budget_max']}"
            )
            print("  tightest tokenizer distribution:")
            for t, c in sorted(tightest_counts.items(), key=lambda x: -x[1]):
                print(f"    {c:>5}  {t}")
        else:
            eligible = records
            drop_stats = None
            tightest_counts = None

        if not eligible:
            print(f"  No eligible records for mode {mode}; skipping.")
            continue

        run_mode(
            mode,
            eligible,
            args.models,
            tokenizers,
            args,
            device,
            tightest_counts,
            drop_stats,
        )


if __name__ == "__main__":
    main()
