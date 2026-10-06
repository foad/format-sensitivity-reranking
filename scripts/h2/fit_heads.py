"""Fit every head of the probe to one model's cached representations.

Writes `h2/head_probe/frontier_{model}.json`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from fsr.cli import add_data_root_arg, resolve_model
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import frontier_path
from fsr.head_probe.frontier import frontier_point
from fsr.head_probe.heads import HEAD_NAMES, build_head, parameter_count
from fsr.head_probe.store import load_store
from fsr.head_probe.train import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_LR,
    DEFAULT_MAX_STEPS,
    DEFAULT_WARMUP_STEPS,
    HeadProbeConfig,
    train_head,
)
from fsr.reporting import ProgressCounter, heading, report_elapsed, report_saved
from fsr.training.data import DEFAULT_TRAIN_NEGATIVES

DEFAULT_LAMBDAS = (0.0, 0.01, 0.1, 1.0, 10.0)
DEFAULT_SEEDS = (0, 1, 2)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--heads",
        nargs="+",
        default=list(HEAD_NAMES),
        choices=list(HEAD_NAMES),
        help="The head architectures to fit",
    )
    ap.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=list(DEFAULT_LAMBDAS),
        help="The invariance weights to sweep",
    )
    ap.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS), help="Fit seeds"
    )
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--neg-k", type=int, default=DEFAULT_TRAIN_NEGATIVES)
    ap.add_argument("--progress-file", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Redo a present frontier")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Sweep the heads of one model and write the frontier."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    out_path = frontier_path(args.data_root, entry.slug)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug}: {out_path} is present.")
        return

    heading(f"HEAD PROBE  {entry.label}", width=70)
    fit_store = load_store(args.data_root, "train", entry.slug)
    eval_store = load_store(args.data_root, "dev", entry.slug)
    formats = list(FORMAT_NAMES)
    print(f"  fit on:      {len(fit_store):,} train records")
    print(f"  measure on:  {len(eval_store):,} dev records")
    print(f"  heads:       {' '.join(args.heads)}")
    print(f"  weights:     {' '.join(str(w) for w in args.lambdas)}")
    print(f"  seeds:       {' '.join(str(s) for s in args.seeds)}")

    total = len(args.heads) * len(args.lambdas) * len(args.seeds)
    counter = ProgressCounter(total, args.progress_file)
    started = time.time()
    rows = []
    failures = []

    for head_name in args.heads:
        for lambda_inv in args.lambdas:
            for seed in args.seeds:
                label = f"{head_name}/lam{lambda_inv:g}/seed{seed}"
                config = HeadProbeConfig(
                    lambda_inv=lambda_inv,
                    max_steps=args.max_steps,
                    batch_size=args.batch_size,
                    n_negatives=args.neg_k,
                    lr=args.lr,
                    warmup_steps=args.warmup_steps,
                    seed=seed,
                )
                head = build_head(head_name, fit_store.dim, seed=seed)
                try:
                    fit = train_head(head, fit_store, config, formats)
                    point = frontier_point(
                        head, eval_store, formats, args.neg_k, seed=seed
                    )
                except Exception as error:
                    print(f"  x {label}: {type(error).__name__}: {error}")
                    failures.append({"point": label, "error": str(error)})
                    counter.fail()
                    continue
                rows.append(
                    {
                        "head": head_name,
                        "lambda_inv": lambda_inv,
                        "seed": seed,
                        "parameters": parameter_count(head),
                        "max_abs_cohen_d": point["max_abs_cohen_d"],
                        "mean_mrr": point["mean_mrr"],
                        "min_mrr": point["min_mrr"],
                        "mrr_per_format": point["mrr_per_format"],
                        "final_loss": fit.loss,
                        "final_rank_loss": fit.rank_loss,
                        "final_inv_loss": fit.inv_loss,
                        "history": fit.history,
                    }
                )
                print(
                    f"  {label:<28} max|d|={point['max_abs_cohen_d']:.4f}  "
                    f"MRR={point['mean_mrr']:.4f}"
                )
                counter.step(label)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": entry.slug,
                "model_id": entry.model_id,
                "fit_split": fit_store.split,
                "eval_split": eval_store.split,
                "formats": formats,
                "n_fit_records": len(fit_store),
                "n_eval_records": len(eval_store),
                "feature_dim": fit_store.dim,
                "neg_k": args.neg_k,
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "warmup_steps": args.warmup_steps,
                "points": rows,
                "failures": failures,
            },
            indent=2,
        )
    )
    report_elapsed("head sweep", started)
    report_saved(out_path, f"{entry.slug} frontier")


if __name__ == "__main__":
    main()
