"""Test whether an arm kept its ranking quality on ordinary prose.

Checks Mean Reciprocal Rank (MRR) between the baseline and the trained arm.

Writes `h2/comparison/{model}_prose_{arm}.json`.
"""

from __future__ import annotations

import argparse
import json

from fsr.cli import add_data_root_arg, resolve_model
from fsr.comparison import NI_MARGIN, delta_mrr_ci, paired_reciprocal_ranks
from fsr.h2_layout import BASE_ARM, comparison_path, result_path
from fsr.metrics import DEFAULT_N_BOOT
from fsr.reporting import heading, report_saved

PROSE_SPLIT = "prose"
MRR_AXIS = "mrr"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument("--arm", required=True, help="The trained arm to test")
    ap.add_argument(
        "--n-boot",
        type=int,
        default=DEFAULT_N_BOOT,
        help="Resamples behind the interval",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="Test again over a finished run"
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Compare one arm against the baseline on prose, and write the result."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    out_path = comparison_path(args.data_root, entry.slug, f"prose_{args.arm}")

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} prose {args.arm}: {out_path} is present.")
        return

    paths = {
        name: result_path(args.data_root, PROSE_SPLIT, MRR_AXIS, entry.slug, name)
        for name in (BASE_ARM, args.arm)
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"no prose ranking of {name} at {path}.")

    heading(f"PROSE COMPARE  {entry.label}  {args.arm}", width=70)
    base = json.loads(paths[BASE_ARM].read_text())
    trained = json.loads(paths[args.arm].read_text())

    shared, rr_base, rr_trained = paired_reciprocal_ranks(base, trained)
    if len(shared) < max(len(base["record_ids"]), len(trained["record_ids"])):
        print(
            f"  baseline covers {len(base['record_ids']):,} records, "
            f"the arm covers {len(trained['record_ids']):,}, "
            f"{len(shared):,} are shared"
        )

    mean, low, high = delta_mrr_ci(rr_base, rr_trained, args.seed, args.n_boot)
    kept = low > -NI_MARGIN

    print(f"  paired records: {len(shared):,}")
    print(f"  baseline MRR:   {rr_base.mean():.4f}")
    print(f"  trained MRR:    {rr_trained.mean():.4f}")
    print(f"  change:         {mean:+.4f}  [{low:+.4f}, {high:+.4f}]")
    print(f"  kept, margin {NI_MARGIN}: {'yes' if kept else 'no'}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": entry.slug,
                "arm": args.arm,
                "split": PROSE_SPLIT,
                "baseline_source": paths[BASE_ARM].name,
                "trained_source": paths[args.arm].name,
                "adapter": trained.get("adapter"),
                "n_paired_records": len(shared),
                "baseline_mrr": float(rr_base.mean()),
                "trained_mrr": float(rr_trained.mean()),
                "delta_mrr_mean": mean,
                "delta_mrr_ci": [low, high],
                "n_boot": args.n_boot,
                "ni_margin": NI_MARGIN,
                "ni_pass": kept,
                "record_ids": shared,
                "per_record_delta_rr": (rr_trained - rr_base).tolist(),
            },
            indent=2,
        )
    )
    report_saved(out_path, f"prose {args.arm}")


if __name__ == "__main__":
    main()
