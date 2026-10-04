"""Compare a trained fold against the untrained baseline on one split.

Writes `h2/comparison/{model}_{arm}.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.answer_location import classify_records
from fsr.cli import add_data_root_arg, resolve_model
from fsr.comparison import (
    IN_TRAINING,
    NI_MARGIN,
    OOD,
    PRIMARY_THRESHOLD,
    STRETCH_THRESHOLD,
    SUBSET_NAMES,
    compare,
)
from fsr.corpus.splitting import load_records
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, arm, comparison_path, result_path
from fsr.metrics import DEFAULT_N_BOOT
from fsr.reporting import heading, report_saved, table

METADATA_ONLY = "metadata_only"
EVAL_SPLITS = ("dev", "test")


def metadata_only_ids(data_root: Path, split: str) -> set[str]:
    """Return the records whose short answer appears only in the metadata.

    Args:
        data_root: The corpus directory.
        split: The split the evaluation covered.

    Returns:
        The matching record identifiers.
    """
    labels = classify_records(load_records(data_root, split, verbose=False))
    return {rid for rid, label in labels.items() if label == METADATA_ONLY}


def report(result: dict) -> None:
    """Print the comparison.

    Args:
        result: The result of the comparison.
    """
    held_out = result["held_out_format"]
    print(f"records: {result['n_records']}   held out: {held_out}\n")

    table(
        (
            ("pair", 24),
            ("|d| base", 10),
            ("|d| trained", 12),
            ("change", 9),
            ("subset", 9),
        ),
        [
            [
                row["pair"],
                f"{row['abs_d_baseline']:.3f}",
                f"{row['abs_d_trained']:.3f}",
                f"{row['delta_abs_d']:+.3f}",
                OOD if held_out in row["pair"] else IN_TRAINING,
            ]
            for row in result["per_pair_delta"]
        ],
    )

    print()
    table(
        (
            ("subset", 14),
            ("baseline", 20),
            ("trained", 20),
            ("change", 9),
            ("interval", 20),
        ),
        [
            [
                name,
                f"{point['baseline']:.3f} ({point['baseline_band']})",
                f"{point['trained']:.3f} ({point['trained_band']})",
                f"{point['trained'] - point['baseline']:+.3f}",
                f"[{interval['delta_ci'][0]:+.3f}, {interval['delta_ci'][1]:+.3f}]",
            ]
            for name in SUBSET_NAMES
            for point in [result["point_estimates_max_d"][name]]
            for interval in [result["bootstrap_delta_max_d"][name]]
        ],
    )

    verdict = result["in_training_verdict"]
    print(
        f"\nIn-training thresholds, trained max |d| = "
        f"{result['point_estimates_max_d'][IN_TRAINING]['trained']:.3f}"
    )
    print(f"  below {PRIMARY_THRESHOLD}:  {_mark(verdict['primary_pass'])}")
    print(f"  below {STRETCH_THRESHOLD}:  {_mark(verdict['stretch_pass'])}")
    print(f"  interval upper below zero: {_mark(verdict['statistical_floor_pass'])}")

    descriptive = result["ood_descriptive"]
    print("\nHeld-out pairs, reported without a threshold")
    print(f"  bands moved: {descriptive['cohen_bands_crossed']}")
    if descriptive["relative_reduction"] is not None:
        print(f"  relative change: {descriptive['relative_reduction']:+.1%}")

    guardrail = result["mrr_non_inferiority"]
    print(f"\nRanking guardrail, margin {NI_MARGIN}")
    if guardrail["passed"] is None:
        print("  no guardrail in one of the evaluations")
    else:
        boot = guardrail["delta_mrr_bootstrap"]
        print(f"  baseline MRR: {guardrail['baseline_mean_mrr']:.4f}")
        print(f"  trained MRR:  {guardrail['trained_mean_mrr']:.4f}")
        print(
            f"  change: {boot['delta_mean']:+.4f}  "
            f"[{boot['delta_ci'][0]:+.4f}, {boot['delta_ci'][1]:+.4f}]  "
            f"{_mark(guardrail['passed'])}"
        )

    transfer = result["heldout_transfer"]
    if transfer is not None:
        print(f"\nTransfer to {held_out}")
        print(f"  MRR: {transfer['baseline_mrr']:.4f} -> {transfer['trained_mrr']:.4f}")
        print(
            f"  change: {transfer['delta_mrr']:+.4f}  "
            f"[{transfer['delta_mrr_ci'][0]:+.4f}, {transfer['delta_mrr_ci'][1]:+.4f}]"
        )
        print(
            f"  training formats changed by "
            f"{transfer['training_format_mean_delta_mrr']:+.4f} on average, "
            f"a ratio of {transfer['transfer_ratio']:.2f}"
        )

    subset = result["metadata_only_subset"]
    if subset is None:
        print("\nToo few metadata-only records to report that subset.")
        return
    print(f"\nMetadata-only records: {subset['n_records']}")
    for name in (IN_TRAINING, OOD):
        point = subset["point_estimates_max_d"][name]
        print(
            f"  {name:<12} max |d|: {point['baseline']:.3f} "
            f"({point['baseline_band']}) -> {point['trained']:.3f} "
            f"({point['trained_band']})"
        )


def _mark(passed: bool) -> str:
    """Return the word for a verdict."""
    return "pass" if passed else "fail"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--held-out-format",
        required=True,
        choices=list(FORMAT_NAMES),
        help="The format the fold withheld from training",
    )
    ap.add_argument("--lambda-inv", type=float, required=True)
    ap.add_argument(
        "--rank-tag",
        type=int,
        default=None,
        help="The adapter rank the arm name carries, for a rank sweep",
    )
    ap.add_argument("--split", default="test", choices=list(EVAL_SPLITS))
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--no-metadata-subset",
        action="store_true",
        help="Skip the metadata-only diagnostic",
    )
    ap.add_argument(
        "--force", action="store_true", help="Compare again over a finished run"
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Compare one fold and write the result."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    arm_name = arm(args.held_out_format, args.lambda_inv, args.rank_tag)
    out_path = comparison_path(args.data_root, entry.slug, arm_name)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {arm_name}: {out_path} is present.")
        return

    paths = {
        name: result_path(args.data_root, args.split, "cross", entry.slug, name)
        for name in (BASE_ARM, arm_name)
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"no evaluation of {name} at {path}.")

    heading(f"COMPARE  {entry.label}  {arm_name}  split={args.split}", width=70)
    base = json.loads(paths[BASE_ARM].read_text())
    trained = json.loads(paths[arm_name].read_text())

    metadata_only = (
        set()
        if args.no_metadata_subset
        else metadata_only_ids(args.data_root, args.split)
    )
    result = compare(
        base,
        trained,
        args.held_out_format,
        n_boot=args.n_boot,
        seed=args.seed,
        metadata_only=metadata_only,
    )
    result["model"] = entry.slug
    result["arm"] = arm_name
    result["split"] = args.split
    result["baseline_source"] = str(paths[BASE_ARM])
    result["trained_source"] = str(paths[arm_name])
    report(result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    report_saved(out_path, arm_name)


if __name__ == "__main__":
    main()
