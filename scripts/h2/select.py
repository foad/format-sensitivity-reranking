"""Choose the value of a swept setting.

Writes `h2/selection/{model}_{sweep}.json`. Exits non-zero when no candidate
keeps ranking quality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fsr.cli import add_data_root_arg, resolve_model
from fsr.h2_layout import BASE_ARM, PHASE1_FOLD, h2_dir, result_path, selection_path
from fsr.reporting import NAME_WIDTH, heading, report_saved, table
from fsr.selection import RANK, SWEEPS, Sweep, extract_value, select

HALT = 1


def candidate_paths(data_root: Path, model: str, sweep: Sweep) -> list[Path]:
    """Return the development evaluations one sweep compares.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        sweep: The quantity being swept.

    Returns:
        The evaluation files, ordered by the value each one carries.

    Raises:
        SystemExit: If the rank sweep runs before a weight has been chosen.
    """
    stem = f"dev_cross_{model}_{PHASE1_FOLD}_lam"
    if sweep is RANK:
        chosen = selection_path(data_root, model, "lambda")
        if not chosen.exists():
            raise SystemExit(f"no weight chosen yet: {chosen} is absent.")
        weight = json.loads(chosen.read_text())["winner_lambda"]
        found = list(h2_dir(data_root).glob(f"{stem}{weight}_r*.json"))
    else:
        found = [
            path
            for path in h2_dir(data_root).glob(f"{stem}*.json")
            if not RANK.pattern.search(path.name)
        ]
    return sorted(found, key=lambda path: sweep.parse(extract_value(path, sweep)))


def report(selection: dict, sweep: Sweep) -> None:
    """Print the candidates, the tie, and the winner.

    Args:
        selection: The result of the selection rule.
        sweep: The quantity being swept.
    """
    print(f"baseline mean MRR: {selection['baseline_mean_mrr']:.4f}")
    print(f"baseline max |d|:  {selection['baseline_max_abs_d']:.4f}\n")
    table(
        (
            (sweep.name, 10),
            ("dev max|d|", 11),
            ("dev MRR", 10),
            ("delta MRR", 11),
            ("delta interval", 22),
            ("keeps MRR", 10),
        ),
        [
            [
                row[sweep.name],
                f"{row['dev_max_abs_d']:.4f}",
                f"{row['dev_mean_mrr']:.4f}",
                f"{row['delta_mrr_mean']:+.4f}",
                f"[{row['delta_mrr_ci_lo']:+.4f}, {row['delta_mrr_ci_hi']:+.4f}]",
                "yes" if row["mrr_ni_pass"] else "no",
            ]
            for row in selection["candidates"]
        ],
    )
    missing = [row for row in selection["candidates"] if row["ci_missing"]]
    for row in missing:
        print(
            f"  {Path(row['path']).name} carries no interval on the largest "
            "effect, so it ties only itself."
        )
    if selection["status"] != "selected":
        return
    if selection["tie_broken"]:
        tied = selection[f"tied_{sweep.name}s"]
        print(f"\n{len(tied)} candidates tie: both intervals meet the leader's.")
        print(f"  tied: {', '.join(tied)}")
        print(f"  the sweep takes {selection[f'winner_{sweep.name}']}")
    winner = selection["winner_summary"]
    print(f"\nWinner: {sweep.name} = {selection[f'winner_{sweep.name}']}")
    print(f"  dev max |d|:  {winner['dev_max_abs_d']:.4f}")
    print(f"  dev mean MRR: {winner['dev_mean_mrr']:.4f}")
    print(
        f"  delta MRR:    {winner['delta_mrr_mean']:+.4f}  "
        f"[{winner['delta_mrr_ci_lo']:+.4f}, {winner['delta_mrr_ci_hi']:+.4f}]"
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--sweep",
        default="lambda",
        choices=sorted(SWEEPS),
        help="The setting to choose a value for",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--force", action="store_true", help="Choose again over a finished selection"
    )
    add_data_root_arg(ap)
    return ap


def main() -> int:
    """Choose one value and write the selection.

    Returns:
        The exit code. HALT when no candidate keeps ranking quality.
    """
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    sweep = SWEEPS[args.sweep]
    out_path = selection_path(args.data_root, entry.slug, sweep.name)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {sweep.name}: {out_path} is present.")
        return 0

    baseline = result_path(args.data_root, "dev", "cross", entry.slug, BASE_ARM)
    if not baseline.exists():
        raise SystemExit(f"no baseline evaluation at {baseline}.")
    candidates = candidate_paths(args.data_root, entry.slug, sweep)
    if not candidates:
        raise SystemExit(f"no candidate evaluations for {entry.slug} {sweep.name}.")

    heading(f"SELECT {sweep.name}  {entry.label}", width=70)
    print(f"  baseline:   {baseline.name}")
    print(f"  candidates: {len(candidates)}")
    for path in candidates:
        print(f"    {path.name[:NAME_WIDTH]}")
    print()

    selection = select(baseline, candidates, sweep, args.seed)
    selection["model"] = entry.slug
    report(selection, sweep)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(selection, indent=2))
    report_saved(out_path, sweep.name)

    if selection["status"] != "selected":
        print(
            f"\nNo {sweep.name} keeps ranking quality within the "
            f"{selection['ni_margin']} margin. Phase 2 cannot start.",
            file=sys.stderr,
        )
        return HALT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
