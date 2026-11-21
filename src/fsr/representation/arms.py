"""Trained arms whose representation a capture reads."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import arm, selection_path
from fsr.models.registry import by_slug

WINNER = "winner"
SWEEP = "lambda"


def chosen_weight(data_root: Path, model: str) -> float:
    """Return the invariance weight the selection chose for one model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The weight.

    Raises:
        ValueError: If no selection is present or it chose no weight.
    """
    source = by_slug(model).weight_from or model
    path = selection_path(data_root, source, SWEEP)
    if not path.exists():
        raise ValueError(f"no weight selection for {model}: {path}")
    selection = json.loads(path.read_text())
    if selection.get("status") != "selected":
        raise ValueError(f"no weight was chosen for {model}")
    return float(selection[f"winner_{SWEEP}"])


def trained_arms(
    data_root: Path,
    model: str,
    weights: Sequence[str],
    seeds: Sequence[int],
) -> list[str]:
    """Return the held-out-format arms for every seed and weight.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        weights: Invariance weights, each a number or `winner`.
        seeds: Training seeds.

    Returns:
        The arm names, seed-major, then weight, then held-out format.

    Raises:
        ValueError: If `winner` is named and no weight was chosen.
    """
    values = [
        chosen_weight(data_root, model) if w == WINNER else float(w) for w in weights
    ]
    return [
        arm(fold, value, seed=seed)
        for seed in seeds
        for value in values
        for fold in FORMAT_NAMES
    ]


def main(argv: Sequence[str] | None = None) -> int:
    """Print the trained arms of one model, one per line.

    Args:
        argv: Command-line arguments. Defaults to the process arguments.

    Returns:
        The exit code. 2 when `winner` is named and no weight was chosen.
    """
    ap = argparse.ArgumentParser(description="List the trained arms to capture.")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args(argv)
    try:
        arms = trained_arms(args.data_root, args.model, args.weights, args.seeds)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for name in arms:
        print(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
