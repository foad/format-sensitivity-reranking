"""Measure how far each pair of formats separates the gold passages, per adapter.

Writes `h2/format_geometry/separation/{model}.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fsr.cli import add_data_root_arg
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import arm, h2_dir
from fsr.h2_results import TRAINING_SEEDS, load_selection, winner_weight
from fsr.models.registry import by_slug
from fsr.reporting import ProgressCounter, heading, report_saved, table
from fsr.representation import seeds
from fsr.representation.plane import pair_separation
from fsr.representation.separation import split_pairs
from fsr.representation.store import load_store

SPLIT = "dev"
OUT_SUBDIR = ("format_geometry", "separation")
SPACES = ("raw", "whitened")
UNTRAINED = seeds.UNTRAINED


def separations(features: Any) -> dict[str, Any]:
    """Return the pair separation matrices of one adapter, raw and whitened."""
    return {
        "raw": pair_separation(features),
        "whitened": pair_separation(features, whiten=True),
    }


def folds_of(matrices: dict[str, Any], formats: list[str], fold: str) -> dict:
    """Return the training and held-out pair means of one fold, per space."""
    return {space: split_pairs(m, formats, fold) for space, m in matrices.items()}


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS))
    ap.add_argument(
        "--feature-dir",
        type=Path,
        required=True,
        help="Directory the trained-arm captures are in",
    )
    ap.add_argument("--progress-file", type=Path, default=None)
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Measure every adapter of each model and write the results."""
    args = build_parser().parse_args()
    out_dir = h2_dir(args.data_root).joinpath(*OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    winners = {
        slug: winner_weight(load_selection(args.data_root, slug))
        for slug in args.models
    }
    per_model = 1 + len(args.seeds) * 2 * len(FORMAT_NAMES)
    counter = ProgressCounter(len(args.models) * per_model, args.progress_file)
    for slug in args.models:
        store = load_store(args.data_root, SPLIT, slug)
        formats = store.formats
        base = separations(store.features)
        counter.step(f"{slug}/{UNTRAINED}")
        adapters = {UNTRAINED: {k: v.tolist() for k, v in base.items()}}
        untrained = {f: folds_of(base, formats, f) for f in FORMAT_NAMES}
        seeds: dict[int, dict[float, dict[str, Any]]] = {}
        for seed in args.seeds:
            for weight in (0.0, winners[slug]):
                for fold in FORMAT_NAMES:
                    name = arm(fold, weight, seed=seed)
                    trained = load_store(
                        args.data_root, SPLIT, slug, name, args.feature_dir
                    )
                    matrices = separations(trained.features)
                    adapters[name] = {k: v.tolist() for k, v in matrices.items()}
                    seeds.setdefault(seed, {}).setdefault(weight, {})[fold] = folds_of(
                        matrices, formats, fold
                    )
                    counter.step(f"{slug}/{name}")
        summary = seeds.summarise(untrained, seeds, SPACES)
        out = out_dir / f"{slug}.json"
        out.write_text(
            json.dumps(
                {
                    "model": slug,
                    "split": SPLIT,
                    "formats": formats,
                    "weight": winners[slug],
                    "seeds": args.seeds,
                    "summary": summary,
                    "adapters": adapters,
                },
                indent=2,
            )
        )
        report_saved(out)
        heading(f"{by_slug(slug).label}: gold pair separation, mean over seeds")
        table(
            [("condition", 22), ("space", 10), ("training", 10), ("held_out", 10)],
            [
                (
                    label,
                    space,
                    *(
                        f"{value['mean'] if isinstance(value, dict) else value:.3f}"
                        for value in (summary[label][space][s] for s in seeds.SETS)
                    ),
                )
                for label in summary
                for space in SPACES
            ],
        )


if __name__ == "__main__":
    main()
