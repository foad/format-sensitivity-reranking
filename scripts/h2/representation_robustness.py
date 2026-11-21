"""Check the representation study's whitened results under other whitening settings.

Writes `h2/format_geometry/robustness/{setting}.json` and `summary.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from fsr import h1_results
from fsr.cli import add_data_root_arg
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import arm, h2_dir
from fsr.h2_results import TRAINING_SEEDS, load_selection, winner_weight
from fsr.models.registry import BASE_MODELS, by_slug
from fsr.reporting import ProgressCounter, heading, report_saved
from fsr.representation.plane import pair_separation, plane_view
from fsr.representation.robustness import range_table, settings
from fsr.representation.seeds import SETS
from fsr.representation.separation import split_pairs
from fsr.representation.store import load_store

SPLIT = "dev"
H1_SPLIT = "test"
OUT_SUBDIR = ("format_geometry", "robustness")


def h1_pair_d(data_root: Path) -> dict[tuple[str, str, str], float]:
    """Return H1's |d_z| for each model and format pair, keyed both ways round."""
    table_ = h1_results.pairwise_table(
        h1_results.load_axis(data_root, H1_SPLIT, "cross", "with_body")
    )
    out = {}
    for row in table_.itertuples():
        out[(row.model, row.left, row.right)] = row.abs_d
        out[(row.model, row.right, row.left)] = row.abs_d
    return out


def pair_values(matrix: np.ndarray, formats: list[str]) -> dict[tuple[str, str], float]:
    """Return the upper triangle of a format-pair matrix, keyed by pair."""
    return {
        (formats[a], formats[b]): float(matrix[a, b])
        for a in range(len(formats))
        for b in range(a + 1, len(formats))
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=[m.slug for m in BASE_MODELS])
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
    """Evaluate every setting and write the results."""
    args = build_parser().parse_args()
    out_dir = h2_dir(args.data_root).joinpath(*OUT_SUBDIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = settings()
    h1 = h1_pair_d(args.data_root)
    winners = {s: winner_weight(load_selection(args.data_root, s)) for s in args.models}
    per_model = 1 + len(args.seeds) * 2 * len(FORMAT_NAMES)
    counter = ProgressCounter(len(args.models) * per_model, args.progress_file)

    index: dict[str, dict[str, float]] = {g.tag: {} for g in grid}
    pairs: dict[str, list[tuple[float, float]]] = {g.tag: [] for g in grid}
    folds: dict[str, dict[str, Any]] = {g.tag: {} for g in grid}
    for slug in args.models:
        label = by_slug(slug).label
        store = load_store(args.data_root, SPLIT, slug)
        formats = store.formats
        for g in grid:
            view = plane_view(
                store.features, seed=g.seed, whiten=True, shrinkage=g.shrinkage
            )
            index[g.tag][slug] = view["separation"]
            matrix = pair_separation(
                store.features, seed=g.seed, whiten=True, shrinkage=g.shrinkage
            )
            for (a, b), value in pair_values(matrix, formats).items():
                pairs[g.tag].append((value, h1[(label, a, b)]))
        counter.step(f"{slug}/untrained")
        for seed in args.seeds:
            for weight in (0.0, winners[slug]):
                for fold in FORMAT_NAMES:
                    name = arm(fold, weight, seed=seed)
                    trained = load_store(
                        args.data_root, SPLIT, slug, name, args.feature_dir
                    )
                    for g in grid:
                        matrix = pair_separation(
                            trained.features,
                            seed=g.seed,
                            whiten=True,
                            shrinkage=g.shrinkage,
                        )
                        split = split_pairs(matrix, formats, fold)
                        folds[g.tag].setdefault(slug, {}).setdefault(
                            seed, {}
                        ).setdefault(weight, []).append(split)
                    counter.step(f"{slug}/{name}")

    summary = []
    for g in grid:
        ratios: dict[str, dict[str, list[float]]] = {}
        for slug, by_seed in folds[g.tag].items():
            low, high = 0.0, winners[slug]
            ratios[slug] = {
                s: [
                    float(np.mean([f[s] for f in by_weight[high]]))
                    / float(np.mean([f[s] for f in by_weight[low]]))
                    for by_weight in by_seed.values()
                ]
                for s in SETS
            }
        values, references = zip(*pairs[g.tag], strict=True)
        rho = float(spearmanr(values, references).statistic)
        payload = {
            "shrinkage": g.shrinkage,
            "split_seed": g.seed,
            "separation_index": index[g.tag],
            "pooled_rho": rho,
            "ratios": ratios,
        }
        out = out_dir / f"{g.tag}.json"
        out.write_text(json.dumps(payload, indent=2))
        summary.append(payload)
    path = out_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    report_saved(path)

    heading("Whitening robustness of the representation study")
    print(range_table(summary).round(3).to_string())


if __name__ == "__main__":
    main()
