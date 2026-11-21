"""Summaries over training seeds of measures taken per fold of each arm."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

SETS = ("training", "held_out")
UNTRAINED = "untrained"


def seed_values(
    folds: dict[str, Any], spaces: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Return the mean over folds of each space's training and held-out pairs."""
    return {
        space: {s: float(np.mean([f[space][s] for f in folds.values()])) for s in SETS}
        for space in spaces
    }


def spread(values: list[float]) -> dict[str, Any]:
    """Return the mean, sample SD, range and per-seed values of a statistic."""
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
        "per_seed": [float(v) for v in array],
    }


def summarise(
    untrained: dict[str, Any],
    seeds: dict[int, dict[float, dict[str, Any]]],
    spaces: Sequence[str],
) -> dict[str, Any]:
    """Return each condition's values and the ratios between conditions.

    Args:
        untrained: The untrained model's fold summaries, by held-out format.
        seeds: Each seed's fold summaries, by weight then held-out format.
        spaces: The measures to summarise.

    Returns:
        The summary of each condition's values and the ratios between conditions.
    """
    base = seed_values(untrained, spaces)
    per_seed = {
        s: {w: seed_values(f, spaces) for w, f in by_w.items()}
        for s, by_w in seeds.items()
    }
    weights = sorted(next(iter(per_seed.values())))
    low, high = weights[0], weights[-1]
    out: dict[str, Any] = {UNTRAINED: base}
    for w in weights:
        out[f"lambda {w:g}"] = {
            space: {
                s: spread([per_seed[seed][w][space][s] for seed in per_seed])
                for s in SETS
            }
            for space in spaces
        }
    out["winner / lambda 0"] = {
        space: {
            s: spread(
                [
                    per_seed[seed][high][space][s] / per_seed[seed][low][space][s]
                    for seed in per_seed
                ]
            )
            for s in SETS
        }
        for space in spaces
    }
    out["lambda 0 / untrained"] = {
        space: {
            s: spread(
                [per_seed[seed][low][space][s] / base[space][s] for seed in per_seed]
            )
            for s in SETS
        }
        for space in spaces
    }
    return out
