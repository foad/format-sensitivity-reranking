"""Whitening settings the representation study's results are checked under."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from fsr.representation.whitening import DEFAULT_SHRINKAGE

SHRINKAGES = (1e-4, 1e-3, 1e-2)
SPLIT_SEEDS = (0, 1, 2)
LOWEST_MODEL = "bge_base"


@dataclass(frozen=True)
class Setting:
    """One whitening setting.

    Attributes:
        shrinkage: The share of the mean variance added to every direction.
        seed: The seed of the record split.
    """

    shrinkage: float
    seed: int

    @property
    def tag(self) -> str:
        """Return a short name for files and tables."""
        return f"shrink{self.shrinkage:g}_seed{self.seed}"


def settings() -> list[Setting]:
    """Return the grid: every shrinkage at seed 0, the other seeds at the default."""
    out = [Setting(s, 0) for s in SHRINKAGES]
    out += [Setting(DEFAULT_SHRINKAGE, seed) for seed in SPLIT_SEEDS if seed != 0]
    return out


def seed_mean_ratios(
    ratios: Mapping[str, Mapping[str, Sequence[float]]],
) -> dict[str, dict[str, float]]:
    """Return each model's mean ratio over seeds, for each pair set."""
    return {
        model: {name: float(np.mean(values)) for name, values in by_set.items()}
        for model, by_set in ratios.items()
    }


def setting_row(payload: Mapping[str, Any]) -> dict[str, float]:
    """Return one setting's summary statistics.

    Args:
        payload: One setting's results: the separation index of each model,
            the pooled rank correlation, and each model's per-seed ratios.

    Returns:
        The summary statistics for the setting.
    """
    index = payload["separation_index"]
    means = [
        value
        for by_set in seed_mean_ratios(payload["ratios"]).values()
        for value in by_set.values()
    ]
    return {
        "bge_base_separation": float(index[LOWEST_MODEL]),
        "other_lowest_separation": float(
            min(v for k, v in index.items() if k != LOWEST_MODEL)
        ),
        "pooled_rho": float(payload["pooled_rho"]),
        "lowest_ratio": float(min(means)),
        "highest_ratio": float(max(means)),
    }


def range_table(payloads: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Return every setting's summary statistics, one row per setting."""
    rows = {
        f"shrinkage {p['shrinkage']:g}, split {p['split_seed']}": setting_row(p)
        for p in payloads
    }
    return pd.DataFrame.from_dict(rows, orient="index")
