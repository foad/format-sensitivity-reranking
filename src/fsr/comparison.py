"""Comparison of a trained arm against the untrained baseline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from fsr.formats import FORMAT_NAMES
from fsr.metrics import (
    COHEN_D_EPSILON,
    DEFAULT_CI,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    ScoresPerFormat,
    _percentile_ci,
    bootstrap_ci_of_mean,
    cohen_d,
)

NI_MARGIN = 0.03
PRIMARY_THRESHOLD = 0.5
STRETCH_THRESHOLD = 0.2

BANDS = ((0.8, "large"), (0.5, "medium"), (0.2, "small"))


def cohen_band(d: float) -> str:
    """Name the conventional size band of an effect.

    Args:
        d: The effect size.

    Returns:
        `large`, `medium`, `small` or `trivial`.
    """
    size = abs(d)
    for floor, name in BANDS:
        if size >= floor:
            return name
    return "trivial"


def max_abs_d_over_pairs(
    scores_per_fmt: ScoresPerFormat,
    pairs: Sequence[tuple[str, str]],
    indices: np.ndarray | None = None,
) -> tuple[float, str | None]:
    """Find the largest absolute effect size over a set of format pairs.

    Args:
        scores_per_fmt: The per-record scores, by format name.
        pairs: The format pairs to compare.
        indices: The records to restrict to. The default is every record.

    Returns:
        The largest absolute effect size and its pair, or None.
    """
    largest = 0.0
    found = None
    for f1, f2 in pairs:
        a = np.asarray(scores_per_fmt[f1])
        b = np.asarray(scores_per_fmt[f2])
        if indices is not None:
            a = a[indices]
            b = b[indices]
        size = abs(cohen_d(a, b))
        if size > largest:
            largest = size
            found = f"{f1} - {f2}"
    return largest, found


def all_pair_deltas(
    base_scores: ScoresPerFormat,
    trained_scores: ScoresPerFormat,
    pairs: Sequence[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Compare each format pair before and after training.

    Args:
        base_scores: The per-record scores of the untrained model.
        trained_scores: The per-record scores of the trained arm.
        pairs: The format pairs to compare.

    Returns:
        One entry per pair, containing the deltas.
    """
    rows = []
    for f1, f2 in pairs:
        base = cohen_d(base_scores[f1], base_scores[f2])
        trained = cohen_d(trained_scores[f1], trained_scores[f2])
        rows.append(
            {
                "pair": f"{f1} - {f2}",
                "d_baseline": base,
                "abs_d_baseline": abs(base),
                "d_trained": trained,
                "abs_d_trained": abs(trained),
                "delta_abs_d": abs(trained) - abs(base),
            }
        )
    return rows


def bootstrap_delta_max_d(
    base_scores: ScoresPerFormat,
    trained_scores: ScoresPerFormat,
    pairs: Sequence[tuple[str, str]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> dict[str, Any]:
    """Compare the largest absolute effect size before and after training.

    Args:
        base_scores: The per-record scores of the untrained model.
        trained_scores: The per-record scores of the trained arm.
        pairs: The format pairs to compare.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The mean and per-run deltas of the largest absolute effect size.
    """
    base = {f: np.asarray(base_scores[f]) for f in FORMAT_NAMES}
    trained = {f: np.asarray(trained_scores[f]) for f in FORMAT_NAMES}
    n = len(next(iter(base.values())))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(n_boot, n))

    deltas = np.empty(n_boot)
    base_boot = np.empty(n_boot)
    trained_boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = draws[b]
        max_base = 0.0
        max_trained = 0.0
        for f1, f2 in pairs:
            diff = base[f1][idx] - base[f2][idx]
            max_base = max(max_base, abs(diff.mean() / (diff.std() + COHEN_D_EPSILON)))
            diff = trained[f1][idx] - trained[f2][idx]
            max_trained = max(
                max_trained, abs(diff.mean() / (diff.std() + COHEN_D_EPSILON))
            )
        base_boot[b] = max_base
        trained_boot[b] = max_trained
        deltas[b] = max_trained - max_base

    return {
        "delta_ci": list(_percentile_ci(deltas, ci)),
        "delta_mean": float(deltas.mean()),
        "baseline_max_d_ci": list(_percentile_ci(base_boot, ci)),
        "trained_max_d_ci": list(_percentile_ci(trained_boot, ci)),
    }


def mean_mrr_per_query(guardrail: dict[str, Any]) -> np.ndarray:
    """Return the reciprocal rank of each query, averaged over the formats.

    Args:
        guardrail: The ranking guardrail section of an evaluation.

    Returns:
        One value per query.
    """
    by_format = guardrail["per_format_reciprocal_ranks"]
    return np.stack(
        [np.asarray(by_format[name]) for name in FORMAT_NAMES], axis=0
    ).mean(axis=0)


def delta_mrr_ci(
    rr_base: np.ndarray,
    rr_trained: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Compare ranking quality before and after training, query by query.

    Args:
        rr_base: The baseline reciprocal ranks.
        rr_trained: The trained reciprocal ranks, for the same queries.
        seed: The seed for the resample.

    Returns:
        The mean change, and the lower and upper bounds of its interval.
    """
    delta = rr_trained - rr_base
    low, high = bootstrap_ci_of_mean(delta, seed=seed)
    return float(delta.mean()), low, high


def bootstrap_delta_mrr(
    rr_base: np.ndarray,
    rr_trained: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Report the change in ranking quality as a section of an output.

    Args:
        rr_base: The baseline reciprocal ranks.
        rr_trained: The trained reciprocal ranks, for the same queries.
        seed: The seed for the resample.

    Returns:
        The mean change and its interval.
    """
    mean, low, high = delta_mrr_ci(rr_base, rr_trained, seed)
    return {"delta_mean": mean, "delta_ci": [low, high]}
