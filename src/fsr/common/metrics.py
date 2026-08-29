"""Format-sensitivity statistics, ranking metrics, and bootstrap intervals."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from .rendering import FORMAT_NAMES

COHEN_D_EPSILON = 1e-12
DEFAULT_SAMPLE_SIZE = 20_000
DEFAULT_N_BOOT = 10_000
DEFAULT_SEED = 0
DEFAULT_CI = 0.95
MIN_RANK_STABILITY_RECORDS = 2

ScoresPerFormat = Mapping[str, Sequence[float]]


def _percentile_ci(samples: np.ndarray, ci: float) -> tuple[float, float]:
    """Return the lower and upper percentile bounds of the samples.

    Args:
        samples: The bootstrap replicates.
        ci: The interval width, such as 0.95.

    Returns:
        The lower bound and the upper bound.
    """
    lo_pct = (1 - ci) / 2 * 100
    hi_pct = 100 - lo_pct
    return float(np.percentile(samples, lo_pct)), float(np.percentile(samples, hi_pct))


def _extreme(
    entries: list[dict[str, Any]], key: str, pick: Any
) -> tuple[float, str | None]:
    """Return the extreme value of a key and the pair that gives it.

    Args:
        entries: The per-pair entries to search.
        key: The entry key to compare.
        pick: The selector, either min or max.

    Returns:
        The extreme value and its pair name. Both are NaN and None when no entry
        has a defined value.
    """
    defined = [e for e in entries if not math.isnan(e[key])]
    if not defined:
        return math.nan, None
    best = pick(defined, key=lambda e: e[key])
    return best[key], best["pair"]


def per_format_summary(scores_per_fmt: ScoresPerFormat) -> dict[str, dict[str, float]]:
    """Summarise the score distribution for each format.

    Args:
        scores_per_fmt: The per-record scores, by format name.

    Returns:
        The mean, the standard deviation, and the median for each format.
    """
    return {
        fmt: {
            "mean": float(np.mean(s)),
            "std": float(np.std(s)),
            "median": float(np.median(s)),
        }
        for fmt, s in scores_per_fmt.items()
    }


def pairwise_cohen_d(
    scores_per_fmt: ScoresPerFormat,
    *,
    formats: Sequence[str] = FORMAT_NAMES,
) -> tuple[list[dict[str, Any]], float, str | None]:
    """Compute the paired Cohen's d for each pair of formats.

    The effect size is the mean of the paired score differences divided by their
    standard deviation. The Wilcoxon signed-rank p-value is 1.0 if every
    difference is zero.

    Args:
        scores_per_fmt: The per-record scores, by format name. The scores must
            be aligned across formats.
        formats: The format names to pair. The default is FORMAT_NAMES.

    Returns:
        One entry for each pair, the largest absolute effect size, and the pair
        that gives it. The pair is None when formats has fewer than two names.
    """
    out = []
    max_abs_d = 0.0
    max_d_pair = None
    for f1, f2 in combinations(formats, 2):
        a = np.array(scores_per_fmt[f1])
        b = np.array(scores_per_fmt[f2])
        diff = a - b
        d = float(diff.mean() / (diff.std() + COHEN_D_EPSILON))
        if not diff.any():
            wp = 1.0
        else:
            try:
                wp = float(stats.wilcoxon(a, b).pvalue)
            except ValueError:
                wp = 1.0
        out.append(
            {
                "pair": f"{f1} - {f2}",
                "mean_diff": float(diff.mean()),
                "cohen_d": d,
                "wilcoxon_p": wp,
            }
        )
        if abs(d) > max_abs_d:
            max_abs_d = abs(d)
            max_d_pair = f"{f1} - {f2}"
    return out, max_abs_d, max_d_pair


def rank_stability(
    scores_per_fmt: ScoresPerFormat,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    *,
    formats: Sequence[str] = FORMAT_NAMES,
) -> tuple[list[dict[str, Any]], float, str | None, float, str | None]:
    """Measure the rank agreement between each pair of formats.

    The flip rate is the percentage of sampled record pairs that the two formats
    put in the opposite order. The sample excludes a record paired with itself,
    and it excludes a pair that either format scores equally.

    Spearman rho is NaN for a format whose scores are all equal. Such a pair is
    excluded from the reported minimum.

    Args:
        scores_per_fmt: The per-record scores, by format name.
        sample_size: The number of record pairs to sample.
        seed: The seed for the record-pair sample.
        formats: The format names to pair. The default is FORMAT_NAMES.

    Returns:
        One entry for each pair, the smallest Spearman rho with the pair that
        gives it, and the largest flip rate with the pair that gives it. An
        extreme is NaN and its pair is None when no pair has a defined value.

    Raises:
        ValueError: If the scores hold fewer than MIN_RANK_STABILITY_RECORDS
            records. A single record yields no orderable pair to sample.
    """
    n = len(next(iter(scores_per_fmt.values())))
    if n < MIN_RANK_STABILITY_RECORDS:
        raise ValueError(
            f"rank stability needs at least {MIN_RANK_STABILITY_RECORDS} records, "
            f"got {n}"
        )
    rng = np.random.default_rng(seed)
    i_arr = rng.integers(0, n, sample_size)
    j_arr = rng.integers(0, n, sample_size)
    mask = i_arr != j_arr
    i_arr, j_arr = i_arr[mask], j_arr[mask]

    out = []
    for f1, f2 in combinations(formats, 2):
        a = np.array(scores_per_fmt[f1])
        b = np.array(scores_per_fmt[f2])
        rho = float(stats.spearmanr(a, b).statistic)
        sign1 = np.sign(a[i_arr] - a[j_arr])
        sign2 = np.sign(b[i_arr] - b[j_arr])
        flips = int(np.sum((sign1 != sign2) & (sign1 != 0) & (sign2 != 0)))
        flip_rate = 100.0 * flips / len(sign1)
        out.append(
            {"pair": f"{f1} vs {f2}", "spearman_rho": rho, "flip_rate_pct": flip_rate}
        )
    min_rho, min_rho_pair = _extreme(out, "spearman_rho", min)
    max_flip, max_flip_pair = _extreme(out, "flip_rate_pct", max)
    return out, min_rho, min_rho_pair, max_flip, max_flip_pair


def format_sensitivity_summary(
    scores_per_fmt: ScoresPerFormat,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    *,
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, Any]:
    """Combine the per-format, pairwise, and rank-stability statistics.

    Args:
        scores_per_fmt: The per-record scores, by format name.
        sample_size: The number of record pairs to sample for rank stability.
        seed: The seed for the record-pair sample.
        formats: The format names to pair. The default is FORMAT_NAMES.

    Returns:
        The per_format, pairwise, and rank sections, and a summary section that
        holds the extreme values and the formats ordered by descending mean.
    """
    per_fmt = per_format_summary(scores_per_fmt)
    pairwise, max_d, max_d_pair = pairwise_cohen_d(scores_per_fmt, formats=formats)
    ranks, min_rho, min_rho_pair, max_flip, max_flip_pair = rank_stability(
        scores_per_fmt, sample_size, seed, formats=formats
    )
    fmt_by_mean = sorted(per_fmt.keys(), key=lambda f: -per_fmt[f]["mean"])
    mean_abs_d = float(np.mean([abs(p["cohen_d"]) for p in pairwise]))
    return {
        "per_format": per_fmt,
        "pairwise": pairwise,
        "rank": ranks,
        "summary": {
            "max_abs_cohen_d": max_d,
            "max_d_pair": max_d_pair,
            "mean_abs_cohen_d": mean_abs_d,
            "min_spearman_rho": min_rho,
            "min_rho_pair": min_rho_pair,
            "max_flip_rate_pct": max_flip,
            "max_flip_pair": max_flip_pair,
            "format_ranking_by_mean": fmt_by_mean,
        },
    }


def reciprocal_ranks(
    gold_scores: Sequence[float],
    neg_scores_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """Compute the reciprocal rank of the gold passage for each query.

    The rank counts only the negatives that score strictly higher. A tie
    therefore counts in the gold passage's favour.

    Args:
        gold_scores: The gold score for each query, shape (n_queries,).
        neg_scores_matrix: The negative scores for each query, shape
            (n_queries, n_negatives).

    Returns:
        The reciprocal rank for each query, shape (n_queries,).
    """
    gold = np.asarray(gold_scores)
    negs = np.asarray(neg_scores_matrix)
    rank = 1 + np.sum(negs > gold[:, None], axis=1)
    return 1.0 / rank


def bootstrap_ci_of_mean(
    values: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the mean.

    Args:
        values: The observations to resample.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The lower bound and the upper bound.
    """
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    idx = rng.integers(0, n, size=(n_boot, n))
    for i in range(n_boot):
        means[i] = values[idx[i]].mean()
    return _percentile_ci(means, ci)


def bootstrap_ci_of_max_abs_d(
    scores_per_fmt: ScoresPerFormat,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    *,
    formats: Sequence[str] = FORMAT_NAMES,
) -> tuple[float, float]:
    """Compute a bootstrap confidence interval for the largest absolute Cohen's d.

    Each replicate resamples the record indices and recomputes the largest
    absolute effect size across the format pairs.

    Args:
        scores_per_fmt: The per-record scores, by format name.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        ci: The interval width.
        formats: The format names to pair. The default is FORMAT_NAMES.

    Returns:
        The lower bound and the upper bound.
    """
    scores = {f: np.asarray(s) for f, s in scores_per_fmt.items()}
    n = len(next(iter(scores.values())))
    pairs = list(combinations(formats, 2))
    rng = np.random.default_rng(seed)
    stats_boot = np.empty(n_boot)
    idx_matrix = rng.integers(0, n, size=(n_boot, n))
    for b in range(n_boot):
        idx = idx_matrix[b]
        max_d = 0.0
        for f1, f2 in pairs:
            a = scores[f1][idx]
            bb = scores[f2][idx]
            diff = a - bb
            d = abs(diff.mean() / (diff.std() + COHEN_D_EPSILON))
            max_d = max(max_d, d)
        stats_boot[b] = max_d
    return _percentile_ci(stats_boot, ci)
