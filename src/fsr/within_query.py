"""Within-query format sensitivity, measured over one query's candidate list.

Every comparison is between two candidates for the same query, with the model
and the candidate set held fixed, so format is the only variable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from typing import Any

import numpy as np
from scipy import stats

from fsr.formats import FORMAT_NAMES
from fsr.metrics import reciprocal_ranks

SCALE_EPSILON = 1e-12
MatricesPerFormat = Mapping[str, Any]


def _present(source: Mapping[str, Any], formats: Sequence[str]) -> list[str]:
    """Return the given formats that the mapping holds, in order."""
    return [f for f in formats if f in source]


def gold_leads(
    matrices_per_fmt: MatricesPerFormat, gold_col: int = 0
) -> dict[str, np.ndarray]:
    """Report, per format, whether the gold passage ranks first for each query.

    Args:
        matrices_per_fmt: The candidate scores of each format, shape
            (n_queries, n_candidates).
        gold_col: The column holding the gold passage.

    Returns:
        A boolean array per format, True where no negative outscores the gold.
    """
    leads = {}
    for fmt, m in matrices_per_fmt.items():
        m = np.asarray(m, dtype=float)
        neg_cols = [c for c in range(m.shape[1]) if c != gold_col]
        leads[fmt] = (m[:, neg_cols] > m[:, [gold_col]]).sum(axis=1) == 0
    return leads


def within_query_rank_stability(
    matrices_per_fmt: MatricesPerFormat,
    gold_col: int = 0,
    formats: Sequence[str] = FORMAT_NAMES,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Measure rank disturbance between format pairs inside each query.

    A candidate pair counts only when both formats order it strictly. A query
    with no such pair has a NaN flip rate and is excluded from the mean.

    Args:
        matrices_per_fmt: The candidate scores of each format.
        gold_col: The column holding the gold passage.
        formats: The format names to pair.

    Returns:
        One entry per format pair, a summary of the extremes, and the per-query
        flip rate and Kendall tau of each pair.
    """
    fmts = _present(matrices_per_fmt, formats)
    mats = {f: np.asarray(matrices_per_fmt[f], dtype=float) for f in fmts}
    n_q, n_c = mats[fmts[0]].shape
    iu, ju = np.triu_indices(n_c, k=1)
    neg_cols = [c for c in range(n_c) if c != gold_col]

    per_pair: list[dict[str, Any]] = []
    per_query: dict[str, Any] = {}
    extremes = {"flip": (0.0, None), "tau": (1.0, None), "top1": (0.0, None)}

    for f1, f2 in combinations(fmts, 2):
        a, b = mats[f1], mats[f2]

        sign_a = np.sign(a[:, iu] - a[:, ju])
        sign_b = np.sign(b[:, iu] - b[:, ju])
        comparable = (sign_a != 0) & (sign_b != 0)
        flipped = (sign_a != sign_b) & comparable
        denom = comparable.sum(axis=1)
        q_flip = 100.0 * flipped.sum(axis=1) / np.maximum(denom, 1)
        q_flip[denom == 0] = np.nan

        q_tau = np.array(
            [float(stats.kendalltau(a[q], b[q]).statistic) for q in range(n_q)]
        )

        rank_a = 1 + (a[:, neg_cols] > a[:, [gold_col]]).sum(axis=1)
        rank_b = 1 + (b[:, neg_cols] > b[:, [gold_col]]).sum(axis=1)
        gold_delta = np.abs(rank_a - rank_b)
        top1_changed = a.argmax(axis=1) != b.argmax(axis=1)

        label = f"{f1} vs {f2}"
        mean_flip = float(np.nanmean(q_flip))
        mean_tau = float(np.nanmean(q_tau))
        pct_top1 = float(100.0 * top1_changed.mean())
        per_pair.append(
            {
                "pair": label,
                "mean_flip_rate_pct": mean_flip,
                "mean_kendall_tau": mean_tau,
                "mean_gold_rank_delta": float(gold_delta.mean()),
                "gold_rank_changed_pct": float(100.0 * (gold_delta > 0).mean()),
                "top1_changed_pct": pct_top1,
            }
        )
        per_query[label] = {
            "flip_rate_pct": q_flip.tolist(),
            "kendall_tau": q_tau.tolist(),
        }

        if mean_flip > extremes["flip"][0]:
            extremes["flip"] = (mean_flip, label)
        if mean_tau < extremes["tau"][0]:
            extremes["tau"] = (mean_tau, label)
        if pct_top1 > extremes["top1"][0]:
            extremes["top1"] = (pct_top1, label)

    summary = {
        "n_queries": int(n_q),
        "n_candidates": int(n_c),
        "max_flip_rate_pct": extremes["flip"][0],
        "max_flip_pair": extremes["flip"][1],
        "min_kendall_tau": extremes["tau"][0],
        "min_tau_pair": extremes["tau"][1],
        "max_top1_changed_pct": extremes["top1"][0],
        "max_top1_pair": extremes["top1"][1],
    }
    return per_pair, summary, per_query


def within_query_mrr(
    matrices_per_fmt: MatricesPerFormat, gold_col: int = 0
) -> dict[str, dict[str, Any]]:
    """Compute the reciprocal rank of the gold passage per format.

    Args:
        matrices_per_fmt: The candidate scores of each format.
        gold_col: The column holding the gold passage.

    Returns:
        The mean reciprocal rank and the per-query reciprocal ranks, by format.
    """
    out = {}
    for fmt, m in matrices_per_fmt.items():
        m = np.asarray(m, dtype=float)
        neg_cols = [c for c in range(m.shape[1]) if c != gold_col]
        rr = reciprocal_ranks(m[:, gold_col], m[:, neg_cols])
        out[fmt] = {"mrr": float(rr.mean()), "reciprocal_ranks": rr.tolist()}
    return out


def score_scale_diagnostic(
    matrices_per_fmt: MatricesPerFormat,
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, float]:
    """Compare the spread a format change causes with the spread within a query.

    Args:
        matrices_per_fmt: The candidate scores of each format.
        formats: The format names to include.

    Returns:
        The mean per-candidate standard deviation across formats, the mean
        standard deviation inside one query's candidate list, and their ratio.
    """
    fmts = _present(matrices_per_fmt, formats)
    stack = np.stack(
        [np.asarray(matrices_per_fmt[f], dtype=float) for f in fmts], axis=0
    )
    delta = float(stack.std(axis=0).mean())
    s_within = float(stack.std(axis=2).mean())
    return {
        "delta": delta,
        "s_within": s_within,
        "ratio": delta / (s_within + SCALE_EPSILON),
    }


def gold_top1_stability(
    top1_per_fmt: Mapping[str, Any],
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, Any]:
    """Split queries by whether the gold leads under every format, none, or some.

    Args:
        top1_per_fmt: A boolean array per format, True where the gold leads.
        formats: The format names to include.

    Returns:
        The counts and shares of each group, the worst pairwise disagreement,
        and the per-format share of queries the gold leads.
    """
    fmts = _present(top1_per_fmt, formats)
    stack = np.stack([np.asarray(top1_per_fmt[f], dtype=bool) for f in fmts])
    n_q = stack.shape[1]
    n_leading = stack.sum(axis=0)
    always = int((n_leading == len(fmts)).sum())
    never = int((n_leading == 0).sum())
    sometimes = int(n_q - always - never)

    worst_pct, worst_pair = 0.0, None
    for f1, f2 in combinations(fmts, 2):
        pct = float(100.0 * (stack[fmts.index(f1)] != stack[fmts.index(f2)]).mean())
        if pct > worst_pct:
            worst_pct, worst_pair = pct, f"{f1} vs {f2}"

    return {
        "n_queries": n_q,
        "gold_top1_all_formats": always,
        "gold_top1_all_formats_pct": float(100.0 * always / n_q),
        "gold_top1_format_dependent": sometimes,
        "gold_top1_format_dependent_pct": float(100.0 * sometimes / n_q),
        "gold_top1_never": never,
        "gold_top1_never_pct": float(100.0 * never / n_q),
        "worst_pair_disagreement_pct": worst_pct,
        "worst_pair": worst_pair,
        "per_format_top1_pct": {
            f: float(100.0 * stack[fmts.index(f)].mean()) for f in fmts
        },
    }


def gold_top1_from_matrices(
    matrices_per_fmt: MatricesPerFormat,
    gold_col: int = 0,
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, Any]:
    """Run gold_top1_stability on candidate score matrices."""
    return gold_top1_stability(gold_leads(matrices_per_fmt, gold_col), formats)


def conditional_inconsistency(
    top1_per_fmt: Mapping[str, Any],
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, Any]:
    """Measure format inconsistency among the queries a model can answer.

    A query is answerable when the gold leads under at least one format. It is
    inconsistent when it leads under some formats but not all.

    Args:
        top1_per_fmt: A boolean array per format, True where the gold leads.
        formats: The format names to include.

    Returns:
        The answerable count, the inconsistent count, the inconsistency share,
        and the answerable share. The share is NaN when nothing is answerable.
    """
    fmts = _present(top1_per_fmt, formats)
    stack = np.stack([np.asarray(top1_per_fmt[f], dtype=bool) for f in fmts])
    n_leading = stack.sum(axis=0)
    n_answerable = int((n_leading > 0).sum())
    if n_answerable == 0:
        return {
            "n_answerable": 0,
            "inconsistent": 0,
            "inconsistency_pct": float("nan"),
            "answerable_pct": 0.0,
        }
    inconsistent = int(((n_leading > 0) & (n_leading < len(fmts))).sum())
    return {
        "n_answerable": n_answerable,
        "inconsistent": inconsistent,
        "inconsistency_pct": float(100.0 * inconsistent / n_answerable),
        "answerable_pct": float(100.0 * n_answerable / stack.shape[1]),
    }


def conditional_inconsistency_from_matrices(
    matrices_per_fmt: MatricesPerFormat,
    gold_col: int = 0,
    formats: Sequence[str] = FORMAT_NAMES,
) -> dict[str, Any]:
    """Run conditional_inconsistency on candidate score matrices."""
    return conditional_inconsistency(gold_leads(matrices_per_fmt, gold_col), formats)
