"""Comparison of a trained arm against the untrained baseline."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
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
    n_boot: int = DEFAULT_N_BOOT,
) -> tuple[float, float, float]:
    """Compare ranking quality before and after training, query by query.

    Args:
        rr_base: The baseline reciprocal ranks.
        rr_trained: The trained reciprocal ranks, for the same queries.
        seed: The seed for the resample.
        n_boot: The number of replicates.

    Returns:
        The mean change, and the lower and upper bounds of its interval.
    """
    delta = rr_trained - rr_base
    low, high = bootstrap_ci_of_mean(delta, n_boot=n_boot, seed=seed)
    return float(delta.mean()), low, high


def bootstrap_delta_mrr(
    rr_base: np.ndarray,
    rr_trained: np.ndarray,
    seed: int = DEFAULT_SEED,
    n_boot: int = DEFAULT_N_BOOT,
) -> dict[str, Any]:
    """Report the change in ranking quality as a section of an output.

    Args:
        rr_base: The baseline reciprocal ranks.
        rr_trained: The trained reciprocal ranks, for the same queries.
        seed: The seed for the resample.
        n_boot: The number of replicates.

    Returns:
        The mean change and its interval.
    """
    mean, low, high = delta_mrr_ci(rr_base, rr_trained, seed, n_boot)
    return {"delta_mean": mean, "delta_ci": [low, high]}


ALL_FORMATS = "all_formats"
IN_TRAINING = "in_training"
OOD = "ood"
SUBSET_NAMES = (ALL_FORMATS, IN_TRAINING, OOD)
MIN_SUBSET_RECORDS = 20
RATIO_EPSILON = 1e-6


def pair_subsets(held_out: str) -> dict[str, list[tuple[str, str]]]:
    """Split the format pairs by whether they involve the held-out format.

    Args:
        held_out: The format withheld from training.

    Returns:
        Every pair, the pairs between training formats, and the pairs that
        involve the held-out format.
    """
    every = list(combinations(FORMAT_NAMES, 2))
    return {
        ALL_FORMATS: every,
        IN_TRAINING: [p for p in every if held_out not in p],
        OOD: [p for p in every if held_out in p],
    }


def bands_crossed(baseline: float, trained: float) -> int:
    """Count the size bands training moved an effect across.

    Args:
        baseline: The effect size before training.
        trained: The effect size after training.

    Returns:
        The number of bands.
    """
    order = [name for _, name in BANDS] + ["trivial"]
    return order.index(cohen_band(baseline)) - order.index(cohen_band(trained))


def max_d_section(
    base_scores: ScoresPerFormat,
    trained_scores: ScoresPerFormat,
    pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    """Report the largest absolute effect of one pair subset, before and after.

    Args:
        base_scores: The per-record scores of the untrained model.
        trained_scores: The per-record scores of the trained arm.
        pairs: The format pairs of the subset.

    Returns:
        Both effect sizes, their bands, and the pair that gives each.
    """
    baseline, base_pair = max_abs_d_over_pairs(base_scores, pairs)
    trained, trained_pair = max_abs_d_over_pairs(trained_scores, pairs)
    return {
        "baseline": baseline,
        "trained": trained,
        "baseline_band": cohen_band(baseline),
        "trained_band": cohen_band(trained),
        "baseline_pair": base_pair,
        "trained_pair": trained_pair,
    }


def relative_change(baseline: float, trained: float) -> float | None:
    """Return the change as a share of the baseline, or None at zero."""
    return (trained - baseline) / baseline if baseline > 0 else None


def slice_scores(
    scores_per_fmt: ScoresPerFormat, positions: Sequence[int]
) -> dict[str, list[float]]:
    """Keep only the records at the given positions, in every format.

    Args:
        scores_per_fmt: The per-record scores, by format name.
        positions: The record positions to keep.

    Returns:
        The reduced scores, by format name.
    """
    return {name: [scores_per_fmt[name][i] for i in positions] for name in FORMAT_NAMES}


def held_out_transfer(
    base_guardrail: dict[str, Any],
    trained_guardrail: dict[str, Any],
    held_out: str,
    seed: int = DEFAULT_SEED,
    positions: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare ranking quality on the held-out format against the others.

    Args:
        base_guardrail: The guardrail section of the baseline evaluation.
        trained_guardrail: The guardrail section of the trained evaluation.
        held_out: The format withheld from training.
        seed: The seed for the resample.
        positions: The queries to keep. The default is every query.

    Returns:
        The details of the held-out transfer.
    """

    def ranks(guardrail: dict[str, Any], name: str) -> np.ndarray:
        values = np.asarray(guardrail["per_format_reciprocal_ranks"][name])
        return values if positions is None else values[list(positions)]

    base_held = ranks(base_guardrail, held_out)
    trained_held = ranks(trained_guardrail, held_out)
    delta = float(trained_held.mean() - base_held.mean())
    per_format = {
        name: float(
            ranks(trained_guardrail, name).mean() - ranks(base_guardrail, name).mean()
        )
        for name in FORMAT_NAMES
        if name != held_out
    }
    training_mean = float(np.mean(list(per_format.values())))
    ratio = (
        delta / training_mean if abs(training_mean) > RATIO_EPSILON else float("nan")
    )
    return {
        "held_out_format": held_out,
        "baseline_mrr": float(base_held.mean()),
        "trained_mrr": float(trained_held.mean()),
        "delta_mrr": delta,
        "delta_mrr_ci": bootstrap_delta_mrr(base_held, trained_held, seed)["delta_ci"],
        "training_format_mean_delta_mrr": training_mean,
        "training_format_delta_mrr_per_fmt": per_format,
        "transfer_ratio": ratio,
    }


def metadata_only_subset(
    base: dict[str, Any],
    trained: dict[str, Any],
    held_out: str,
    metadata_only: set[str],
    pairs: dict[str, list[tuple[str, str]]],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any] | None:
    """Measure the arm on the records whose answer is only in the metadata.

    Args:
        base: The baseline evaluation.
        trained: The trained evaluation.
        held_out: The format withheld from training.
        metadata_only: The identifiers of the metadata-only records.
        pairs: The pair subsets, from `pair_subsets`.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        The subset statistics, or None.
    """
    positions = [i for i, rid in enumerate(base["record_ids"]) if rid in metadata_only]
    if len(positions) < MIN_SUBSET_RECORDS:
        return None
    base_scores = slice_scores(base["scores_per_fmt"], positions)
    trained_scores = slice_scores(trained["scores_per_fmt"], positions)
    return {
        "n_records": len(positions),
        "point_estimates_max_d": {
            name: max_d_section(base_scores, trained_scores, pairs[name])
            for name in (IN_TRAINING, OOD)
        },
        "ood_bootstrap_delta_max_d": bootstrap_delta_max_d(
            base_scores, trained_scores, pairs[OOD], n_boot=n_boot, seed=seed
        ),
        "mrr": _subset_mrr(base, trained, held_out, metadata_only, seed),
    }


def _subset_mrr(
    base: dict[str, Any],
    trained: dict[str, Any],
    held_out: str,
    metadata_only: set[str],
    seed: int,
) -> dict[str, Any] | None:
    """Measure ranking quality on the metadata-only records.

    Args:
        base: The baseline evaluation.
        trained: The trained evaluation.
        held_out: The format withheld from training.
        metadata_only: The identifiers of the metadata-only records.
        seed: The seed for the resample.

    Returns:
        The overall and held-out changes, or None.
    """
    base_guardrail = base.get("mrr_guardrail")
    trained_guardrail = trained.get("mrr_guardrail")
    if not base_guardrail or not trained_guardrail:
        return None
    positions = [
        i for i, rid in enumerate(base_guardrail["record_ids"]) if rid in metadata_only
    ]
    if len(positions) < MIN_SUBSET_RECORDS:
        return None
    rr_base = mean_mrr_per_query(base_guardrail)[positions]
    rr_trained = mean_mrr_per_query(trained_guardrail)[positions]
    return {
        "n_records": len(positions),
        "overall_delta_mrr_bootstrap": bootstrap_delta_mrr(rr_base, rr_trained, seed),
        "heldout_transfer": held_out_transfer(
            base_guardrail, trained_guardrail, held_out, seed, positions
        ),
    }


def compare(
    base: dict[str, Any],
    trained: dict[str, Any],
    held_out: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    metadata_only: set[str] | None = None,
) -> dict[str, Any]:
    """Compare a trained arm against the untrained baseline on one split.

    Args:
        base: The baseline evaluation.
        trained: The trained evaluation.
        held_out: The format withheld from training.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        metadata_only: The identifiers of the records whose answer appears
            only in the metadata. None skips that subset.

    Returns:
        The details of the comparison.

    Raises:
        ValueError: If the two evaluations cover different records.
    """
    if base["n_records_kept"] != trained["n_records_kept"]:
        raise ValueError(
            f"baseline kept {base['n_records_kept']} records, "
            f"trained kept {trained['n_records_kept']}"
        )
    pairs = pair_subsets(held_out)
    base_scores = base["scores_per_fmt"]
    trained_scores = trained["scores_per_fmt"]

    points = {
        name: max_d_section(base_scores, trained_scores, pairs[name])
        for name in SUBSET_NAMES
    }
    intervals = {
        name: bootstrap_delta_max_d(
            base_scores, trained_scores, pairs[name], n_boot=n_boot, seed=seed
        )
        for name in SUBSET_NAMES
    }
    in_training = points[IN_TRAINING]["trained"]
    ood = points[OOD]

    return {
        "held_out_format": held_out,
        "n_records": base["n_records_kept"],
        "per_pair_delta": all_pair_deltas(
            base_scores, trained_scores, pairs[ALL_FORMATS]
        ),
        "point_estimates_max_d": points,
        "bootstrap_delta_max_d": intervals,
        "mrr_non_inferiority": _non_inferiority(base, trained, seed),
        "in_training_verdict": {
            "primary_threshold": PRIMARY_THRESHOLD,
            "stretch_threshold": STRETCH_THRESHOLD,
            "primary_pass": in_training < PRIMARY_THRESHOLD,
            "stretch_pass": in_training < STRETCH_THRESHOLD,
            "statistical_floor_pass": intervals[IN_TRAINING]["delta_ci"][1] < 0,
        },
        "ood_descriptive": {
            "baseline_max_d": ood["baseline"],
            "baseline_band": ood["baseline_band"],
            "trained_max_d": ood["trained"],
            "trained_band": ood["trained_band"],
            "cohen_bands_crossed": bands_crossed(ood["baseline"], ood["trained"]),
            "delta_mean": intervals[OOD]["delta_mean"],
            "delta_ci": intervals[OOD]["delta_ci"],
            "relative_reduction": relative_change(ood["baseline"], ood["trained"]),
        },
        "heldout_transfer": (
            held_out_transfer(
                base["mrr_guardrail"], trained["mrr_guardrail"], held_out, seed
            )
            if base.get("mrr_guardrail") and trained.get("mrr_guardrail")
            else None
        ),
        "metadata_only_subset": (
            metadata_only_subset(
                base, trained, held_out, metadata_only, pairs, n_boot, seed
            )
            if metadata_only
            else None
        ),
    }


def _non_inferiority(
    base: dict[str, Any], trained: dict[str, Any], seed: int
) -> dict[str, Any]:
    """Test whether the arm kept ranking quality.

    Args:
        base: The baseline evaluation.
        trained: The trained evaluation.
        seed: The seed for the resample.

    Returns:
        The two mean reciprocal ranks, the change with its interval, and
        whether the change stays inside the margin. The values are None when
        either evaluation carries no guardrail.

    Raises:
        ValueError: If the two guardrails cover different query counts.
    """
    empty = {
        "margin": NI_MARGIN,
        "baseline_mean_mrr": None,
        "trained_mean_mrr": None,
        "delta_mrr_bootstrap": None,
        "passed": None,
    }
    if not base.get("mrr_guardrail") or not trained.get("mrr_guardrail"):
        return empty
    rr_base = mean_mrr_per_query(base["mrr_guardrail"])
    rr_trained = mean_mrr_per_query(trained["mrr_guardrail"])
    if len(rr_base) != len(rr_trained):
        raise ValueError(
            f"the guardrails cover {len(rr_base)} and {len(rr_trained)} queries"
        )
    boot = bootstrap_delta_mrr(rr_base, rr_trained, seed)
    return {
        "margin": NI_MARGIN,
        "baseline_mean_mrr": float(rr_base.mean()),
        "trained_mean_mrr": float(rr_trained.mean()),
        "delta_mrr_bootstrap": boot,
        "passed": boot["delta_ci"][0] > -NI_MARGIN,
    }


def paired_reciprocal_ranks(
    base: dict[str, Any], trained: dict[str, Any]
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Line up two rankings on the records they share.

    Args:
        base: A ranking result holding `record_ids` and `reciprocal_ranks`.
        trained: The ranking result to compare it against.

    Returns:
        The reciprocal-rank arrays aligned to the shared record identifiers.

    Raises:
        ValueError: If the two share no record.
    """
    base_by_id = dict(zip(base["record_ids"], base["reciprocal_ranks"], strict=True))
    trained_by_id = dict(
        zip(trained["record_ids"], trained["reciprocal_ranks"], strict=True)
    )
    shared = [rid for rid in base["record_ids"] if rid in trained_by_id]
    if not shared:
        raise ValueError("the two rankings share no record")
    return (
        shared,
        np.array([base_by_id[rid] for rid in shared]),
        np.array([trained_by_id[rid] for rid in shared]),
    )
