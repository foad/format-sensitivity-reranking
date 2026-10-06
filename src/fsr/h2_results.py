"""H2 result file reader and table builder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsr.comparison import (
    OOD,
    answer_subsets,
    answerable_pool,
    leads_per_format,
)
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import (
    BASE_ARM,
    arm,
    comparison_path,
    result_path,
    selection_path,
)
from fsr.metrics import DEFAULT_CI, DEFAULT_N_BOOT, DEFAULT_SEED
from fsr.models.registry import BASE_MODELS, by_slug

CONTROL_WEIGHT = 0.0
SPLIT = "test"
SWEEP = "lambda"
AXES = ("score", "answer")


def load_selection(data_root: Path, model: str) -> dict[str, Any]:
    """Load the weight selection of one model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The selection.
    """
    return json.loads(selection_path(data_root, model, SWEEP).read_text())


def winner_weight(selection: dict[str, Any]) -> float:
    """Return the invariance weight a selection chose."""
    return float(selection["winner_lambda"])


def load_comparison(
    data_root: Path, model: str, held_out: str, weight: float
) -> dict[str, Any]:
    """Load one arm's comparison against the untrained model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        held_out: The format withheld from training.
        weight: The weight of the invariance term.

    Returns:
        The comparison.
    """
    path = comparison_path(data_root, model, arm(held_out, weight))
    return json.loads(path.read_text())


def load_within(data_root: Path, model: str, arm_name: str) -> dict[str, Any]:
    """Load the answer-axis entry of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The entry of the one model the file holds.

    Raises:
        ValueError: If the file holds results for more than one model.
    """
    path = result_path(data_root, SPLIT, "within", model, arm_name)
    results = json.loads(path.read_text())["results"]
    if len(results) != 1:
        raise ValueError(f"{path.name} holds {len(results)} models, expected 1")
    return next(iter(results.values()))


def arm_paths(data_root: Path, model: str, weight: float) -> list[Path]:
    """Return the comparison file of every fold of one arm."""
    return [
        comparison_path(data_root, model, arm(fold, weight)) for fold in FORMAT_NAMES
    ]


def available(data_root: Path) -> list[str]:
    """Return the core models whose selection and every fold comparison are present.

    Args:
        data_root: The corpus directory.

    Returns:
        The registry slugs, in roster order.
    """
    found = []
    for model in BASE_MODELS:
        if not selection_path(data_root, model.slug, SWEEP).exists():
            continue
        weight = winner_weight(load_selection(data_root, model.slug))
        paths = arm_paths(data_root, model.slug, CONTROL_WEIGHT) + arm_paths(
            data_root, model.slug, weight
        )
        if all(path.exists() for path in paths):
            found.append(model.slug)
    return found


def _frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return a frame indexed by model label, in the order given."""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = [by_slug(slug).label for slug in frame.index]
    frame.index.name = "model"
    return frame


def selection_table(data_root: Path, slugs: Sequence[str]) -> pd.DataFrame:
    """Tabulate the dev score axis of every swept weight.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.

    Returns:
        The untrained maximum absolute Cohen's d, then one column per weight,
        then the weight the selection chose.
    """
    rows: dict[str, dict[str, Any]] = {}
    for slug in slugs:
        selection = load_selection(data_root, slug)
        row: dict[str, Any] = {"baseline": selection["baseline_max_abs_d"]}
        for candidate in sorted(
            selection["candidates"], key=lambda c: float(c["lambda"])
        ):
            row[candidate["lambda"]] = candidate["dev_max_abs_d"]
        row["winner"] = selection["winner_lambda"]
        rows[slug] = row
    return _frame(rows)


def selection_counts(data_root: Path, slugs: Sequence[str]) -> dict[str, Any]:
    """Count the candidates that pass the ranking guardrail.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.

    Returns:
        The passing count, the candidate count, and the labels of the models
        whose winner came from a tie-break.
    """
    passed = 0
    total = 0
    tied = []
    for slug in slugs:
        selection = load_selection(data_root, slug)
        candidates = selection["candidates"]
        total += len(candidates)
        passed += sum(1 for c in candidates if c["mrr_ni_pass"])
        if selection.get("tie_broken"):
            tied.append(by_slug(slug).label)
    return {"passed": passed, "candidates": total, "tie_broken": tied}


def fold_max_d(
    data_root: Path, model: str, weight: float, subset: str = OOD
) -> dict[str, float]:
    """Return the trained score axis of every fold of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        weight: The weight of the invariance term.
        subset: The format pairs to report, one of the comparison subsets.

    Returns:
        The maximum absolute Cohen's d, by held-out format.
    """
    values = {}
    for fold in FORMAT_NAMES:
        comparison = load_comparison(data_root, model, fold, weight)
        section = comparison["score_axis"]["point_estimates_max_d"][subset]
        values[fold] = section["trained"]
    return values


def fold_mean_mrr(data_root: Path, model: str, weight: float) -> dict[str, float]:
    """Return the trained ranking quality of every fold of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        weight: The weight of the invariance term.

    Returns:
        The mean reciprocal rank, by held-out format.
    """
    values = {}
    for fold in FORMAT_NAMES:
        comparison = load_comparison(data_root, model, fold, weight)
        values[fold] = comparison["score_axis"]["mrr_non_inferiority"][
            "trained_mean_mrr"
        ]
    return values


def baseline_max_d(data_root: Path, model: str) -> float:
    """Return the untrained maximum absolute Cohen's d over every format pair."""
    comparison = load_comparison(data_root, model, FORMAT_NAMES[0], CONTROL_WEIGHT)
    return comparison["score_axis"]["point_estimates_max_d"]["all_formats"]["baseline"]


def score_table(
    data_root: Path, slugs: Sequence[str], subset: str = OOD
) -> pd.DataFrame:
    """Summarise the score axis of every model.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        subset: The format pairs to report, one of the comparison subsets.

    Returns:
        The untrained maximum absolute Cohen's d, the fold mean of the control
        and of the treatment, the change, the change as a share of the
        control, the folds that improved, and the mean ranking quality.
    """
    rows = {}
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        control = fold_max_d(data_root, slug, CONTROL_WEIGHT, subset)
        treated = fold_max_d(data_root, slug, weight, subset)
        deltas = np.array([treated[f] - control[f] for f in FORMAT_NAMES])
        control_mean = float(np.mean(list(control.values())))
        rows[slug] = {
            "weight": weight,
            "baseline_max_d": baseline_max_d(data_root, slug),
            "control_max_d": control_mean,
            "trained_max_d": float(np.mean(list(treated.values()))),
            "delta": float(deltas.mean()),
            "delta_pct": 100.0 * float(deltas.mean()) / control_mean,
            "folds_improved": int((deltas < 0).sum()),
            "mean_mrr": float(
                np.mean(list(fold_mean_mrr(data_root, slug, weight).values()))
            ),
        }
    return _frame(rows)


def baseline_pool(data_root: Path, model: str) -> np.ndarray:
    """Mark the queries the untrained model answers under at least one format.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        One boolean per query.
    """
    return answerable_pool(leads_per_format(load_within(data_root, model, BASE_ARM)))


def inconsistent_matrix(
    data_root: Path, model: str, weight: float, subset: str = OOD
) -> np.ndarray:
    """Stack the inconsistent queries of every fold of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        weight: The weight of the invariance term.
        subset: The formats to report, one of the comparison subsets.

    Returns:
        One row per held-out format, one column per query.
    """
    rows = []
    for fold in FORMAT_NAMES:
        entry = load_within(data_root, model, arm(fold, weight))
        rows.append(answer_subsets(leads_per_format(entry), fold)[subset])
    return np.stack(rows)


def pooled_inconsistency(
    flags: np.ndarray, pool: np.ndarray, index: np.ndarray | None = None
) -> float:
    """Return the share of the pooled folds whose formats disagree.

    Args:
        flags: One row per fold, one column per query, True where the formats
            disagree.
        pool: The queries to count over.
        index: The resample to apply, or None to take every query once.

    Returns:
        The share, as a percentage. The share is NaN when the pool is empty.
    """
    kept = pool if index is None else pool[index]
    total = int(kept.sum()) * flags.shape[0]
    if total == 0:
        return float("nan")
    taken = flags if index is None else flags[:, index]
    return 100.0 * float((taken & kept).sum()) / total


def bootstrap_pooled_delta(
    control: np.ndarray,
    treatment: np.ndarray,
    pool: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> tuple[float, float]:
    """Bound the change in pooled inconsistency by resampling the queries.

    Args:
        control: The inconsistent queries of the control arm, by fold.
        treatment: The inconsistent queries of the treatment arm, by fold.
        pool: The queries to count over.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The lower bound and the upper bound, in percentage points.
    """
    rng = np.random.default_rng(seed)
    size = len(pool)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, size, size)
        deltas[i] = pooled_inconsistency(treatment, pool, index) - pooled_inconsistency(
            control, pool, index
        )
    tail = (1.0 - ci) / 2.0 * 100.0
    return float(np.percentile(deltas, tail)), float(np.percentile(deltas, 100 - tail))


def answer_table(
    data_root: Path,
    slugs: Sequence[str],
    subset: str = OOD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Summarise the answer axis of every model.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        subset: The formats to report, one of the comparison subsets.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        A DataFrame of the answer axis summary for each model.
    """
    rows = {}
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        pool = baseline_pool(data_root, slug)
        control = inconsistent_matrix(data_root, slug, CONTROL_WEIGHT, subset)
        treated = inconsistent_matrix(data_root, slug, weight, subset)
        before = pooled_inconsistency(control, pool)
        after = pooled_inconsistency(treated, pool)
        low, high = bootstrap_pooled_delta(
            control, treated, pool, n_boot=n_boot, seed=seed, ci=ci
        )
        rows[slug] = {
            "weight": weight,
            "n_pool": int(pool.sum()),
            "control_pct": before,
            "trained_pct": after,
            "delta_pp": after - before,
            "ci_lo": low,
            "ci_hi": high,
            "excludes_zero": bool(high < 0 or low > 0),
        }
    return _frame(rows)


def per_fold_matrix(
    data_root: Path, slugs: Sequence[str], axis: str, subset: str = OOD
) -> pd.DataFrame:
    """Tabulate the change of every model against every held-out format.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        axis: `score` for the maximum absolute Cohen's d, `answer` for the
            inconsistency in percentage points.
        subset: The formats to report, one of the comparison subsets.

    Returns:
        One row per model, one column per held-out format.

    Raises:
        ValueError: If the axis is not a known name.
    """
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}, expected one of: {AXES}")
    rows = {}
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        if axis == "score":
            control = fold_max_d(data_root, slug, CONTROL_WEIGHT, subset)
            treated = fold_max_d(data_root, slug, weight, subset)
            rows[slug] = {f: treated[f] - control[f] for f in FORMAT_NAMES}
            continue
        pool = baseline_pool(data_root, slug)
        before = inconsistent_matrix(data_root, slug, CONTROL_WEIGHT, subset)
        after = inconsistent_matrix(data_root, slug, weight, subset)
        rows[slug] = {
            fold: pooled_inconsistency(after[[i]], pool)
            - pooled_inconsistency(before[[i]], pool)
            for i, fold in enumerate(FORMAT_NAMES)
        }
    return _frame(rows)


def both_axes(score: pd.DataFrame, answer: pd.DataFrame) -> pd.DataFrame:
    """Join the change on each axis, one row per model.

    Args:
        score: The frame from `score_table`.
        answer: The frame from `answer_table`.

    Returns:
        The change on the score axis beside the change on the answer axis.
    """
    joined = score[["delta"]].join(answer[["delta_pp", "ci_lo", "ci_hi"]])
    return joined.rename(columns={"delta": "score_delta"})


def correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    """Return the Pearson correlation of two columns.

    Args:
        frame: The frame holding both columns.
        left: The first column.
        right: The second column.

    Returns:
        The correlation, or NaN when either column does not vary.
    """
    if len(frame) < 2:
        return float("nan")
    return float(np.corrcoef(frame[left], frame[right])[0, 1])
