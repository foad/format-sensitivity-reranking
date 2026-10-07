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
    max_abs_d_over_pairs,
    pair_subsets,
)
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import (
    BASE_ARM,
    arm,
    comparison_path,
    result_path,
    selection_path,
)
from fsr.metrics import (
    DEFAULT_CI,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    bootstrap_ci_of_mean,
)
from fsr.models.registry import BASE_MODELS, by_slug

CONTROL_WEIGHT = 0.0
SPLIT = "test"
SWEEP = "lambda"
AXES = ("score", "answer")

PROSE_SPLIT = "prose"
MRR_AXIS = "mrr"
CONTROL = "control"
WINNER = "winner"
PROSE_ARMS = (CONTROL, WINNER)


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


def weight_frame(data_root: Path, model: str) -> pd.DataFrame:
    """Tabulate one model's weight sweep, with intervals and the tied group.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        One row per weight in ascending order, holding the dev maximum
        absolute Cohen's d, its interval, whether the weight is in the tied
        group, and whether the selection chose it. A missing interval is NaN.
    """
    selection = load_selection(data_root, model)
    tied = {float(value) for value in selection.get("tied_lambdas", [])}
    winner = float(selection["winner_lambda"])
    rows = []
    for candidate in sorted(selection["candidates"], key=lambda c: float(c["lambda"])):
        value = float(candidate["lambda"])
        rows.append(
            {
                "weight": candidate["lambda"],
                "max_abs_d": candidate["dev_max_abs_d"],
                "ci_lo": candidate.get("dev_max_abs_d_ci_lo", float("nan")),
                "ci_hi": candidate.get("dev_max_abs_d_ci_hi", float("nan")),
                "tied": value in tied,
                "winner": value == winner,
            }
        )
    return pd.DataFrame(rows).set_index("weight")


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


def guardrail_frame(data_root: Path, slugs: Sequence[str]) -> pd.DataFrame:
    """Tabulate the ranking-quality guardrail of every fold of every model.

    The guardrail compares the treatment arm against the untrained model. The
    interval is reported whether or not the arm passes.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.

    Returns:
        One row per model and held-out format, holding the change in mean
        reciprocal rank, its interval, the margin, and the outcome.
    """
    rows = []
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        for fold in FORMAT_NAMES:
            section = load_comparison(data_root, slug, fold, weight)["score_axis"][
                "mrr_non_inferiority"
            ]
            low, high = section["delta_mrr_bootstrap"]["delta_ci"]
            rows.append(
                {
                    "model": by_slug(slug).label,
                    "held_out": fold,
                    "delta_mrr": section["delta_mrr_bootstrap"]["delta_mean"],
                    "ci_lo": low,
                    "ci_hi": high,
                    "margin": section["margin"],
                    "passed": section["passed"],
                }
            )
    return pd.DataFrame(rows).set_index(["model", "held_out"])


def transfer_frame(data_root: Path, slugs: Sequence[str]) -> pd.DataFrame:
    """Tabulate the ranking quality of the held-out format of every fold.

    The quantities are descriptive, so no threshold is applied to them.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.

    Returns:
        One row per model and held-out format, holding the change in mean
        reciprocal rank on the held-out format, its interval, the mean change
        over the training formats, and the ratio of the two.
    """
    rows = []
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        for fold in FORMAT_NAMES:
            section = load_comparison(data_root, slug, fold, weight)["score_axis"][
                "heldout_transfer"
            ]
            low, high = section["delta_mrr_ci"]
            rows.append(
                {
                    "model": by_slug(slug).label,
                    "held_out": fold,
                    "delta_mrr": section["delta_mrr"],
                    "ci_lo": low,
                    "ci_hi": high,
                    "training_mean": section["training_format_mean_delta_mrr"],
                    "transfer_ratio": section["transfer_ratio"],
                }
            )
    return pd.DataFrame(rows).set_index(["model", "held_out"])


def baseline_max_d(data_root: Path, model: str) -> float:
    """Return the untrained maximum absolute Cohen's d over every format pair."""
    comparison = load_comparison(data_root, model, FORMAT_NAMES[0], CONTROL_WEIGHT)
    return comparison["score_axis"]["point_estimates_max_d"]["all_formats"]["baseline"]


def load_scores(data_root: Path, model: str, arm_name: str) -> dict[str, np.ndarray]:
    """Load the per-record score of each format, from one score-axis evaluation.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        One array per format, one entry per record.
    """
    path = result_path(data_root, SPLIT, "cross", model, arm_name)
    stored = json.loads(path.read_text())["scores_per_fmt"]
    return {name: np.asarray(stored[name], dtype=float) for name in FORMAT_NAMES}


def fold_scores(
    data_root: Path, model: str, weight: float
) -> list[tuple[dict[str, np.ndarray], list[tuple[str, str]]]]:
    """Load the scores and the format pairs of every fold of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        weight: The weight of the invariance term.

    Returns:
        The scores and the pair split of each held-out format.
    """
    return [
        (load_scores(data_root, model, arm(fold, weight)), pair_subsets(fold))
        for fold in FORMAT_NAMES
    ]


def mean_max_d(
    folds: Sequence[tuple[dict[str, np.ndarray], dict[str, list[tuple[str, str]]]]],
    subset: str,
    index: np.ndarray | None = None,
) -> float:
    """Average the largest absolute effect size over the folds of one arm.

    Args:
        folds: The scores and the pair split of each fold.
        subset: The format pairs to measure, one of the comparison subsets.
        index: The resample to apply, or None to take every record once.

    Returns:
        The mean over the folds.
    """
    return float(
        np.mean(
            [
                max_abs_d_over_pairs(scores, pairs[subset], index)[0]
                for scores, pairs in folds
            ]
        )
    )


def bootstrap_delta_max_d(
    control: Sequence[tuple[dict[str, np.ndarray], dict[str, list[tuple[str, str]]]]],
    treatment: Sequence[tuple[dict[str, np.ndarray], dict[str, list[tuple[str, str]]]]],
    subset: str = OOD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> tuple[float, float]:
    """Bound the change in the score axis by resampling the records.

    One resample is applied to both arms and to every fold, so the interval is
    paired.

    Args:
        control: The folds of the control arm.
        treatment: The folds of the treatment arm.
        subset: The format pairs to measure, one of the comparison subsets.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The lower bound and the upper bound.
    """
    rng = np.random.default_rng(seed)
    size = len(next(iter(control[0][0].values())))
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        index = rng.integers(0, size, size)
        deltas[i] = mean_max_d(treatment, subset, index) - mean_max_d(
            control, subset, index
        )
    tail = (1.0 - ci) / 2.0 * 100.0
    return float(np.percentile(deltas, tail)), float(np.percentile(deltas, 100 - tail))


def score_table(
    data_root: Path,
    slugs: Sequence[str],
    subset: str = OOD,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Summarise the score axis of every model.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        subset: The format pairs to report, one of the comparison subsets.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The untrained maximum absolute Cohen's d, the fold mean of the control
        and of the treatment, the change, its interval, whether the interval
        excludes zero, the change as a share of the control, the folds that
        improved, and the mean ranking quality.
    """
    rows = {}
    for slug in slugs:
        weight = winner_weight(load_selection(data_root, slug))
        control = fold_max_d(data_root, slug, CONTROL_WEIGHT, subset)
        treated = fold_max_d(data_root, slug, weight, subset)
        deltas = np.array([treated[f] - control[f] for f in FORMAT_NAMES])
        control_mean = float(np.mean(list(control.values())))
        low, high = bootstrap_delta_max_d(
            fold_scores(data_root, slug, CONTROL_WEIGHT),
            fold_scores(data_root, slug, weight),
            subset=subset,
            n_boot=n_boot,
            seed=seed,
            ci=ci,
        )
        rows[slug] = {
            "weight": weight,
            "baseline_max_d": baseline_max_d(data_root, slug),
            "control_max_d": control_mean,
            "trained_max_d": float(np.mean(list(treated.values()))),
            "delta": float(deltas.mean()),
            "ci_lo": low,
            "ci_hi": high,
            "excludes_zero": bool(high < 0 or low > 0),
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


def load_prose_result(data_root: Path, model: str, arm_name: str) -> dict[str, Any]:
    """Load the prose ranking of one arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The result.
    """
    path = result_path(data_root, PROSE_SPLIT, MRR_AXIS, model, arm_name)
    return json.loads(path.read_text())


def load_prose_comparison(data_root: Path, model: str, arm_name: str) -> dict[str, Any]:
    """Load one arm's prose comparison against the untrained model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition.

    Returns:
        The comparison.
    """
    return json.loads(
        comparison_path(data_root, model, f"{PROSE_SPLIT}_{arm_name}").read_text()
    )


def prose_arms(data_root: Path, model: str) -> dict[str, dict[str, str]]:
    """Return the control arm and the winner arm of every fold.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The arm names, by held-out format and then by condition.
    """
    weight = winner_weight(load_selection(data_root, model))
    return {
        fold: {CONTROL: arm(fold, CONTROL_WEIGHT), WINNER: arm(fold, weight)}
        for fold in FORMAT_NAMES
    }


def prose_available(data_root: Path) -> list[str]:
    """Return the core models whose prose ranking is complete, in roster order.

    Args:
        data_root: The corpus directory.

    Returns:
        The registry slugs.
    """
    found = []
    for model in BASE_MODELS:
        if not selection_path(data_root, model.slug, SWEEP).exists():
            continue
        names = prose_arms(data_root, model.slug)
        paths = [result_path(data_root, PROSE_SPLIT, MRR_AXIS, model.slug, BASE_ARM)]
        paths += [
            comparison_path(data_root, model.slug, f"{PROSE_SPLIT}_{names[fold][key]}")
            for fold in FORMAT_NAMES
            for key in PROSE_ARMS
        ]
        if all(path.exists() for path in paths):
            found.append(model.slug)
    return found


def prose_baseline_mrr(data_root: Path, model: str) -> float:
    """Return the untrained ranking quality on prose.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The mean reciprocal rank.
    """
    return float(load_prose_result(data_root, model, BASE_ARM)["mrr"])


def prose_paired_gains(data_root: Path, model: str) -> np.ndarray:
    """Return the gain of the winner over the control, record by record.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        One row per held-out format, one column per record.

    Raises:
        ValueError: If two arms cover different records.
    """
    names = prose_arms(data_root, model)
    rows = []
    reference = None
    for fold in FORMAT_NAMES:
        control = load_prose_comparison(data_root, model, names[fold][CONTROL])
        winner = load_prose_comparison(data_root, model, names[fold][WINNER])
        reference = control["record_ids"] if reference is None else reference
        for name, entry in ((CONTROL, control), (WINNER, winner)):
            if entry["record_ids"] != reference:
                raise ValueError(
                    f"{model} {fold} {name} covers different records from the rest"
                )
        rows.append(
            np.asarray(winner["per_record_delta_rr"], dtype=float)
            - np.asarray(control["per_record_delta_rr"], dtype=float)
        )
    return np.stack(rows)


def prose_paired_delta(
    data_root: Path,
    model: str,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> tuple[float, float, float]:
    """Bound the gain of the winner over the control, over every fold.

    One resample is applied to every fold, so the interval is paired.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The mean gain and the bounds of its interval.
    """
    per_record = prose_paired_gains(data_root, model).mean(axis=0)
    low, high = bootstrap_ci_of_mean(per_record, n_boot=n_boot, seed=seed, ci=ci)
    return float(per_record.mean()), low, high


def prose_fold_frame(data_root: Path, model: str) -> pd.DataFrame:
    """Tabulate the prose ranking of every fold of one model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        One row per held-out format, holding each arm's mean reciprocal rank,
        the change against the untrained model with its interval, whether the
        arm kept its quality, and the gain of the winner over the control.
    """
    names = prose_arms(data_root, model)
    gains = prose_paired_gains(data_root, model)
    rows = {}
    for index, fold in enumerate(FORMAT_NAMES):
        row: dict[str, Any] = {}
        for key in PROSE_ARMS:
            entry = load_prose_comparison(data_root, model, names[fold][key])
            low, high = entry["delta_mrr_ci"]
            row[f"{key}_mrr"] = entry["trained_mrr"]
            row[f"{key}_delta"] = entry["delta_mrr_mean"]
            row[f"{key}_lo"] = low
            row[f"{key}_hi"] = high
            row[f"{key}_kept"] = bool(entry["ni_pass"])
        row["winner_gain"] = float(gains[index].mean())
        rows[fold] = row
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "held_out"
    return frame


def prose_table(
    data_root: Path,
    slugs: Sequence[str],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Summarise prose ranking quality for every model.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The untrained mean reciprocal rank, the fold mean of each arm and its
        change, the gain of the winner over the control with its interval,
        and the arms that kept their quality.
    """
    rows = {}
    for slug in slugs:
        folds = prose_fold_frame(data_root, slug)
        gain, low, high = prose_paired_delta(
            data_root, slug, n_boot=n_boot, seed=seed, ci=ci
        )
        rows[slug] = {
            "weight": winner_weight(load_selection(data_root, slug)),
            "baseline_mrr": prose_baseline_mrr(data_root, slug),
            "control_mrr": float(folds["control_mrr"].mean()),
            "winner_mrr": float(folds["winner_mrr"].mean()),
            "control_delta": float(folds["control_delta"].mean()),
            "winner_delta": float(folds["winner_delta"].mean()),
            "winner_gain": gain,
            "gain_lo": low,
            "gain_hi": high,
            "gain_excludes_zero": bool(high < 0 or low > 0),
            "arms_kept": int(folds["control_kept"].sum() + folds["winner_kept"].sum()),
        }
    return _frame(rows)


def prose_matrix(
    data_root: Path, slugs: Sequence[str], condition: str = WINNER
) -> pd.DataFrame:
    """Tabulate one arm's prose change against every held-out format.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        condition: The arm to report, one of PROSE_ARMS.

    Returns:
        One row per model, one column per held-out format.

    Raises:
        ValueError: If the condition is not a known name.
    """
    if condition not in PROSE_ARMS:
        raise ValueError(f"unknown arm {condition!r}, expected one of: {PROSE_ARMS}")
    rows = {}
    for slug in slugs:
        folds = prose_fold_frame(data_root, slug)
        rows[slug] = dict(folds[f"{condition}_delta"])
    return _frame(rows)
