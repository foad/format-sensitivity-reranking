"""Standardisation study result reader and table builder."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fsr.comparison import NI_MARGIN
from fsr.h2_layout import BASE_ARM, PHASE1_FOLD, arm, selection_path
from fsr.h2_results import SWEEP, load_selection, winner_weight
from fsr.metrics import DEFAULT_CI, DEFAULT_N_BOOT, DEFAULT_SEED
from fsr.models.registry import BASE_MODELS, by_slug
from fsr.standardisation.layout import tensor_path
from fsr.standardisation.metrics import (
    inconsistency,
    margin,
    mixed_lead,
    oracle_band,
    uniform_leads,
    uniform_table,
)
from fsr.standardisation.store import load_tensor

UNTRAINED = "untrained"
CONTROL = "rank only"
WINNER = "invariance"
CONTROL_WEIGHT = 0.0
SPLIT = "test"
HARD_QUANTILE = 0.2
N_PERMUTATIONS = 20000

STANDARDISED = "standardised"
MIXED = "mixed"


def arm_names(data_root: Path, model: str) -> dict[str, str]:
    """Return the arm behind each condition of the study.

    The winner is left out when the sweep chose a weight of zero, where it
    is the control arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The arm name, by condition, in reporting order.
    """
    weight = winner_weight(load_selection(data_root, model))
    names = {UNTRAINED: BASE_ARM, CONTROL: arm(PHASE1_FOLD, CONTROL_WEIGHT)}
    if weight != CONTROL_WEIGHT:
        names[WINNER] = arm(PHASE1_FOLD, weight)
    return names


def available(data_root: Path, split: str = SPLIT) -> list[str]:
    """Return the models whose every condition is captured, in roster order.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.

    Returns:
        The registry slugs.
    """
    found = []
    for model in BASE_MODELS:
        if not selection_path(data_root, model.slug, SWEEP).exists():
            continue
        names = arm_names(data_root, model.slug).values()
        if all(
            tensor_path(data_root, split, model.slug, name).exists() for name in names
        ):
            found.append(model.slug)
    return found


def tensors(data_root: Path, model: str, split: str = SPLIT) -> dict[str, Any]:
    """Load every condition of one model, by condition name."""
    return {
        condition: load_tensor(data_root, split, model, name)
        for condition, name in arm_names(data_root, model).items()
    }


def hard_subset(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    quantile: float = HARD_QUANTILE,
    condition: str = UNTRAINED,
) -> dict[str, np.ndarray]:
    """Return the queries the panel of models finds hardest.

    Every model is judged on one subset, so the rows of a table share a
    population. Margins become within-model percentiles, as the score
    scales differ.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs that form the panel.
        split: The split the queries come from.
        quantile: The share of queries the subset holds.
        condition: The condition whose margin defines difficulty.

    Returns:
        The query positions of the subset, by registry slug.

    Raises:
        ValueError: If fewer than two models are given.
    """
    if len(slugs) < 2:
        raise ValueError("the subset needs at least two models to form a panel")
    percentile = []
    for slug in slugs:
        values = margin(tensors(data_root, slug, split)[condition])
        percentile.append(rankdata(values) / len(values))
    size = int(len(percentile[0]) * quantile)
    picked = np.argsort(np.mean(percentile, axis=0))[:size]
    return {slug: picked for slug in slugs}


def _paired(
    values: np.ndarray,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
    n_permutations: int = N_PERMUTATIONS,
) -> dict[str, float]:
    """Bound a per-query difference and test it against no difference.

    The interval resamples the queries. The test flips the sign of each
    query's difference, under the null that the two arms are alike.
    """
    rng = np.random.default_rng(seed)
    size = len(values)
    replicates = np.array(
        [values[rng.integers(0, size, size)].mean() for _ in range(n_boot)]
    )
    tail = (1.0 - ci) / 2.0 * 100.0
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, size))
    extreme = np.abs((signs * values).mean(axis=1)) >= abs(values.mean())
    return {
        "diff": float(values.mean()),
        "ci_lo": float(np.percentile(replicates, tail)),
        "ci_hi": float(np.percentile(replicates, 100 - tail)),
        "p": float((1 + extreme.sum()) / (n_permutations + 1)),
        "p_floor": float(1 / (n_permutations + 1)),
    }


def holm(p_values: Sequence[float]) -> np.ndarray:
    """Adjust p-values for the number of models tested.

    Args:
        p_values: One p-value per model.

    Returns:
        The adjusted values, in the order given.
    """
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    count = len(values)
    adjusted = np.empty(count)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def _policies(tensor: Any) -> dict[str, np.ndarray]:
    """Return the per-query outcome of each way of handling the formats."""
    leads = uniform_leads(tensor)
    return {
        "standardised_mean": leads.mean(axis=1),
        "standardised_best": leads[:, int(leads.mean(axis=0).argmax())].astype(float),
        MIXED: mixed_lead(tensor),
    }


def levels_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Tabulate the share of queries the gold leads, by condition and policy.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.

    Returns:
        One row per model and condition, one column per policy.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        for condition, tensor in tensors(data_root, slug, split).items():
            rows[(by_slug(slug).label, condition)] = {
                name: float(values[index].mean())
                for name, values in _policies(tensor).items()
            }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["model", "condition"])
    return frame


def contrast_table(
    data_root: Path,
    slugs: Sequence[str],
    treatment: tuple[str, str],
    reference: tuple[str, str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare two condition and policy pairs, query by query.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        treatment: The condition and policy to test.
        reference: The condition and policy to test against.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        One row per model, holding the change, its interval, the p-value
        before and after the correction for the models tested, and whether
        the corrected value clears five percent.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        loaded = tensors(data_root, slug, split)
        pair = {}
        for role, (condition, policy) in (
            ("treatment", treatment),
            ("reference", reference),
        ):
            pair[role] = _policies(loaded[condition])[policy][index]
        rows[by_slug(slug).label] = _paired(
            pair["treatment"] - pair["reference"], n_boot=n_boot, seed=seed
        )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"
    frame["holm_p"] = holm(frame["p"])
    frame["resolved"] = frame["holm_p"] < 0.05
    return frame


def agreement_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Measure how often the five standardisations give different answers.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        One row per model, holding the share of queries the formats
        disagree on in each condition, and the change the invariance term
        makes to that share.

    Raises:
        ValueError: If a model carries no winner condition.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        loaded = tensors(data_root, slug, split)
        if WINNER not in loaded:
            raise ValueError(f"{slug} carries no {WINNER!r} condition")
        disagrees = {}
        for condition, tensor in loaded.items():
            leads = uniform_leads(tensor)
            disagrees[condition] = (leads.any(axis=1) & ~leads.all(axis=1)).astype(
                float
            )
        row = {f"disagree_{c}": float(v[index].mean()) for c, v in disagrees.items()}
        row.update(
            _paired(
                disagrees[WINNER][index] - disagrees[CONTROL][index],
                n_boot=n_boot,
                seed=seed,
            )
        )
        rows[by_slug(slug).label] = row
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"
    frame["holm_p"] = holm(frame["p"])
    frame["resolved"] = frame["holm_p"] < 0.05
    return frame


def format_table(
    data_root: Path,
    slugs: Sequence[str],
    condition: str = UNTRAINED,
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Tabulate each standardisation choice on its own.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        condition: The condition to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.

    Returns:
        One row per model, one column per format, then the mixed corpus.

    Raises:
        ValueError: If a model does not carry the condition.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        loaded = tensors(data_root, slug, split)
        if condition not in loaded:
            raise ValueError(f"{slug} carries no {condition!r} condition")
        tensor = loaded[condition]
        leads = uniform_leads(tensor)
        row = {
            name: float(leads[index, i].mean()) for i, name in enumerate(tensor.formats)
        }
        row[MIXED] = float(mixed_lead(tensor)[index].mean())
        rows[by_slug(slug).label] = row
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"
    return frame


def guardrail_table(
    data_root: Path, slugs: Sequence[str], split: str = SPLIT
) -> pd.DataFrame:
    """Check that mixing the serialisations keeps the ranking quality.

    The best single format is the reference, as the strongest standardisation
    on offer.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.

    Returns:
        One row per model and condition, holding the mean reciprocal rank of
        the best single format and of the mixed corpus, the change, and
        whether the change stays inside the margin.
    """
    rows = {}
    for slug in slugs:
        for condition, tensor in tensors(data_root, slug, split).items():
            uniform = uniform_table(tensor)["mrr"]
            best = float(uniform.max())
            mixed = float(mixed_lead(tensor).mean())
            rows[(by_slug(slug).label, condition)] = {
                "uniform_best_mrr": best,
                "uniform_mean_mrr": float(uniform.mean()),
                "mixed_top1": mixed,
                "delta": mixed - best,
                "ni_margin": NI_MARGIN,
                "kept": bool(mixed - best > -NI_MARGIN),
            }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["model", "condition"])
    return frame


def inconsistency_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Measure the disagreement between the five standardisation choices.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.

    Returns:
        One row per model and condition, holding the answerable pool and the
        queries whose leading passage depends on the format.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        for condition, tensor in tensors(data_root, slug, split).items():
            rows[(by_slug(slug).label, condition)] = inconsistency(
                uniform_leads(tensor)[index]
            )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["model", "condition"])
    return frame


def oracle_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Bound what any choice of format can do to the leading passage.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.

    Returns:
        One row per model and condition, holding the share of queries the
        gold leads under every format, under the best single format, and
        under the best format of each query.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        for condition, tensor in tensors(data_root, slug, split).items():
            rows[(by_slug(slug).label, condition)] = oracle_band(
                uniform_leads(tensor)[index]
            )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["model", "condition"])
    return frame
