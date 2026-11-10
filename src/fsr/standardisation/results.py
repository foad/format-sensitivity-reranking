"""Standardisation study result reader and table builder."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from fsr.comparison import NI_MARGIN
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import (
    BASE_ARM,
    PHASE1_FOLD,
    arm,
    selection_path,
)
from fsr.h2_layout import DEFAULT_SEED as DEFAULT_TRAINING_SEED
from fsr.h2_results import SWEEP, load_selection, winner_weight
from fsr.metrics import DEFAULT_CI, DEFAULT_N_BOOT, DEFAULT_SEED
from fsr.models.registry import BASE_MODELS, by_slug
from fsr.standardisation.layout import standardisation_dir, tensor_path
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
SPECIALIST = "standardised training"
CONTROL_WEIGHT = 0.0
SPLIT = "test"
HARD_QUANTILE = 0.2
N_PERMUTATIONS = 20000

STANDARDISED = "standardised"
SEED_SUFFIX = re.compile(r"_s(\d+)$")
MIXED = "mixed"


def arm_names(
    data_root: Path, model: str, training_seed: int = DEFAULT_TRAINING_SEED
) -> dict[str, str]:
    """Return the arm behind each condition of the study.

    The winner is left out when the sweep chose a weight of zero, where it
    is the control arm.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        training_seed: The seed the trained arms were trained under. The
            untrained arm has none.

    Returns:
        The arm name, by condition, in reporting order.
    """
    weight = winner_weight(load_selection(data_root, model))
    names = {
        UNTRAINED: BASE_ARM,
        CONTROL: arm(PHASE1_FOLD, CONTROL_WEIGHT, seed=training_seed),
    }
    if weight != CONTROL_WEIGHT:
        names[WINNER] = arm(PHASE1_FOLD, weight, seed=training_seed)
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


def tensors(
    data_root: Path,
    model: str,
    split: str = SPLIT,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> dict[str, Any]:
    """Load every condition of one model, by condition name."""
    return {
        condition: load_tensor(data_root, split, model, name)
        for condition, name in arm_names(data_root, model, training_seed).items()
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


def _seeded(
    data_root: Path, model: str, split: str, seeds: Sequence[int]
) -> dict[str, list[Any]]:
    """Load every condition of one model under each training seed.

    The untrained condition has no seed, so it holds one tensor.

    Raises:
        ValueError: If no seed is named.
    """
    if not seeds:
        raise ValueError("name at least one training seed")
    per_seed = [tensors(data_root, model, split, s) for s in seeds]
    return {
        condition: [per_seed[0][condition]]
        if condition == UNTRAINED
        else [loaded[condition] for loaded in per_seed]
        for condition in per_seed[0]
    }


def _mean_policies(loaded: list[Any]) -> dict[str, np.ndarray]:
    """Return each query's outcome under every policy, averaged over seeds."""
    per = [_policies(tensor) for tensor in loaded]
    return {name: np.mean([p[name] for p in per], axis=0) for name in per[0]}


def common_training_seeds(
    data_root: Path, slugs: Sequence[str], split: str = SPLIT
) -> list[int]:
    """Return the training seeds every model has captured in full.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs.
        split: The split the queries come from.

    Returns:
        The seeds, in ascending order.
    """
    found = [set(training_seeds(data_root, slug, split)) for slug in slugs]
    return sorted(set.intersection(*found)) if found else []


def levels_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Tabulate the share of queries the gold leads, by condition and policy.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average the trained conditions over.

    Returns:
        One row per model and condition, one column per policy.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        for condition, loaded in _seeded(data_root, slug, split, seeds).items():
            rows[(by_slug(slug).label, condition)] = {
                name: float(values[index].mean())
                for name, values in _mean_policies(loaded).items()
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
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Compare two condition and policy pairs, query by query.

    Over several training seeds, each query's outcome is first averaged
    across them.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        treatment: The condition and policy to test.
        reference: The condition and policy to test against.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        seeds: The training seeds to average the trained conditions over.

    Returns:
        One row per model, holding the change, its interval, the p-value
        before and after the correction for the models tested, and whether
        the corrected value clears five percent.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        loaded = _seeded(data_root, slug, split, seeds)
        pair = {}
        for role, (condition, policy) in (
            ("treatment", treatment),
            ("reference", reference),
        ):
            pair[role] = _mean_policies(loaded[condition])[policy][index]
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
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Measure how often the five standardisations give different answers.

    Over several training seeds, a query's disagreement is the share of the
    seeds under which its formats disagree.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        seeds: The training seeds to average the trained conditions over.

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
        loaded = _seeded(data_root, slug, split, seeds)
        if WINNER not in loaded:
            raise ValueError(f"{slug} carries no {WINNER!r} condition")
        disagrees = {}
        for condition, group in loaded.items():
            per_seed = []
            for tensor in group:
                leads = uniform_leads(tensor)
                per_seed.append(leads.any(axis=1) & ~leads.all(axis=1))
            disagrees[condition] = np.mean(per_seed, axis=0)
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


def specialist_arms(training_seed: int = DEFAULT_TRAINING_SEED) -> dict[str, str]:
    """Return the arm trained on each format alone, by format.

    Each arm sees the training corpus standardised to one format, at the
    weight of the control, as the invariance term needs two formats.

    Args:
        training_seed: The seed the arms were trained under.
    """
    return {
        name: arm(None, CONTROL_WEIGHT, only=name, seed=training_seed)
        for name in FORMAT_NAMES
    }


def specialist_available(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> list[str]:
    """Return the models with a capture of every single-format arm.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to look at.
        split: The split the queries come from.
        training_seed: The seed the arms were trained under.

    Returns:
        The registry slugs, in the order given.
    """
    return [
        slug
        for slug in slugs
        if all(
            tensor_path(data_root, split, slug, name).exists()
            for name in specialist_arms(training_seed).values()
        )
    ]


def training_seeds(data_root: Path, model: str, split: str = SPLIT) -> list[int]:
    """Return the training seeds whose every trained arm of one model is captured.

    A seed counts once the single-format arms and the trained arms of
    `arm_names` all hold a capture under it.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        split: The split the queries come from.

    Returns:
        The seeds, in ascending order.
    """
    prefix = tensor_path(data_root, split, model, "").name.removesuffix(".npy")
    found = {DEFAULT_TRAINING_SEED}
    for path in standardisation_dir(data_root).glob(f"{prefix}*_s*.npy"):
        match = SEED_SUFFIX.search(path.stem)
        if match:
            found.add(int(match.group(1)))
    complete = []
    for seed in sorted(found):
        names = [
            *specialist_arms(seed).values(),
            *(
                name
                for condition, name in arm_names(data_root, model, seed).items()
                if condition != UNTRAINED
            ),
        ]
        if all(tensor_path(data_root, split, model, name).exists() for name in names):
            complete.append(seed)
    return complete


def standardised_leads(
    data_root: Path,
    model: str,
    condition: str,
    split: str = SPLIT,
    training_seed: int = DEFAULT_TRAINING_SEED,
) -> np.ndarray:
    """Report whether the gold leads with the corpus standardised to each format.

    Under SPECIALIST, the column of a format comes from the arm trained on
    that format alone, so each column is a separate model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        condition: A condition of `arm_names`, or SPECIALIST.
        split: The split the queries come from.
        training_seed: The seed the trained arms were trained under.

    Returns:
        One row per query, one column per format of FORMAT_NAMES.

    Raises:
        ValueError: If a single-format capture covers other queries or
            formats than the untrained one.
    """
    if condition != SPECIALIST:
        tensor = tensors(data_root, model, split, training_seed)[condition]
        if list(tensor.formats) != list(FORMAT_NAMES):
            raise ValueError(f"{model} {condition} covers formats {tensor.formats}")
        return uniform_leads(tensor)
    reference = load_tensor(data_root, split, model, BASE_ARM)
    columns = []
    for name, arm_name in specialist_arms(training_seed).items():
        tensor = load_tensor(data_root, split, model, arm_name)
        if tensor.query_ids != reference.query_ids:
            raise ValueError(f"{model} {arm_name} covers other queries than {BASE_ARM}")
        columns.append(uniform_leads(tensor)[:, tensor.formats.index(name)])
    return np.stack(columns, axis=1)


def _outcome(
    data_root: Path,
    model: str,
    side: tuple[str, str],
    split: str,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> np.ndarray:
    """Return one side of a per-format contrast, one column per format.

    Over several training seeds, each query's outcome is its mean across
    the seeds.
    """
    condition, policy = side
    if policy == MIXED and condition == SPECIALIST:
        raise ValueError(f"no {policy!r} outcome for the {condition!r} condition")
    if policy not in (STANDARDISED, MIXED):
        raise ValueError(f"no {policy!r} outcome for the {condition!r} condition")
    if not seeds:
        raise ValueError("name at least one training seed")
    per_seed = []
    for training_seed in seeds:
        if policy == STANDARDISED:
            values = standardised_leads(
                data_root, model, condition, split, training_seed
            ).astype(float)
        else:
            lead = mixed_lead(
                tensors(data_root, model, split, training_seed)[condition]
            )
            values = np.repeat(lead[:, None], len(FORMAT_NAMES), axis=1)
        per_seed.append(values)
    return np.mean(per_seed, axis=0)


def standardised_table(
    data_root: Path,
    model: str,
    conditions: Sequence[str] = (UNTRAINED, SPECIALIST),
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Tabulate each standardisation choice of one model, by condition.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        conditions: The conditions to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average the trained arms over.

    Returns:
        One row per format, one column per condition, holding the share of
        queries the gold leads.
    """
    index = slice(None) if subset is None else subset[model]
    return pd.DataFrame(
        {
            condition: _outcome(
                data_root, model, (condition, STANDARDISED), split, seeds
            )[index].mean(axis=0)
            for condition in conditions
        },
        index=pd.Index(FORMAT_NAMES, name="format"),
    )


def comparison_points_table(
    data_root: Path,
    model: str,
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Set standardising against fine-tuning on a mixed corpus, for one model.

    The two standardised conditions vary with the format. The trained
    conditions on the mixed corpus do not, so their value repeats down
    their columns.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average the trained arms over.

    Returns:
        One row per format, holding the share of queries the gold leads
        when the corpus is standardised, untrained and with the arm trained
        on that format, then on the mixed corpus for each trained condition
        of `arm_names`.
    """
    frame = standardised_table(
        data_root, model, (UNTRAINED, SPECIALIST), split, subset, seeds
    ).add_prefix(f"{STANDARDISED}, ")
    index = slice(None) if subset is None else subset[model]
    for condition in arm_names(data_root, model):
        if condition != UNTRAINED:
            values = _outcome(data_root, model, (condition, MIXED), split, seeds)
            frame[f"{MIXED}, {condition}"] = float(values[index, 0].mean())
    return frame


def format_contrast_table(
    data_root: Path,
    model: str,
    treatment: tuple[str, str],
    reference: tuple[str, str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Compare two condition and policy pairs of one model, format by format.

    A standardised side takes the format of the row. A mixed side does not
    depend on the format, so it is the same in every row. Over several
    training seeds, each query's outcome is first averaged across them, so
    the test resamples queries with the training noise averaged out.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        treatment: The condition and policy to test, the policy STANDARDISED
            or MIXED.
        reference: The condition and policy to test against.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        n_boot: The number of replicates.
        seed: The seed for the resample.
        seeds: The training seeds to average the trained arms over.

    Returns:
        One row per format, holding the change, its interval, the p-value
        before and after the correction for the formats tested, and whether
        the corrected value clears five percent.

    Raises:
        ValueError: If neither side is standardised, so no row depends on
            the format.
    """
    if STANDARDISED not in (treatment[1], reference[1]):
        raise ValueError("one side must be standardised for the format to matter")
    index = slice(None) if subset is None else subset[model]
    difference = (
        _outcome(data_root, model, treatment, split, seeds)
        - _outcome(data_root, model, reference, split, seeds)
    )[index]
    rows = {
        name: _paired(difference[:, i], n_boot=n_boot, seed=seed)
        for i, name in enumerate(FORMAT_NAMES)
    }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "format"
    frame["holm_p"] = holm(frame["p"])
    frame["resolved"] = frame["holm_p"] < 0.05
    return frame


def seed_table(
    data_root: Path,
    model: str,
    treatment: tuple[str, str],
    reference: tuple[str, str],
    seeds: Sequence[int],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """Report one per-format contrast under each training seed on its own.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        treatment: The condition and policy to test.
        reference: The condition and policy to test against.
        seeds: The training seeds to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.

    Returns:
        One row per format, holding the change under each seed, their mean
        and standard deviation, and how many seeds share the sign of the
        mean.

    Raises:
        ValueError: If neither side is standardised, or no seed is named.
    """
    if STANDARDISED not in (treatment[1], reference[1]):
        raise ValueError("one side must be standardised for the format to matter")
    if not seeds:
        raise ValueError("name at least one training seed")
    index = slice(None) if subset is None else subset[model]
    changes = {
        f"seed {s}": (
            _outcome(data_root, model, treatment, split, (s,))
            - _outcome(data_root, model, reference, split, (s,))
        )[index].mean(axis=0)
        for s in seeds
    }
    frame = pd.DataFrame(changes, index=pd.Index(FORMAT_NAMES, name="format"))
    values = frame.to_numpy()
    mean = values.mean(axis=1)
    frame["mean"] = mean
    frame["sd"] = values.std(axis=1, ddof=1) if len(seeds) > 1 else np.nan
    frame["same_sign"] = (np.sign(values) == np.sign(mean)[:, None]).sum(axis=1)
    return frame


TRAINING_REFERENCES = (CONTROL, WINNER)


def training_levels_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
) -> pd.DataFrame:
    """Set out the four ways of handling the formats, model by model.

    The standardised conditions are averaged over the formats.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average the trained arms over.

    Returns:
        One row per model, holding the share of queries the gold leads when
        standardising untrained, standardising with the arm trained on that
        format, and on the mixed corpus for each trained condition.
    """
    rows = {}
    for slug in slugs:
        points = comparison_points_table(data_root, slug, split, subset, seeds).mean()
        rows[by_slug(slug).label] = {
            UNTRAINED: points[f"{STANDARDISED}, {UNTRAINED}"],
            SPECIALIST: points[f"{STANDARDISED}, {SPECIALIST}"],
            **{
                condition: points[f"{MIXED}, {condition}"]
                for condition in TRAINING_REFERENCES
                if f"{MIXED}, {condition}" in points
            },
        }
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "model"
    return frame


def training_level_contrast_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare each other trained condition to fine-tuning at the control weight.

    Standardising with the arm trained on that format is averaged over the
    formats, as `training_levels_table` reports it. Each query's outcome is
    averaged over the seeds before the test.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average over.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        One row per model and condition, holding the change over the control
        on the mixed corpus, its interval, the p-value before and after the
        correction for the models tested, and whether the corrected value
        clears five percent.
    """
    rows = {}
    for slug in slugs:
        index = slice(None) if subset is None else subset[slug]
        reference = _outcome(data_root, slug, (CONTROL, MIXED), split, seeds)[:, 0]
        treatments = {
            SPECIALIST: _outcome(
                data_root, slug, (SPECIALIST, STANDARDISED), split, seeds
            ).mean(axis=1)
        }
        if WINNER in arm_names(data_root, slug):
            treatments[WINNER] = _outcome(
                data_root, slug, (WINNER, MIXED), split, seeds
            )[:, 0]
        for condition, values in treatments.items():
            rows[(by_slug(slug).label, condition)] = _paired(
                (values - reference)[index], n_boot=n_boot, seed=seed
            )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = pd.MultiIndex.from_tuples(frame.index, names=["model", "condition"])
    frame["holm_p"] = np.nan
    for condition in frame.index.get_level_values("condition").unique():
        mask = frame.index.get_level_values("condition") == condition
        frame.loc[mask, "holm_p"] = holm(frame.loc[mask, "p"])
    frame["resolved"] = frame["holm_p"] < 0.05
    return frame


def training_gap_table(
    data_root: Path,
    slugs: Sequence[str],
    reference: str = CONTROL,
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Compare standardising with training on that format to a mixed corpus.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        reference: The trained condition on the mixed corpus to compare to.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average over.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        One row per model and format, holding the seed-averaged change, its
        interval, the p-value after the correction for the formats tested,
        whether that clears five percent, how many seeds share the sign of
        the change, and the lowest and highest change of any one seed.
    """
    sides = ((SPECIALIST, STANDARDISED), (reference, MIXED))
    frames = {}
    for slug in slugs:
        tested = format_contrast_table(
            data_root, slug, *sides, split, subset, n_boot, seed, seeds
        )
        spread = seed_table(data_root, slug, *sides, seeds, split, subset)
        per_seed = spread[[f"seed {s}" for s in seeds]]
        tested["same_sign"] = spread["same_sign"].astype(int)
        tested["seed_lo"] = per_seed.min(axis=1)
        tested["seed_hi"] = per_seed.max(axis=1)
        frames[by_slug(slug).label] = tested[
            [
                "diff",
                "ci_lo",
                "ci_hi",
                "holm_p",
                "resolved",
                "same_sign",
                "seed_lo",
                "seed_hi",
            ]
        ]
    return pd.concat(frames, names=["model", "format"])


def training_summary_table(
    data_root: Path,
    slugs: Sequence[str],
    split: str = SPLIT,
    subset: dict[str, np.ndarray] | None = None,
    seeds: Sequence[int] = (DEFAULT_TRAINING_SEED,),
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Summarise standardising with training against the mixed corpus.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        split: The split the queries come from.
        subset: The queries to measure, by slug, or None for every query.
        seeds: The training seeds to average over.
        n_boot: The number of replicates.
        seed: The seed for the resample.

    Returns:
        One row per model, holding the four levels of
        `training_levels_table`, then for each trained condition on the
        mixed corpus the mean change over the formats, the formats where it
        resolves, and the seeds that share the sign of the mean change.
    """
    frame = training_levels_table(data_root, slugs, split, subset, seeds)
    for reference in TRAINING_REFERENCES:
        sides = ((SPECIALIST, STANDARDISED), (reference, MIXED))
        gaps = training_gap_table(
            data_root, slugs, reference, split, subset, seeds, n_boot, seed
        )
        frame[f"diff vs {reference}"] = gaps["diff"].groupby(level="model").mean()
        frame[f"formats resolved vs {reference}"] = (
            gaps["resolved"].groupby(level="model").sum().astype(int)
        )
        agreeing = {}
        for slug in slugs:
            per_seed = seed_table(data_root, slug, *sides, seeds, split, subset)
            means = per_seed[[f"seed {s}" for s in seeds]].mean(axis=0)
            overall = np.sign(means.mean())
            agreeing[by_slug(slug).label] = int((np.sign(means) == overall).sum())
        frame[f"seeds agreeing vs {reference}"] = pd.Series(agreeing)
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
