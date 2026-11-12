"""H1 result file reader and table builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsr.formats import FORMAT_NAMES
from fsr.metrics import (
    DEFAULT_CI,
    DEFAULT_N_BOOT,
    DEFAULT_SEED,
    bootstrap_ci_of_max_abs_d,
    bootstrap_max_abs_d_replicates,
    bootstrap_p_above,
    holm,
    pairwise_cohen_d,
)
from fsr.models.registry import BASE_MODELS, MODELS, by_slug
from fsr.within_query import (
    bootstrap_ci_of_inconsistency,
    bootstrap_inconsistency_replicates,
    bootstrap_tables,
    conditional_inconsistency,
)

RESULTS_SUBDIR = "h1"
AXES = ("cross", "within")
MODES = ("with_body", "metadata_only")
SPLITS = ("test", "nq_val", "all")
# Cohen's conventional threshold for a medium effect
MEDIUM_EFFECT = 0.5
# family-wise error rate for claims made across the roster
FAMILY_ALPHA = 0.05


def results_dir(data_root: Path) -> Path:
    """Return the directory holding the H1 results."""
    return data_root / RESULTS_SUBDIR


def result_path(data_root: Path, split: str, axis: str, mode: str, slug: str) -> Path:
    """Return the file holding one model's results.

    Args:
        data_root: The corpus directory.
        split: The split measured.
        axis: `cross` or `within`.
        mode: `with_body` or `metadata_only`.
        slug: The registry slug of the model.

    Returns:
        The path, whether or not it exists.
    """
    return results_dir(data_root) / f"{split}_{axis}_{mode}_{slug}.json"


def load_axis(data_root: Path, split: str, axis: str, mode: str) -> dict[str, Any]:
    """Load every model's results for one split, axis and mode.

    A model whose file is absent is left out, so a partial run still reads.

    Args:
        data_root: The corpus directory.
        split: The split measured.
        axis: `cross` or `within`.
        mode: `with_body` or `metadata_only`.

    Returns:
        The entry of each model, keyed by registry slug, in roster order.

    Raises:
        ValueError: If a file holds results for more than one model.
    """
    entries = {}
    for model in MODELS:
        path = result_path(data_root, split, axis, mode, model.slug)
        if not path.exists():
            continue
        results = json.loads(path.read_text())["results"]
        if len(results) != 1:
            raise ValueError(f"{path.name} holds {len(results)} models, expected 1")
        entries[model.slug] = next(iter(results.values()))
    return entries


def available(data_root: Path, axis: str, mode: str) -> list[str]:
    """Return the splits that hold results for an axis and mode."""
    return [split for split in SPLITS if load_axis(data_root, split, axis, mode)]


def measured_counts(data_root: Path, mode: str) -> dict[str, dict[str, int]]:
    """Return the records and the queries each split contributed.

    Args:
        data_root: The corpus directory.
        mode: `with_body` or `metadata_only`.

    Returns:
        The `records` and `queries` count of each split that has results.
    """
    counts: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        found = {}
        for axis, key, name in (
            ("cross", "n_records", "records"),
            ("within", "n_queries", "queries"),
        ):
            for model in MODELS:
                path = result_path(data_root, split, axis, mode, model.slug)
                if path.exists():
                    found[name] = json.loads(path.read_text())[key]
                    break
        if found:
            counts[split] = found
    return counts


def _frame(rows: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Return a frame indexed by model label, in roster order."""
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index = [by_slug(slug).label for slug in frame.index]
    frame.index.name = "model"
    return frame


def score_table(entries: dict[str, Any]) -> pd.DataFrame:
    """Summarise the score axis for each model.

    Args:
        entries: The cross-query entry of each model, by slug.

    Returns:
        Maximum and mean absolute Cohen's d, the pair that produced the
        maximum, the worst flip rate, and the lowest rank correlation.
    """
    rows = {}
    for slug, entry in entries.items():
        summary = entry["stats"]["summary"]
        rows[slug] = {
            "max_abs_d": summary["max_abs_cohen_d"],
            "max_d_pair": summary["max_d_pair"],
            "mean_abs_d": summary["mean_abs_cohen_d"],
            "max_flip_pct": summary["max_flip_rate_pct"],
            "min_rho": summary["min_spearman_rho"],
        }
    return _frame(rows)


def pairwise_table(entries: dict[str, Any]) -> pd.DataFrame:
    """Return every format pair's effect size, one row per model and pair.

    Args:
        entries: The cross-query entry of each model, by slug.

    Returns:
        The model label, the pair, and the signed and absolute effect size.
    """
    rows = []
    for slug, entry in entries.items():
        for pair in entry["stats"]["pairwise"]:
            left, right = (part.strip() for part in pair["pair"].split("-", 1))
            rows.append(
                {
                    "model": by_slug(slug).label,
                    "left": left,
                    "right": right,
                    "cohen_d": pair["cohen_d"],
                    "abs_d": abs(pair["cohen_d"]),
                }
            )
    return pd.DataFrame(rows)


def answer_table(entries: dict[str, Any]) -> pd.DataFrame:
    """Summarise the answer axis for each model.

    Args:
        entries: The within-query entry of each model, by slug.

    Returns:
        The model label and various answer-axis statistics.
    """
    rows = {}
    for slug, entry in entries.items():
        summary = entry["within_query"]["summary"]
        gold = entry["gold_top1"]
        conditional = entry["conditional_inconsistency"]
        rows[slug] = {
            "inconsistency_pct": conditional["inconsistency_pct"],
            "answerable_pct": conditional["answerable_pct"],
            "top1_all_formats_pct": gold["gold_top1_all_formats_pct"],
            "top1_format_dependent_pct": gold["gold_top1_format_dependent_pct"],
            "top1_never_pct": gold["gold_top1_never_pct"],
            "max_flip_pct": summary["max_flip_rate_pct"],
            "min_tau": summary["min_kendall_tau"],
            "delta_over_s": entry["scale_diagnostic"]["ratio"],
        }
    return _frame(rows)


def per_format_top1(entries: dict[str, Any]) -> pd.DataFrame:
    """Return the share of queries with the gold first, by model and format.

    Args:
        entries: The within-query entry of each model, by slug.

    Returns:
        One row per model, one column per format, with the spread.
    """
    rows = {
        slug: dict(entry["gold_top1"]["per_format_top1_pct"])
        for slug, entry in entries.items()
    }
    frame = _frame(rows)
    frame["spread_pp"] = frame.max(axis=1) - frame.min(axis=1)
    return frame


def per_format_scores(entries: dict[str, Any]) -> pd.DataFrame:
    """Return the mean score of each format, by model.

    Args:
        entries: The cross-query entry of each model, by slug.

    Returns:
        One row per model, one column per format.
    """
    rows = {
        slug: {
            name: stats["mean"] for name, stats in entry["stats"]["per_format"].items()
        }
        for slug, entry in entries.items()
    }
    return _frame(rows)


def bootstrap_frame(
    entries: dict[str, Any], n_boot: int = 10_000, seed: int = 0
) -> pd.DataFrame:
    """Bootstrap the answer-axis statistics over queries.

    Args:
        entries: The within-query entry of each model, by slug.
        n_boot: The number of resamples.
        seed: The seed for the resampling.

    Returns:
        One row per model, with the bootstrapped statistics.
    """
    by_id = {by_slug(slug).model_id: entry for slug, entry in entries.items()}
    per_model, _, _, _ = bootstrap_tables(
        by_id, list(FORMAT_NAMES), n_boot=n_boot, seed=seed
    )
    rows = {}
    for slug in entries:
        value = per_model[by_slug(slug).model_id]
        rows[slug] = {
            "format_dependent_pct": value["format_dependent_pct"],
            "format_dependent_lo": value["format_dependent_ci"][0],
            "format_dependent_hi": value["format_dependent_ci"][1],
            "flip_rate_pct": value["flip_rate_pct"],
            "flip_rate_lo": value["flip_rate_ci"][0],
            "flip_rate_hi": value["flip_rate_ci"][1],
            "kendall_tau": value["kendall_tau"],
            "worst_pair": value["worst_pair"],
        }
    return _frame(rows)


def score_intervals(
    entries: dict[str, Any],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Bootstrap the largest absolute effect size, resampling records.

    Args:
        entries: The cross-query entry of each model, by slug.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The point estimate and the interval bounds, by model label.
    """
    rows = {}
    for slug, entry in entries.items():
        low, high = bootstrap_ci_of_max_abs_d(entry["scores"], n_boot, seed, ci)
        rows[slug] = {
            "max_abs_d": entry["stats"]["summary"]["max_abs_cohen_d"],
            "low": low,
            "high": high,
        }
    return _frame(rows)


def answer_intervals(
    entries: dict[str, Any],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Bootstrap the conditional inconsistency share, resampling queries.

    Args:
        entries: The within-query entry of each model, by slug.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        The point estimate and the interval bounds, by model label.
    """
    rows = {}
    for slug, entry in entries.items():
        top1 = {
            name: np.asarray(ranks) == 1.0
            for name, ranks in entry["reciprocal_ranks"].items()
        }
        low, high = bootstrap_ci_of_inconsistency(top1, n_boot, seed, ci)
        rows[slug] = {
            "inconsistency_pct": entry["conditional_inconsistency"][
                "inconsistency_pct"
            ],
            "low": low,
            "high": high,
        }
    return _frame(rows)


def overflow_path(data_root: Path, split: str) -> Path:
    """Return the file recording which passages exceed a model's context."""
    return results_dir(data_root) / f"{split}_overflow.json"


def load_overflow(data_root: Path, split: str) -> dict[str, Any]:
    """Load the overflow record of one split.

    Args:
        data_root: The corpus directory.
        split: The split measured.

    Returns:
        The record written by `scripts/h1/overflow.py`.
    """
    return json.loads(overflow_path(data_root, split).read_text())


def overflow_summary(overflow: dict[str, Any]) -> pd.DataFrame:
    """Summarise, per model, the passages that exceeded its context.

    Args:
        overflow: The overflow record from `load_overflow`.

    Returns:
        One row per model: the share of candidate renderings over the context,
        the median and largest excess in tokens, and the records and queries
        holding at least one such rendering.
    """
    cross, within = overflow["cross"], overflow["within"]
    rows = {}
    for model in BASE_MODELS:
        if model.model_id not in within["by_model"]:
            continue
        w = within["by_model"][model.model_id]
        rows[model.slug] = {
            "renderings_over_pct": 100.0 * w["renderings_over"] / w["renderings"],
            "excess_median": w["excess_median"],
            "excess_max": w["excess_max"],
            "records": len(cross["by_model"][model.model_id]["ids"]),
            "queries": len(w["ids"]),
        }
    return _frame(rows)


def rescore_path(data_root: Path, split: str, slug: str) -> Path:
    """Return the file holding one model's re-scored overflow items."""
    return results_dir(data_root) / f"{split}_rescore_{slug}.json"


def load_rescores(data_root: Path, split: str) -> dict[str, Any]:
    """Load every model's re-scored overflow items for one split.

    Args:
        data_root: The corpus directory.
        split: The split measured.

    Returns:
        The record of each model whose file exists, keyed by registry slug.
    """
    out = {}
    for model in MODELS:
        path = rescore_path(data_root, split, model.slug)
        if path.exists():
            out[model.slug] = json.loads(path.read_text())
    return out


def _splice(
    values: dict[str, Any], ids: list[str], replaced: list[str], new: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Return each format's values with the replaced ids taking their new values.

    Raises:
        ValueError: If a replaced id is not among the ids.
    """
    position = {item: n for n, item in enumerate(ids)}
    missing = [i for i in replaced if i not in position]
    if missing:
        raise ValueError(f"{len(missing)} replaced ids are not in the results")
    rows = [position[i] for i in replaced]
    out = {}
    for name, seq in values.items():
        arr = np.array(seq)
        arr[rows] = np.asarray(new[name], dtype=arr.dtype)
        out[name] = arr
    return out


def rescored_table(
    cross_entries: dict[str, Any],
    within_entries: dict[str, Any],
    overflow: dict[str, Any],
    rescores: dict[str, Any],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> pd.DataFrame:
    """Recompute both axes with the overflowing items scored again in full.

    Nothing is dropped: each overflowing record and query takes the scores it
    receives once its passages fit the context, and every other item keeps its
    original scores.

    Args:
        cross_entries: The cross-query entry of each model, by slug.
        within_entries: The within-query entry of each model, by slug.
        overflow: The overflow record from `load_overflow`.
        rescores: The re-scored items of each model, from `load_rescores`.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        ci: The interval width.

    Returns:
        One row per re-scored model: the items replaced, each axis before and
        after with the interval of the new estimate, and the largest score
        difference on the control records.
    """
    rows = {}
    for slug, record in rescores.items():
        cross, within = record["cross"], record["within"]
        scores = _splice(
            cross_entries[slug]["scores"],
            overflow["cross"]["record_ids"],
            cross["record_ids"],
            cross["scores"],
        )
        _, max_abs_d, _ = pairwise_cohen_d(scores)
        d_low, d_high = bootstrap_ci_of_max_abs_d(scores, n_boot, seed, ci)
        ranks = within_entries[slug]["reciprocal_ranks"]
        if within["query_ids"]:
            ranks = _splice(
                ranks,
                overflow["within"]["query_ids"],
                within["query_ids"],
                within["reciprocal_ranks"],
            )
        top1 = {name: np.asarray(r) == 1.0 for name, r in ranks.items()}
        ci_pct = conditional_inconsistency(top1)["inconsistency_pct"]
        ci_low, ci_high = bootstrap_ci_of_inconsistency(top1, n_boot, seed, ci)
        rows[slug] = {
            "records_rescored": len(cross["record_ids"]),
            "max_abs_d": cross_entries[slug]["stats"]["summary"]["max_abs_cohen_d"],
            "max_abs_d_rescored": max_abs_d,
            "max_abs_d_low": d_low,
            "max_abs_d_high": d_high,
            "queries_rescored": len(within["query_ids"]),
            "inconsistency_pct": within_entries[slug]["conditional_inconsistency"][
                "inconsistency_pct"
            ],
            "inconsistency_pct_rescored": ci_pct,
            "inconsistency_low": ci_low,
            "inconsistency_high": ci_high,
            "control_max_abs_diff": record["control"]["max_abs_diff"],
        }
    return _frame(rows)


def family_table(
    cross_entries: dict[str, Any],
    within_entries: dict[str, Any],
    threshold: float = MEDIUM_EFFECT,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    alpha: float = FAMILY_ALPHA,
) -> pd.DataFrame:
    """Test the two claims made across the roster, with a Holm correction.

    Args:
        cross_entries: The cross-query entry of each model, by slug.
        within_entries: The within-query entry of each model, by slug.
        threshold: The effect size the score-axis claim must exceed.
        n_boot: The number of bootstrap replicates.
        seed: The seed for the resample.
        alpha: The family-wise error rate.

    Returns:
        One row per model: each statistic, its raw and Holm-adjusted p-value,
        and whether the claim survives the correction.
    """
    rows = {}
    for slug, entry in cross_entries.items():
        replicates = bootstrap_max_abs_d_replicates(entry["scores"], n_boot, seed)
        rows[slug] = {
            "max_abs_d": entry["stats"]["summary"]["max_abs_cohen_d"],
            "p_above_threshold": bootstrap_p_above(replicates, threshold),
        }
    for slug, entry in within_entries.items():
        top1 = {
            name: np.asarray(ranks) == 1.0
            for name, ranks in entry["reciprocal_ranks"].items()
        }
        replicates = bootstrap_inconsistency_replicates(top1, n_boot, seed)
        rows.setdefault(slug, {}).update(
            {
                "inconsistency_pct": entry["conditional_inconsistency"][
                    "inconsistency_pct"
                ],
                "p_above_zero": bootstrap_p_above(replicates, 0.0),
            }
        )
    frame = _frame(rows)
    for raw, adjusted, holds in (
        ("p_above_threshold", "holm_threshold", "above_threshold"),
        ("p_above_zero", "holm_zero", "inconsistent"),
    ):
        if raw in frame:
            frame[adjusted] = holm(frame[raw])
            frame[holds] = frame[adjusted] < alpha
    return frame
