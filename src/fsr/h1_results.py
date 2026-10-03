"""H1 result file reader and table builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from fsr.models.registry import MODELS, by_slug

RESULTS_SUBDIR = "h1"
AXES = ("cross", "within")
MODES = ("with_body", "metadata_only")
SPLITS = ("test", "nq_val", "all")


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
