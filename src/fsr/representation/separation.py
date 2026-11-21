"""The representation study's pair separations, set beside H1 and H2."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from fsr import h1_results, h2_results
from fsr.comparison import IN_TRAINING
from fsr.h2_layout import h2_dir
from fsr.models.registry import by_slug
from fsr.representation.robustness import range_table

SEPARATION_SUBDIR = ("format_geometry", "separation")
WINNER_OVER_CONTROL = "winner / lambda 0"
SETS = ("training", "held_out")
UNTRAINED = "untrained"
SPACES = ("raw", "whitened")
H1_SPLIT = "test"
H1_MODE = "with_body"
ROBUSTNESS_SUBDIR = ("format_geometry", "robustness")
POOLED = "all models"


def split_pairs(
    effects: np.ndarray, formats: Sequence[str], held_out: str
) -> dict[str, float]:
    """Return the mean effect of training pairs and of held-out pairs.

    Args:
        effects: A (formats, formats) effect matrix.
        formats: The format of each row and column.
        held_out: The held-out format.

    Returns:
        The mean over pairs of two training formats, and over pairs of the
        held-out format with a training format.
    """
    h = list(formats).index(held_out)
    others = [i for i in range(len(formats)) if i != h]
    within = [effects[a, b] for i, a in enumerate(others) for b in others[i + 1 :]]
    across = [effects[h, t] for t in others]
    return {"training": float(np.mean(within)), "held_out": float(np.mean(across))}


def pairs_path(
    data_root: Path, model: str, subdir: tuple[str, ...] = SEPARATION_SUBDIR
) -> Path:
    """Return the file holding one model's pair results."""
    return h2_dir(data_root).joinpath(*subdir) / f"{model}.json"


def load_pairs(
    data_root: Path, model: str, subdir: tuple[str, ...] = SEPARATION_SUBDIR
) -> dict[str, Any]:
    """Load one model's pair results.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        subdir: The results directory, under the H2 directory.

    Returns:
        The results.

    Raises:
        SystemExit: If the results are absent.
    """
    path = pairs_path(data_root, model, subdir)
    if not path.exists():
        raise SystemExit(f"no pair results at {path}; run the measuring script first.")
    return json.loads(path.read_text())


def ratio_frame(
    data_root: Path,
    slugs: Sequence[str],
    space: str,
    contrast: str = WINNER_OVER_CONTROL,
    subdir: tuple[str, ...] = SEPARATION_SUBDIR,
) -> pd.DataFrame:
    """Return each model's per-seed ratio for training and held-out pairs.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs, in the order to report.
        space: The measure, such as `representation` or
            `representation_consistent`.
        contrast: The ratio between conditions.
        subdir: The results directory, under the H2 directory.

    Returns:
        The data frame containing each model's per-seed ratios for training
        and held-out pairs.
    """
    rows = {}
    for slug in slugs:
        results = load_pairs(data_root, slug, subdir)
        entry = results["summary"][contrast][space]
        seeds = results["seeds"]
        row: dict[str, float] = {}
        for name in SETS:
            values = np.asarray(entry[name]["per_seed"], dtype=float)
            row[f"{name}_mean"] = float(values.mean())
            row[f"{name}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            for seed, value in zip(seeds, values, strict=True):
                row[f"{name} seed {seed}"] = float(value)
        row["difference"] = row["held_out_mean"] - row["training_mean"]
        rows[by_slug(slug).label] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def pair_frame(data_root: Path, slugs: Sequence[str]) -> pd.DataFrame:
    """Return every untrained model's pair separation beside H1's |d_z|.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs, in the order to report.

    Returns:
        One row per model and format pair: the raw and whitened separation
        of the gold passages, and H1's absolute paired Cohen's d.
    """
    h1 = h1_results.pairwise_table(
        h1_results.load_axis(data_root, H1_SPLIT, "cross", H1_MODE)
    )
    lookup = {}
    for row in h1.itertuples():
        lookup[(row.model, row.left, row.right)] = row.abs_d
        lookup[(row.model, row.right, row.left)] = row.abs_d
    rows = []
    for slug in slugs:
        results = load_pairs(data_root, slug, SEPARATION_SUBDIR)
        formats = results["formats"]
        matrices = results["adapters"][UNTRAINED]
        label = by_slug(slug).label
        for a in range(len(formats)):
            for b in range(a + 1, len(formats)):
                rows.append(
                    {
                        "model": label,
                        "pair": f"{formats[a]} - {formats[b]}",
                        **{space: matrices[space][a][b] for space in SPACES},
                        "h1_abs_d": lookup[(label, formats[a], formats[b])],
                    }
                )
    return pd.DataFrame(rows)


def full_space_separation(
    data_root: Path, slugs: Sequence[str], space: str = "whitened"
) -> pd.Series:
    """Return each untrained model's mean separation over its format pairs.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs, in the order to report.
        space: `raw` or `whitened`.

    Returns:
        The mean gold-pair separation over every pair of formats, by model
        label, measured in the whole space rather than a projection of it.
    """
    values = {}
    for slug in slugs:
        matrix = load_pairs(data_root, slug, SEPARATION_SUBDIR)["adapters"][UNTRAINED][
            space
        ]
        n = len(matrix)
        upper = [matrix[a][b] for a in range(n) for b in range(a + 1, n)]
        values[by_slug(slug).label] = sum(upper) / len(upper)
    return pd.Series(values)


def within_model_ranks(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return each pair's rank within its model, by H1's |d_z| and by separation.

    Args:
        pairs: The table `pair_frame` returns.

    Returns:
        The same rows with the model and pair, and the rank from 1 (smallest)
        of H1's |d_z| and of each separation among the model's pairs.
    """
    columns = ["h1_abs_d", *SPACES]
    ranks = pairs.groupby("model", sort=False)[columns].rank(method="average")
    return pd.concat([pairs[["model", "pair"]], ranks], axis=1)


def rank_frame(pairs: pd.DataFrame) -> pd.DataFrame:
    """Return the rank correlation of each separation with H1's |d_z|.

    Args:
        pairs: The table `pair_frame` returns.

    Returns:
        One row per model, in the order given, and a last row pooling every
        pair, with the Spearman correlation and its p-value for each space.
    """
    groups = [(model, group) for model, group in pairs.groupby("model", sort=False)]
    groups.append((POOLED, pairs))
    rows = {}
    for name, group in groups:
        row = {}
        for space in SPACES:
            result = spearmanr(group[space], group["h1_abs_d"])
            row[f"{space}_rho"] = float(result.statistic)
            row[f"{space}_p"] = float(result.pvalue)
        rows[name] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def load_robustness(data_root: Path) -> pd.DataFrame:
    """Return the whitening robustness table, one row per setting."""
    path = h2_dir(data_root).joinpath(*ROBUSTNESS_SUBDIR) / "summary.json"
    if not path.exists():
        raise SystemExit(
            f"no robustness results at {path}; run representation_robustness.py."
        )
    return range_table(json.loads(path.read_text()))


def mitigation_frame(
    data_root: Path, slugs: Sequence[str], seeds: Sequence[int]
) -> pd.DataFrame:
    """Return each model and seed's score-axis change beside its separation change.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs, in the order to report.
        seeds: The training seeds.

    Returns:
        One row per model and seed: the model label, the seed, the change in
        maximum |d|, and the percentage change in separation.
    """
    scores = h2_results.seed_score_table(data_root, slugs, subset=IN_TRAINING, n_boot=0)
    rows = []
    for slug in slugs:
        label = by_slug(slug).label
        results = load_pairs(data_root, slug, SEPARATION_SUBDIR)
        ratios = results["summary"][WINNER_OVER_CONTROL]["whitened"]["training"]
        by_seed = dict(zip(results["seeds"], ratios["per_seed"], strict=True))
        for seed in seeds:
            rows.append(
                {
                    "model": label,
                    "seed": seed,
                    "score_delta": float(scores.loc[label, f"seed {seed}"]),
                    "separation_change": 100.0 * (by_seed[seed] - 1.0),
                }
            )
    return pd.DataFrame(rows)
