"""Head-probe result file reader and table builder."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsr.h2_layout import frontier_path
from fsr.head_probe.heads import GELU, LINEAR, TANH, WIDE_LINEAR, is_affine, is_bounded
from fsr.models.registry import BASE_MODELS, by_slug

REPORTED_HEADS = (LINEAR, WIDE_LINEAR, GELU, TANH)
NOISE_MULTIPLE = 2.0


def load_frontier(data_root: Path, model: str) -> dict[str, Any]:
    """Load the head sweep of one model.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The frontier.
    """
    return json.loads(frontier_path(data_root, model).read_text())


def available(data_root: Path) -> list[str]:
    """Return the models whose frontier is present, in roster order.

    Args:
        data_root: The corpus directory.

    Returns:
        The registry slugs.
    """
    return [
        model.slug
        for model in BASE_MODELS
        if frontier_path(data_root, model.slug).exists()
    ]


def points_frame(
    data_root: Path, model: str, heads: Sequence[str] = REPORTED_HEADS
) -> pd.DataFrame:
    """Return every fit of one model, one row each.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        heads: The arms to keep, in the order wanted.

    Returns:
        One row per head, weight and seed.

    Raises:
        ValueError: If the frontier holds none of the named arms.
    """
    rows = [
        {
            "head": point["head"],
            "weight": float(point["lambda_inv"]),
            "seed": int(point["seed"]),
            "parameters": int(point["parameters"]),
            "max_abs_d": float(point["max_abs_cohen_d"]),
            "mean_mrr": float(point["mean_mrr"]),
            "min_mrr": float(point["min_mrr"]),
        }
        for point in load_frontier(data_root, model)["points"]
        if point["head"] in heads
    ]
    if not rows:
        raise ValueError(f"{model} holds no fit of any of: {tuple(heads)}")
    order = pd.CategoricalDtype(list(heads), ordered=True)
    frame = pd.DataFrame(rows).astype({"head": order})
    return frame.sort_values(["head", "weight", "seed"], ignore_index=True)


def arm_table(
    data_root: Path, model: str, heads: Sequence[str] = REPORTED_HEADS
) -> pd.DataFrame:
    """Tabulate the arms one sweep covered.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        heads: The arms to keep, in the order wanted.

    Returns:
        One row per head, holding the parameter count, whether the head
        composes to an affine function, and whether its activation saturates.
    """
    frame = points_frame(data_root, model, heads)
    table = frame.groupby("head", observed=True)["parameters"].first().to_frame()
    table["affine"] = [is_affine(name) for name in table.index]
    table["bounded"] = [is_bounded(name) for name in table.index]
    return table


def frontier_frame(
    data_root: Path, model: str, heads: Sequence[str] = REPORTED_HEADS
) -> pd.DataFrame:
    """Average one sweep over its seeds.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        heads: The arms to keep, in the order wanted.

    Returns:
        One row per head and weight, holding the mean and the deviation of
        the score axis and of the ranking quality over the seeds. A deviation
        taken from one seed is NaN.
    """
    grouped = points_frame(data_root, model, heads).groupby(
        ["head", "weight"], observed=True
    )
    return grouped.agg(
        max_abs_d=("max_abs_d", "mean"),
        max_abs_d_sd=("max_abs_d", "std"),
        mean_mrr=("mean_mrr", "mean"),
        min_mrr=("min_mrr", "mean"),
        seeds=("seed", "size"),
    )


def weights_of(data_root: Path, model: str) -> list[float]:
    """Return the weights one sweep covered, in ascending order.

    Args:
        data_root: The corpus directory.
        model: The registry slug.

    Returns:
        The weights.
    """
    return sorted(float(value) for value in load_frontier(data_root, model)["lambdas"])


def activation_contrast(
    data_root: Path, slugs: Sequence[str], noise: float = NOISE_MULTIPLE
) -> pd.DataFrame:
    """Contrast the two activations the roster carries, on each representation.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        noise: The multiple of the pooled seed deviation the band covers.

    Returns:
        One row per model and weight, holding the score axis of each
        activation, the difference, the band, and whether the difference
        clears the band. A negative difference favours tanh.
    """
    rows = []
    for slug in slugs:
        frame = frontier_frame(data_root, slug, (GELU, TANH))
        for weight in weights_of(data_root, slug):
            gelu = frame.loc[(GELU, weight)]
            tanh = frame.loc[(TANH, weight)]
            spread = float(
                np.sqrt((gelu["max_abs_d_sd"] ** 2 + tanh["max_abs_d_sd"] ** 2) / 2.0)
            )
            difference = float(tanh["max_abs_d"] - gelu["max_abs_d"])
            rows.append(
                {
                    "model": by_slug(slug).label,
                    "weight": weight,
                    "gelu": float(gelu["max_abs_d"]),
                    "tanh": float(tanh["max_abs_d"]),
                    "tanh_minus_gelu": difference,
                    "band": noise * spread,
                    "resolved": bool(abs(difference) > noise * spread),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "weight"])


def contrast_verdict(contrast: pd.DataFrame) -> dict[str, Any]:
    """Count which activation leads over the cells that clear the band.

    Args:
        contrast: The frame from `activation_contrast`.

    Returns:
        The resolved count, the cell count, and the resolved cells that
        favour each activation.
    """
    resolved = contrast[contrast["resolved"]]
    return {
        "resolved": len(resolved),
        "cells": len(contrast),
        "tanh_lower": int((resolved["tanh_minus_gelu"] < 0).sum()),
        "gelu_lower": int((resolved["tanh_minus_gelu"] > 0).sum()),
    }


def capacity_contrast(
    data_root: Path, slugs: Sequence[str], noise: float = NOISE_MULTIPLE
) -> pd.DataFrame:
    """Contrast the one-layer head against the two-layer head of equal width.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        noise: The multiple of the pooled seed deviation the band covers.

    Returns:
        One row per model and weight, holding the score axis of each head,
        the difference, the band, and whether the difference clears the band.
        A negative difference favours the two-layer head.
    """
    rows = []
    for slug in slugs:
        frame = frontier_frame(data_root, slug, (LINEAR, WIDE_LINEAR))
        for weight in weights_of(data_root, slug):
            thin = frame.loc[(LINEAR, weight)]
            wide = frame.loc[(WIDE_LINEAR, weight)]
            spread = float(
                np.sqrt((thin["max_abs_d_sd"] ** 2 + wide["max_abs_d_sd"] ** 2) / 2.0)
            )
            difference = float(wide["max_abs_d"] - thin["max_abs_d"])
            rows.append(
                {
                    "model": by_slug(slug).label,
                    "weight": weight,
                    "linear": float(thin["max_abs_d"]),
                    "wide_linear": float(wide["max_abs_d"]),
                    "wide_minus_linear": difference,
                    "band": noise * spread,
                    "resolved": bool(abs(difference) > noise * spread),
                }
            )
    return pd.DataFrame(rows).set_index(["model", "weight"])


def mrr_table(
    data_root: Path, slugs: Sequence[str], heads: Sequence[str] = REPORTED_HEADS
) -> pd.DataFrame:
    """Tabulate the ranking quality of every arm, one column per weight.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs to report.
        heads: The arms to keep, in the order wanted.

    Returns:
        One row per model and head, one column per weight.
    """
    frames = []
    for slug in slugs:
        frame = frontier_frame(data_root, slug, heads)["mean_mrr"].unstack("weight")
        frame.index = pd.MultiIndex.from_product(
            [[by_slug(slug).label], frame.index], names=["model", "head"]
        )
        frames.append(frame)
    return pd.concat(frames)
