"""Measuring one fitted head against both axes of the study."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fsr.head_probe.store import FeatureStore
from fsr.head_probe.train import score_features
from fsr.metrics import format_sensitivity_summary, reciprocal_ranks

EVAL_CHUNK = 256


def score_store(
    head: nn.Module,
    store: FeatureStore,
    formats: Sequence[str],
    n_negatives: int,
    chunk: int = EVAL_CHUNK,
) -> tuple[dict[str, list[float]], np.ndarray]:
    """Score every record of a cache with one head.

    Args:
        head: The head to score with, in evaluation mode.
        store: The cached representations.
        formats: The formats to score, in the order wanted.
        n_negatives: The negatives taken from each record.
        chunk: The records scored at one time.

    Returns:
        The gold score of each record by format and the negative scores.
    """
    gold_rows: list[np.ndarray] = []
    negative_rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(store), chunk):
            rows = range(start, min(start + chunk, len(store)))
            gold, negatives = store.batch(list(rows), formats, n_negatives)
            gold_rows.append(score_features(head, gold).numpy())
            negative_rows.append(score_features(head, negatives).numpy())
    gold_scores = np.concatenate(gold_rows)
    return (
        {name: gold_scores[:, i].tolist() for i, name in enumerate(formats)},
        np.concatenate(negative_rows),
    )


def mrr_per_format(
    gold_scores: dict[str, list[float]],
    negative_scores: np.ndarray,
    formats: Sequence[str],
) -> dict[str, float]:
    """Return the mean reciprocal rank of the gold passage, by format.

    Args:
        gold_scores: The gold score of each record, by format.
        negative_scores: The negative scores.
        formats: The formats of the format axis, in order.

    Returns:
        The mean reciprocal rank of each format.
    """
    return {
        name: float(reciprocal_ranks(gold_scores[name], negative_scores[:, i]).mean())
        for i, name in enumerate(formats)
    }


def frontier_point(
    head: nn.Module,
    store: FeatureStore,
    formats: Sequence[str],
    n_negatives: int,
    seed: int = 0,
) -> dict[str, Any]:
    """Measure one fitted head on the score axis and the ranking guardrail.

    Args:
        head: The head to measure, in evaluation mode.
        store: The cached representations of the measurement split.
        formats: The formats to measure across.
        n_negatives: The negatives taken from each record.
        seed: The seed for the rank-stability sample.

    Returns:
        The sensitivity summary, the mean reciprocal rank of each format, and
        the two headline values.
    """
    gold_scores, negative_scores = score_store(head, store, formats, n_negatives)
    summary = format_sensitivity_summary(gold_scores, seed=seed, formats=formats)
    per_format = mrr_per_format(gold_scores, negative_scores, formats)
    return {
        "max_abs_cohen_d": summary["summary"]["max_abs_cohen_d"],
        "mean_mrr": float(np.mean(list(per_format.values()))),
        "min_mrr": min(per_format.values()),
        "format_sensitivity": summary,
        "mrr_per_format": per_format,
        "n_records": len(store),
    }
