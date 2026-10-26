"""Measuring a score tensor under each way of handling mixed serialisations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from fsr.metrics import reciprocal_ranks
from fsr.standardisation.conditions import realise, uniform_assignment
from fsr.standardisation.store import GOLD, ScoreTensor


def quality(realised: np.ndarray) -> dict[str, float]:
    """Return the ranking quality of one realised candidate list.

    Args:
        realised: One score per query and candidate, the gold first.

    Returns:
        The mean reciprocal rank and the share of queries the gold leads.
    """
    ranks = reciprocal_ranks(realised[:, GOLD], realised[:, GOLD + 1 :])
    return {"mrr": float(ranks.mean()), "top1": float((ranks == 1.0).mean())}


def uniform_table(tensor: ScoreTensor) -> pd.DataFrame:
    """Measure the tensor once per format, as standardising to it would.

    Args:
        tensor: The scores of one condition.

    Returns:
        One row per format, holding the ranking quality of that format.
    """
    rows = {}
    for index, name in enumerate(tensor.formats):
        assignment = uniform_assignment(tensor.n_queries, tensor.n_candidates, index)
        rows[name] = quality(realise(tensor.scores, assignment))
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.name = "format"
    return frame


def mixed_lead(tensor: ScoreTensor) -> np.ndarray:
    """Return the chance the gold leads when each candidate draws its format.

    A cross-encoder scores one pair at a time, so the candidates of a query
    are independent. Holding the gold at one format, each negative stays
    below it in a known share of the formats, and the shares multiply.

    Args:
        tensor: The scores of one condition.

    Returns:
        One probability per query.
    """
    gold = tensor.gold[:, None, :, None]
    negatives = tensor.negatives[:, :, None, :]
    below = (negatives <= gold).mean(axis=3)
    return below.prod(axis=1).mean(axis=1)


def inconsistency(leads: np.ndarray) -> dict[str, Any]:
    """Count the queries whose answer depends on the format.

    Args:
        leads: One row per query, one column per format.

    Returns:
        The answerable pool, the queries that lead under some formats and
        not others, and that count as a share of the pool.
    """
    answerable = leads.any(axis=1)
    inconsistent = answerable & ~leads.all(axis=1)
    pool = int(answerable.sum())
    return {
        "n_queries": len(leads),
        "n_answerable": pool,
        "n_inconsistent": int(inconsistent.sum()),
        "inconsistency_pct": (
            100.0 * float(inconsistent.sum()) / pool if pool else float("nan")
        ),
    }


def oracle_band(leads: np.ndarray) -> dict[str, float]:
    """Bound what any choice of format can do to the leading passage.

    Args:
        leads: One row per query, one column per format.

    Returns:
        The share of queries the gold leads under every format, under the
        best single format, and under the best format of each query.
    """
    n_queries = len(leads)
    return {
        "floor": float(leads.all(axis=1).mean()),
        "best_format": float(leads.mean(axis=0).max()),
        "ceiling": float(leads.any(axis=1).mean()),
        "headroom": float(leads.any(axis=1).mean() - leads.mean(axis=0).max()),
        "n_queries": n_queries,
    }


def uniform_leads(tensor: ScoreTensor) -> np.ndarray:
    """Report whether the gold leads, once per format.

    Args:
        tensor: The scores of one condition.

    Returns:
        One row per query, one column per format.
    """
    return np.stack(
        [
            leads_of(
                realise(
                    tensor.scores,
                    uniform_assignment(tensor.n_queries, tensor.n_candidates, index),
                )
            )
            for index in range(tensor.n_formats)
        ],
        axis=1,
    )


def leads_of(realised: np.ndarray) -> np.ndarray:
    """Report whether the gold outscores every negative of its query.

    Args:
        realised: One score per query and candidate, the gold first.

    Returns:
        One flag per query. A tie counts for the gold.
    """
    return realised[:, GOLD] >= realised[:, GOLD + 1 :].max(axis=1)


def margin(tensor: ScoreTensor) -> np.ndarray:
    """Return the gold's lead over its best negative, averaged over formats.

    Args:
        tensor: The scores of one condition.

    Returns:
        One margin per query.
    """
    mean_format = tensor.scores.mean(axis=2)
    return mean_format[:, GOLD] - mean_format[:, GOLD + 1 :].max(axis=1)
