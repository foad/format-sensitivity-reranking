"""Realising a score tensor under one way of handling mixed serialisations."""

from __future__ import annotations

import numpy as np

UNIFORM = "uniform"
MIXED = "mixed"
POLICIES = (UNIFORM, MIXED)


def uniform_assignment(
    n_queries: int, n_candidates: int, format_index: int
) -> np.ndarray:
    """Give every candidate of every query the same format.

    Args:
        n_queries: The number of queries.
        n_candidates: The number of candidates of each query.
        format_index: The column of the format axis to take.

    Returns:
        One format index per query and candidate.
    """
    return np.full((n_queries, n_candidates), format_index, dtype=int)


def mixed_assignment(
    n_queries: int,
    n_candidates: int,
    n_formats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one format for each candidate of each query.

    Args:
        n_queries: The number of queries.
        n_candidates: The number of candidates of each query.
        n_formats: The number of formats to draw from.
        rng: The source of the draw.

    Returns:
        One format index per query and candidate.
    """
    return rng.integers(0, n_formats, size=(n_queries, n_candidates))


def realise(scores: np.ndarray, assignment: np.ndarray) -> np.ndarray:
    """Take each candidate's score under the format it was given.

    Args:
        scores: The scores, shaped (queries, candidates, formats).
        assignment: One format index per query and candidate.

    Returns:
        One score per query and candidate.

    Raises:
        ValueError: If the assignment does not cover every candidate.
    """
    n_queries, n_candidates, _ = scores.shape
    if assignment.shape != (n_queries, n_candidates):
        raise ValueError(
            f"the assignment is {assignment.shape}, expected "
            f"{(n_queries, n_candidates)}"
        )
    rows = np.arange(n_queries)[:, None]
    columns = np.arange(n_candidates)[None, :]
    return scores[rows, columns, assignment]
