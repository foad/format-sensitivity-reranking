"""Splitting the records and rescaling a representation by content differences."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

DEFAULT_SPLIT_SEED = 0
DEFAULT_SHRINKAGE = 1e-3


def halves(n: int, seed: int = DEFAULT_SPLIT_SEED) -> tuple[np.ndarray, np.ndarray]:
    """Return a random split of n indices into two halves.

    Args:
        n: The number of indices.
        seed: The seed of the split.

    Returns:
        The two halves.
    """
    order = np.random.default_rng(seed).permutation(n)
    return np.sort(order[: n // 2]), np.sort(order[n // 2 :])


def content_differences(
    passages: np.ndarray, columns: Sequence[int], seed: int = DEFAULT_SPLIT_SEED
) -> np.ndarray:
    """Pair the passages at random and return each pair's difference per format.

    Each difference holds the format fixed, so it is a change of content alone.

    Args:
        passages: The representations, shaped (passages, formats, width).
        columns: The format columns to difference within.
        seed: The seed of the pairing.

    Returns:
        The differences, shaped (passages // 2, len(columns), width).
    """
    order = np.random.default_rng(seed).permutation(len(passages))
    half = len(order) // 2
    left, right = order[:half], order[half : 2 * half]
    return passages[left][:, columns] - passages[right][:, columns]


def whitener(
    differences: np.ndarray, shrinkage: float = DEFAULT_SHRINKAGE
) -> np.ndarray:
    """Return the map that makes content differences equally spread in every direction.

    Args:
        differences: The content differences, with the width last.
        shrinkage: The share of the mean variance added to every direction, so
            a direction content never moves along stays finite.

    Returns:
        The symmetric matrix W with W C W close to the identity, for C the
        covariance of the differences.
    """
    flat = differences.reshape(-1, differences.shape[-1])
    covariance = flat.T @ flat / len(flat)
    ridge = shrinkage * np.trace(covariance) / covariance.shape[0]
    values, vectors = np.linalg.eigh(covariance + ridge * np.eye(len(covariance)))
    return (vectors / np.sqrt(values)) @ vectors.T
