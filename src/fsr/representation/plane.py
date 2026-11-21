"""A two-dimensional view of the direction each format moves a passage."""

from __future__ import annotations

from typing import Any

import numpy as np
import scipy.linalg

from fsr.representation.whitening import (
    DEFAULT_SHRINKAGE,
    DEFAULT_SPLIT_SEED,
    content_differences,
    halves,
    whitener,
)


def format_shifts(features: np.ndarray) -> np.ndarray:
    """Return each rendering less the mean of the passage's renderings.

    Args:
        features: The representations, shaped (records, candidates, formats,
            width).

    Returns:
        The shifts, shaped (passages, formats, width).
    """
    block = np.asarray(features, dtype=np.float64)
    flat = block.reshape(-1, block.shape[2], block.shape[-1])
    return flat - flat.mean(axis=1, keepdims=True)


def plane(shifts: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the plane through the formats' mean shifts.

    Args:
        shifts: The shifts, shaped (passages, formats, width).

    Returns:
        The two orthonormal directions, one per row, and the share of the
        mean shifts' spread they hold.
    """
    means = shifts.mean(axis=0)
    _, values, rows = np.linalg.svd(means, full_matrices=False)
    energy = values**2
    return rows[:2], float(energy[:2].sum() / energy.sum())


def discriminant_plane(
    shifts: np.ndarray, ridge: float = DEFAULT_SHRINKAGE
) -> tuple[np.ndarray, float]:
    """Return the plane that best separates the formats' shifts.

    Args:
        shifts: The shifts, shaped (passages, formats, width).
        ridge: The share of the mean within-format variance added to every
            direction, so the within-format spread can be inverted.

    Returns:
        The two orthonormal directions, one per row, and the share of the
        discriminant criterion they hold.
    """
    means = shifts.mean(axis=0)
    residual = (shifts - means[None]).reshape(-1, shifts.shape[-1])
    within = residual.T @ residual / len(residual)
    within += ridge * np.trace(within) / len(within) * np.eye(len(within))
    between = means.T @ means / len(means)
    values, vectors = scipy.linalg.eigh(between, within)
    order = np.argsort(values)[::-1]
    rows, _ = np.linalg.qr(vectors[:, order[:2]])
    kept = np.clip(values[order], 0.0, None)
    return rows.T, float(kept[:2].sum() / kept.sum())


def separation(points: np.ndarray) -> float:
    """Return the share of the points' spread that lies between the formats.

    Args:
        points: The projected shifts, shaped (passages, formats, 2).

    Returns:
        The spread of each format's mean about the overall mean over the
        spread of every point about it, from 0 (the formats overlap) to 1
        (each format is a single point).
    """
    centre = points.reshape(-1, points.shape[-1]).mean(axis=0)
    means = points.mean(axis=0)
    between = float(((means - centre) ** 2).sum()) * points.shape[0]
    total = float(((points - centre) ** 2).sum())
    return between / total if total > 0 else 0.0


def plane_view(
    features: np.ndarray,
    seed: int = DEFAULT_SPLIT_SEED,
    whiten: bool = False,
    shrinkage: float = DEFAULT_SHRINKAGE,
    method: str = "means",
    gold_only: bool = False,
) -> dict[str, Any]:
    """Fit the plane on half the records and project the other half onto it.

    Args:
        features: The representations, shaped (records, candidates, formats,
            width).
        seed: The seed of the record split.
        whiten: Whether to rescale the space first, by content differences
            fitted on the same half as the plane, so that directions every
            passage moves along do not dominate it.
        shrinkage: The share of the mean variance the whitening adds to every
            direction.
        method: `means` for the plane through the formats' mean shifts, or
            `discriminant` for the plane that best separates the formats.
        gold_only: Whether to fit and show the gold passages alone; the
            whitening is still fitted on every candidate.

    Returns:
        The projection results.
    """
    fit_rows, show_rows = halves(features.shape[0], seed)
    fit = np.asarray(features[fit_rows], dtype=np.float64)
    show = np.asarray(features[show_rows], dtype=np.float64)
    if whiten:
        flat = fit.reshape(-1, fit.shape[2], fit.shape[-1])
        white = whitener(
            content_differences(flat, list(range(fit.shape[2])), seed), shrinkage
        )
        fit, show = fit @ white, show @ white
    if gold_only:
        fit, show = fit[:, :1], show[:, :1]
    fitter = {"means": plane, "discriminant": discriminant_plane}[method]
    basis, held = fitter(format_shifts(fit))
    shown = format_shifts(show)
    points = shown @ basis.T
    return {
        "points": points,
        "means": points.mean(axis=0),
        "mean_share": held,
        "passage_share": float((points**2).sum() / (shown**2).sum()),
        "separation": separation(points),
    }


GOLD_COLUMN = 0


def pair_separation(
    features: np.ndarray,
    seed: int = DEFAULT_SPLIT_SEED,
    whiten: bool = False,
    shrinkage: float = DEFAULT_SHRINKAGE,
) -> np.ndarray:
    """Return how far each pair of formats separates the gold passages.

    Args:
        features: The representations, shaped (records, candidates, formats,
            width), gold first.
        seed: The seed of the record split and the content pairing.
        whiten: Whether to rescale by content differences first.
        shrinkage: The share of the mean variance the whitening adds to every
            direction.

    Returns:
        A symmetric (formats, formats) matrix with zeros on the diagonal.
    """
    fit_rows, show_rows = halves(features.shape[0], seed)
    show = np.asarray(features[show_rows, GOLD_COLUMN], dtype=np.float64)
    n = show.shape[1]
    if whiten:
        fit = np.asarray(features[fit_rows], dtype=np.float64)
        fit = fit.reshape(-1, n, fit.shape[-1])
        white = whitener(content_differences(fit, list(range(n)), seed), shrinkage)
        show = show @ white
    out = np.zeros((n, n))
    for a in range(n):
        for b in range(a + 1, n):
            diff = show[:, a] - show[:, b]
            mean = diff.mean(axis=0)
            spread = np.sqrt(((diff - mean) ** 2).sum(-1).mean())
            out[a, b] = out[b, a] = float(np.linalg.norm(mean) / spread)
    return out
