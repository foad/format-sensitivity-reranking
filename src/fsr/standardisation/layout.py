"""Where the standardisation study keeps its score tensors."""

from __future__ import annotations

from pathlib import Path

SUBDIR = "standardisation"
CANDIDATES_SUBDIR = "h1"


def standardisation_dir(data_root: Path) -> Path:
    """Return the directory holding the study's artefacts.

    Args:
        data_root: The corpus directory.

    Returns:
        The directory.
    """
    return data_root / "h2" / SUBDIR


def tensor_path(data_root: Path, split: str, model: str, arm_name: str) -> Path:
    """Return the file holding the scores of one condition.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The path.
    """
    return standardisation_dir(data_root) / f"{split}_scores_{model}_{arm_name}.npy"


def tensor_meta_path(data_root: Path, split: str, model: str, arm_name: str) -> Path:
    """Return the file describing the scores of one condition.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The path.
    """
    return tensor_path(data_root, split, model, arm_name).with_suffix(".json")


def candidates_path(data_root: Path, split: str, limit: int = 0) -> Path:
    """Return the candidate lists the answer-axis pipeline built.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.
        limit: The record cap the lists were built under, or 0 for all.

    Returns:
        The path.
    """
    suffix = f"_limit{limit}" if limit else ""
    return data_root / CANDIDATES_SUBDIR / f"{split}_candidates{suffix}.json"
