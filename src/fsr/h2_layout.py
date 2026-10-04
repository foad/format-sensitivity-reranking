"""The directory layout and the file names of the mitigation-phase artefacts."""

from __future__ import annotations

from pathlib import Path

H2_SUBDIR = "h2"
TRAIN_SUBDIR = "train"
SELECTION_SUBDIR = "selection"
COMPARISON_SUBDIR = "comparison"
ADAPTER_NAME = "adapter"
ADAPTER_CONFIG_NAME = "adapter_config.json"

BASE_ARM = "base"
PHASE1_FOLD = "5fmt"
SPLITS = ("dev", "test", "prose")
AXES = ("cross", "within", "mrr")
SWEEPS = ("lambda", "rank")


def h2_dir(data_root: Path) -> Path:
    """Return the directory that holds every mitigation-phase artefact."""
    return data_root / H2_SUBDIR


def format_lambda(value: float) -> str:
    """Return the invariance weight as it appears in a file name.

    Args:
        value: The weight.

    Returns:
        The shortest form that reads back as the same number.
    """
    return f"{value:g}"


def arm(held_out: str | None, lambda_inv: float, rank: int | None = None) -> str:
    """Return the name of one trained condition.

    Args:
        held_out: The format withheld from training, or None for the
            five-format phase.
        lambda_inv: The weight of the invariance term.
        rank: The adapter rank, when the run is part of a rank sweep.

    Returns:
        The name, such as `5fmt_lam0.1`, `yaml_lam0` or `5fmt_lam0.1_r8`.
    """
    fold = held_out or PHASE1_FOLD
    name = f"{fold}_lam{format_lambda(lambda_inv)}"
    return f"{name}_r{rank}" if rank is not None else name


def train_dir(data_root: Path, model: str, arm_name: str) -> Path:
    """Return the directory one training run writes to.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, from `arm`.

    Returns:
        The directory.
    """
    return h2_dir(data_root) / TRAIN_SUBDIR / f"{model}_{arm_name}"


def adapter_dir(data_root: Path, model: str, arm_name: str) -> Path:
    """Return the adapter directory of one training run.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, from `arm`.

    Returns:
        The directory the adapter weights are saved to.
    """
    return train_dir(data_root, model, arm_name) / ADAPTER_NAME


def result_path(
    data_root: Path, split: str, axis: str, model: str, arm_name: str
) -> Path:
    """Return the file one evaluation writes.

    Args:
        data_root: The corpus directory.
        split: The record set, one of SPLITS.
        axis: The measurement, one of AXES.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The file.

    Raises:
        ValueError: If the split or the axis is not a known name.
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of: {SPLITS}")
    if axis not in AXES:
        raise ValueError(f"unknown axis {axis!r}, expected one of: {AXES}")
    return h2_dir(data_root) / f"{split}_{axis}_{model}_{arm_name}.json"


def selection_path(data_root: Path, model: str, sweep: str) -> Path:
    """Return the file one selection writes.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        sweep: The swept quantity, one of SWEEPS.

    Returns:
        The file.

    Raises:
        ValueError: If the sweep is not a known name.
    """
    if sweep not in SWEEPS:
        raise ValueError(f"unknown sweep {sweep!r}, expected one of: {SWEEPS}")
    return h2_dir(data_root) / SELECTION_SUBDIR / f"{model}_{sweep}.json"


def comparison_path(data_root: Path, model: str, contrast: str) -> Path:
    """Return the file one comparison writes.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        contrast: The arms being compared, such as `phase1` or `prose`.

    Returns:
        The file.
    """
    return h2_dir(data_root) / COMPARISON_SUBDIR / f"{model}_{contrast}.json"


def adapter_config_path(data_root: Path, model: str, arm_name: str) -> Path:
    """Return the file whose presence means an adapter was saved in full.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        arm_name: The trained condition, from `arm`.

    Returns:
        The configuration file inside the adapter directory.
    """
    return adapter_dir(data_root, model, arm_name) / ADAPTER_CONFIG_NAME
