"""Saving and restoring the state a training run needs to resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

from fsr.corpus.files import atomic_path

CHECKPOINT_NAME = "checkpoint.pt"


class Stateful(Protocol):
    """An object that reports and accepts its own state, such as an optimiser."""

    def state_dict(self) -> dict[str, Any]:
        """Return the state."""
        ...

    def load_state_dict(self, state: dict[str, Any]) -> Any:
        """Replace the state."""
        ...


def should_checkpoint(step: int, every: int) -> bool:
    """Report whether a step is a checkpoint step.

    Args:
        step: The steps completed.
        every: The steps between checkpoints. 0 disables checkpointing.

    Returns:
        True when the run saves at this step.
    """
    if every <= 0:
        return False
    return step % every == 0


def capture_rng(sampler: np.random.Generator) -> dict[str, Any]:
    """Return every random state a resumed run must restore.

    Args:
        sampler: The generator that draws the training batches.

    Returns:
        The torch, numpy and sampler states.
    """
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
        "sampler": sampler.bit_generator.state,
    }


def restore_rng(state: dict[str, Any], sampler: np.random.Generator) -> None:
    """Put every random state back.

    Args:
        state: The states from capture_rng.
        sampler: The generator that draws the training batches.
    """
    torch.set_rng_state(state["torch"])
    cuda = state["torch_cuda"]
    if (
        cuda is not None
        and torch.cuda.is_available()
        and len(cuda) == torch.cuda.device_count()
    ):
        torch.cuda.set_rng_state_all(cuda)
    np.random.set_state(state["numpy"])
    sampler.bit_generator.state = state["sampler"]


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return the parameters a training run updates.

    Args:
        model: The adapted model.

    Returns:
        The trainable parameters, by name, detached and on the host.
    """
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Stateful,
    scheduler: Stateful,
    step: int,
    sampler: np.random.Generator,
) -> Path:
    """Write everything a resumed run needs, in one file.

    Args:
        path: The checkpoint file.
        model: The adapted model.
        optimizer: The optimiser.
        scheduler: The learning-rate scheduler.
        step: The steps completed.
        sampler: The generator that draws the training batches.

    Returns:
        The path written.
    """
    state = {
        "step": step,
        "model": trainable_state(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": capture_rng(sampler),
    }
    with atomic_path(path) as tmp:
        torch.save(state, tmp)
    return path


def load_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: Stateful,
    scheduler: Stateful,
    sampler: np.random.Generator,
) -> int:
    """Put a saved state back and report the step it holds.

    Args:
        path: The checkpoint file.
        model: The adapted model.
        optimizer: The optimiser.
        scheduler: The learning-rate scheduler.
        sampler: The generator that draws the training batches.

    Returns:
        The steps the saved run had completed.

    Raises:
        ValueError: If the checkpoint holds parameters the model does not.
    """
    # The state holds optimiser and generator objects as well as tensors.
    state = torch.load(path, map_location="cpu", weights_only=False)
    report = model.load_state_dict(state["model"], strict=False)
    if report.unexpected_keys:
        raise ValueError(
            f"checkpoint {path} holds parameters this model does not: "
            f"{sorted(report.unexpected_keys)}"
        )
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng(state["rng"], sampler)
    return int(state["step"])
