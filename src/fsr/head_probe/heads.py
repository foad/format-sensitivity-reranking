"""The classifier heads the probe compares, by activation."""

from __future__ import annotations

import torch
import torch.nn as nn

LINEAR = "linear"
WIDE_LINEAR = "wide_linear"
GELU = "gelu"
TANH = "tanh"
DEEP = "deep"
HEAD_NAMES = (LINEAR, WIDE_LINEAR, GELU, TANH, DEEP)

BOUNDED = (TANH,)

DEEP_WIDTH_FACTOR = 2


def _linear(dim: int) -> nn.Module:
    """Return a one-layer head, the capacity floor of the probe."""
    return nn.Linear(dim, 1)


def _wide_linear(dim: int) -> nn.Module:
    """Return a two-layer head with no activation between the layers."""
    return nn.Sequential(nn.Linear(dim, dim), nn.Linear(dim, 1))


def _gelu(dim: int) -> nn.Module:
    """Return the head shape mxbai carries, read from the first token."""
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))


def _tanh(dim: int) -> nn.Module:
    """Return the head shape the other five rerankers carry."""
    return nn.Sequential(nn.Linear(dim, dim), nn.Tanh(), nn.Linear(dim, 1))


def _deep(dim: int) -> nn.Module:
    """Return a head wider and deeper than any in the roster."""
    hidden = dim * DEEP_WIDTH_FACTOR
    return nn.Sequential(
        nn.Linear(dim, hidden),
        nn.Tanh(),
        nn.Linear(hidden, hidden),
        nn.Tanh(),
        nn.Linear(hidden, 1),
    )


BUILDERS = {
    LINEAR: _linear,
    WIDE_LINEAR: _wide_linear,
    GELU: _gelu,
    TANH: _tanh,
    DEEP: _deep,
}


def build_head(name: str, dim: int, seed: int = 0) -> nn.Module:
    """Return one head of the probe, with a reproducible initialisation.

    Args:
        name: The head name, one of HEAD_NAMES.
        dim: The width of the representation the head consumes.
        seed: The seed for the parameter initialisation.

    Returns:
        The head, in training mode.

    Raises:
        ValueError: If the name is not one the probe compares.
    """
    if name not in BUILDERS:
        raise ValueError(f"unknown head {name!r}, expected one of: {HEAD_NAMES}")
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        head = BUILDERS[name](dim)
    finally:
        torch.random.set_rng_state(state)
    return head


def parameter_count(head: nn.Module) -> int:
    """Return the number of trainable parameters of a head.

    Args:
        head: The head to measure.

    Returns:
        The count.
    """
    return sum(p.numel() for p in head.parameters() if p.requires_grad)


def is_affine(name: str) -> bool:
    """Report whether a head composes to an affine function of its input.

    Args:
        name: The head name.

    Returns:
        True when the head holds no activation between its layers.

    Raises:
        ValueError: If the name is not one the probe compares.
    """
    if name not in BUILDERS:
        raise ValueError(f"unknown head {name!r}, expected one of: {HEAD_NAMES}")
    return name in (LINEAR, WIDE_LINEAR)


def is_bounded(name: str) -> bool:
    """Report whether a head's activation has a finite range.

    Args:
        name: The head name.

    Returns:
        True when the activation saturates.

    Raises:
        ValueError: If the name is not one the probe compares.
    """
    if name not in BUILDERS:
        raise ValueError(f"unknown head {name!r}, expected one of: {HEAD_NAMES}")
    return name in BOUNDED
