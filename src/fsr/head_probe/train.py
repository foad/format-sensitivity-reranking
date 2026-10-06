"""Fitting one classifier head to cached representations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from transformers import get_scheduler

from fsr.head_probe.store import FeatureStore
from fsr.training.batch import compute_loss

DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_STEPS = 2000
DEFAULT_WARMUP_STEPS = 100
DEFAULT_BATCH_SIZE = 64
DEFAULT_LOG_EVERY = 100


@dataclass(frozen=True)
class HeadProbeConfig:
    """The settings of one head fit.

    Attributes:
        lambda_inv: The weight of the invariance term.
        max_steps: The optimiser steps the fit takes.
        batch_size: The records in one step.
        n_negatives: The negatives taken from each record.
        lr: The peak learning rate.
        weight_decay: The decay applied by the optimiser.
        warmup_steps: The steps the schedule warms up over.
        seed: The seed for the initialisation and the record order.
        log_every: The steps between history entries.
    """

    lambda_inv: float
    max_steps: int = DEFAULT_MAX_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    n_negatives: int = 7
    lr: float = DEFAULT_LR
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    seed: int = 0
    log_every: int = DEFAULT_LOG_EVERY


@dataclass
class HeadProbeResult:
    """What one head fit produced.

    Attributes:
        steps: The optimiser steps taken.
        loss: The composite loss of the last step.
        rank_loss: The ranking term of the last step.
        inv_loss: The invariance term of the last step.
        history: One entry per logged step, each with step, loss, rank and inv.
    """

    steps: int
    loss: float
    rank_loss: float
    inv_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


def score_features(head: nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Return one score for every representation.

    Args:
        head: The head to score with.
        features: Representations whose last axis is the representation.

    Returns:
        The scores, with the representation axis removed.
    """
    return head(features).squeeze(-1)


def record_order(n_records: int, config: HeadProbeConfig) -> np.ndarray:
    """Return the record rows of every step, one row per batch position.

    Args:
        n_records: The records available.
        config: The settings of the fit.

    Returns:
        An array of shape (max_steps, batch_size).
    """
    rng = np.random.default_rng(config.seed)
    needed = config.max_steps * config.batch_size
    draws = []
    while sum(len(d) for d in draws) < needed:
        draws.append(rng.permutation(n_records))
    return np.concatenate(draws)[:needed].reshape(config.max_steps, config.batch_size)


def train_head(
    head: nn.Module,
    store: FeatureStore,
    config: HeadProbeConfig,
    formats: Sequence[str],
    device: str = "cpu",
    on_step: Callable[[int], None] | None = None,
) -> HeadProbeResult:
    """Fit one head to the cached representations of one model.

    Args:
        head: The head to fit, in training mode.
        store: The cached representations.
        config: The settings of the fit.
        formats: The formats the invariance term spans.
        device: The device to fit on.
        on_step: Called with the step number after each step.

    Returns:
        The result of the fit.

    Raises:
        ValueError: If the batch size exceeds the records available.
    """
    if config.batch_size > len(store):
        raise ValueError(
            f"the cache holds {len(store)} records, "
            f"a batch of {config.batch_size} was asked for"
        )
    torch.manual_seed(config.seed)
    head.to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=config.warmup_steps,
        num_training_steps=config.max_steps,
    )
    order = record_order(len(store), config)
    head.train()

    history: list[dict[str, float]] = []
    loss = rank = inv = torch.zeros(())
    for step in range(1, config.max_steps + 1):
        gold, negatives = store.batch(
            order[step - 1].tolist(), formats, config.n_negatives, device
        )
        loss, rank, inv = compute_loss(
            score_features(head, gold),
            score_features(head, negatives),
            config.lambda_inv,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % config.log_every == 0 or step == config.max_steps:
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "rank": float(rank),
                    "inv": float(inv),
                }
            )
        if on_step is not None:
            on_step(step)

    head.eval()
    return HeadProbeResult(
        steps=config.max_steps,
        loss=float(loss),
        rank_loss=float(rank),
        inv_loss=float(inv),
        history=history,
    )
