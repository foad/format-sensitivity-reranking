"""The encoder representation a head reads, captured from the scoring pass."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fsr.scoring import MAX_TOKENS, score_batch

CLS_POSITION = 0


def encoder_module(model: Any) -> nn.Module:
    """Return the module whose output holds the representation a head reads.

    Args:
        model: The loaded cross-encoder.

    Returns:
        The base transformer, below every pooling and classification layer.

    Raises:
        RuntimeError: If the model exposes no base transformer.
    """
    base = getattr(model, "base_model", None)
    if base is None or base is model:
        raise RuntimeError(
            f"{type(model).__name__} exposes no base transformer, so the "
            "representation its head reads cannot be located"
        )
    return base


@contextmanager
def capture_encoder_output(model: Any) -> Generator[list[torch.Tensor]]:
    """Collect the first-position representation of each forward pass.

    Args:
        model: The loaded cross-encoder.

    Yields:
        One tensor per batch, in call order, on the host in float32.
    """
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, _inputs: tuple, output: Any) -> None:
        hidden = output[0]
        captured.append(hidden[:, CLS_POSITION].detach().float().cpu())

    handle = encoder_module(model).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def features_and_scores(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
    device: str,
    max_tokens: int = MAX_TOKENS,
    on_batch: Callable[[int], None] | None = None,
) -> tuple[np.ndarray, list[float]]:
    """Score every pair and keep the representation its head read.

    Args:
        model: The loaded cross-encoder, in evaluation mode.
        tokenizer: The tokenizer that matches the model.
        pairs: The query and passage pairs.
        batch_size: The number of pairs to encode at once.
        device: The device to move each batch to.
        max_tokens: The encoding length at which a pair is truncated.
        on_batch: Called with the number of pairs scored so far, after each
            batch.

    Returns:
        The representations, one row per pair, and the score of each pair.

    Raises:
        RuntimeError: If the captured representation is not one vector per
            pair.
    """
    with capture_encoder_output(model) as captured:
        scores = score_batch(
            model, tokenizer, pairs, batch_size, device, max_tokens, on_batch
        )
    for batch in captured:
        if batch.ndim != 2:
            raise RuntimeError(
                f"the encoder gave a {batch.ndim}-dimensional representation, "
                "expected one vector per pair"
            )
    features = torch.cat(captured).numpy() if captured else np.empty((0, 0))
    if len(features) != len(scores):
        raise RuntimeError(
            f"captured {len(features)} representations for {len(scores)} pairs"
        )
    return features, scores
