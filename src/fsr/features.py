"""Penultimate representations, captured from the scoring forward pass."""

from __future__ import annotations

from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from fsr.scoring import MAX_TOKENS, score_batch


def classifier_input_module(model: Any) -> nn.Module:
    """Return the module whose input is the representation the head consumes.

    Args:
        model: The loaded cross-encoder.

    Returns:
        The classifier when it is a single linear layer, otherwise the first
        linear layer inside it.

    Raises:
        RuntimeError: If the model has no classifier, or the classifier holds
            no linear layer.
    """
    if not hasattr(model, "classifier"):
        raise RuntimeError("model has no `classifier` attribute")
    classifier = model.classifier
    if isinstance(classifier, nn.Linear):
        return classifier
    for module in classifier.modules():
        if isinstance(module, nn.Linear):
            return module
    raise RuntimeError(
        f"model.classifier is {type(classifier).__name__} and holds no "
        "nn.Linear, so the representation it consumes cannot be located"
    )


@contextmanager
def capture_classifier_input(model: Any) -> Generator[list[torch.Tensor]]:
    """Collect the representation each forward pass gives to the head.

    Args:
        model: The loaded cross-encoder.

    Yields:
        One tensor per batch, in call order, on the host in float32.
    """
    captured: list[torch.Tensor] = []

    def hook(_module: nn.Module, inputs: tuple, _output: Any) -> None:
        captured.append(inputs[0].detach().float().cpu())

    handle = classifier_input_module(model).register_forward_hook(hook)
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
    """Score every pair and keep the representation the head consumed.

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
    with capture_classifier_input(model) as captured:
        scores = score_batch(
            model, tokenizer, pairs, batch_size, device, max_tokens, on_batch
        )
    for batch in captured:
        if batch.ndim != 2:
            raise RuntimeError(
                f"the head consumed a {batch.ndim}-dimensional tensor, "
                "expected one vector per pair"
            )
    features = torch.cat(captured).numpy() if captured else np.empty((0, 0))
    if len(features) != len(scores):
        raise RuntimeError(
            f"captured {len(features)} representations for {len(scores)} pairs"
        )
    return features, scores
