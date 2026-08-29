"""Deterministic tokenizers for tests."""

from __future__ import annotations

import re
from typing import Any


class WordTokenizer:
    """The tokenizer makes one token from each run of non-whitespace characters."""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Tokenize the text and return the encoding."""
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out: dict[str, Any] = {"input_ids": list(range(len(spans)))}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = spans
        return out


class CharTokenizer:
    """The tokenizer makes one token from each character.

    A token budget therefore equals a character budget.
    """

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Tokenize the text and return the encoding."""
        out: dict[str, Any] = {"input_ids": list(range(len(text)))}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class Encoding(dict):
    """A tokenizer output that accepts the device move a scorer performs."""

    def to(self, device: str) -> Encoding:
        """Return the encoding unchanged."""
        return self


class PairTokenizer:
    """A tokenizer that encodes query and passage pairs into a tensor batch."""

    def __call__(self, queries, passages=None, **kwargs) -> Encoding:
        """Return an encoding whose batch size matches the queries."""
        import torch

        return Encoding(input_ids=torch.zeros(len(queries), 4, dtype=torch.long))


class LogitModel:
    """A model that returns deterministic logits of a chosen shape."""

    def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
        """Store the logit shape this model returns."""
        self.shape = shape

    def __call__(self, **kwargs) -> Any:
        """Return a namespace holding the logits for this batch."""
        import types

        import torch

        batch = kwargs["input_ids"].shape[0]
        generator = torch.Generator().manual_seed(batch * 7 + len(self.shape))
        size = (batch, *self.shape) if self.shape else (batch,)
        return types.SimpleNamespace(logits=torch.rand(size, generator=generator))
