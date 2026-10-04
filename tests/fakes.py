"""Deterministic tokenizers for tests."""

from __future__ import annotations

import re
import types
from typing import Any

import torch


class WordTokenizer:
    """The tokenizer makes one token from each run of non-whitespace characters."""

    def __call__(
        self, text: str, text_pair: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Tokenize the text, or a text pair, and return the encoding."""
        if text_pair is not None:
            text = f"{text} {text_pair}"
        spans = [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        out: dict[str, Any] = {"input_ids": list(range(len(spans)))}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = spans
        return out


class CharTokenizer:
    """The tokenizer makes one token from each character.

    A token budget therefore equals a character budget.
    """

    def __call__(
        self, text: str, text_pair: str | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Tokenize the text, or a text pair, and return the encoding."""
        if text_pair is not None:
            text = f"{text} {text_pair}"
        out: dict[str, Any] = {"input_ids": list(range(len(text)))}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class Encoding(dict):
    """A tokenizer output that accepts the device move a scorer performs."""

    def to(self, _device: str) -> Encoding:
        """Return the encoding unchanged."""
        return self


class PairTokenizer:
    """A tokenizer that encodes query and passage pairs into a tensor batch."""

    def __call__(self, queries, _passages=None, **_kwargs) -> Encoding:
        """Return an encoding whose batch size matches the queries."""
        return Encoding(input_ids=torch.zeros(len(queries), 4, dtype=torch.long))


class LogitModel:
    """A model that returns deterministic logits of a chosen shape."""

    def __init__(self, shape: tuple[int, ...] = (1,)) -> None:
        """Store the logit shape this model returns."""
        self.shape = shape

    def __call__(self, **kwargs) -> Any:
        """Return a namespace holding the logits for this batch."""
        batch = kwargs["input_ids"].shape[0]
        generator = torch.Generator().manual_seed(batch * 7 + len(self.shape))
        size = (batch, *self.shape) if self.shape else (batch,)
        return types.SimpleNamespace(logits=torch.rand(size, generator=generator))


class HashingPairTokenizer:
    """A tokenizer that serves both the budget call and the scoring call."""

    def __init__(self, width: int = 4) -> None:
        """Store the encoding width."""
        self.width = width

    def __call__(self, queries, passages=None, **kwargs) -> Any:
        """Return word offsets for one text, or an encoded batch."""
        if isinstance(queries, str):
            return WordTokenizer()(queries, passages, **kwargs)
        passages = passages if passages is not None else [""] * len(queries)
        rows = []
        for query, passage in zip(queries, passages, strict=True):
            text = f"{query}|{passage}"
            buckets = [0] * self.width
            for index, character in enumerate(text):
                buckets[index % self.width] += ord(character)
            rows.append(buckets)
        return Encoding(input_ids=torch.tensor(rows, dtype=torch.long))


class ScoringModel(torch.nn.Module):
    """A differentiable scorer with trainable parameters."""

    def __init__(self, width: int = 4) -> None:
        """Build the scoring head."""
        super().__init__()
        self.classifier = torch.nn.Linear(width, 1)

    def forward(self, input_ids=None, **_kwargs) -> Any:
        """Return one logit per row of the batch."""
        scaled = input_ids.to(torch.float32) / 1000.0
        return types.SimpleNamespace(logits=self.classifier(scaled))
