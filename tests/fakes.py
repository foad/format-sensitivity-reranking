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
