"""Passages that exceed a model's context after budgeting.

The body budget is the smallest over the roster, but the body is cut by counting
tokens with the one tokenizer that set the budget. Another tokenizer can need
more tokens for the same body, so a few renderings still exceed the context and
the model truncates the end of the body itself. This module finds those
renderings and cuts a body short enough to fit every tokenizer.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from fsr.formats import FORMATS, Renderer
from fsr.passages import (
    MAX_TOKENS,
    Tokenizer,
    compute_body_budget,
    truncate_body_semantic,
)

Passage = tuple[Sequence[tuple[str, str]], str]
Item = tuple[str, str, Sequence[Passage]]


def scan(
    items: Iterable[Item],
    tokenizers: Mapping[str, Tokenizer],
    *,
    max_tokens: int = MAX_TOKENS,
    formats: Mapping[str, Renderer] | None = None,
) -> dict[str, dict[str, Any]]:
    """Find the items holding a rendering longer than the context.

    Args:
        items: One (id, question, passages) triple per item, where each passage
            is its metadata pairs and its truncated body.
        tokenizers: The tokenizers to measure, by name.
        max_tokens: The model context length.
        formats: The renderers to measure. The default is FORMATS.

    Returns:
        For each tokenizer, the ids of the items with any rendering over the
        context, the count of such renderings, the count of renderings
        measured, and the median and largest excess in tokens. Both excesses
        are 0 when nothing overflows.
    """
    renderers = FORMATS if formats is None else formats
    ids: dict[str, list[str]] = {name: [] for name in tokenizers}
    excess: dict[str, list[int]] = {name: [] for name in tokenizers}
    total = 0
    for item_id, question, passages in items:
        hit = dict.fromkeys(tokenizers, False)
        for pairs, body in passages:
            for render in renderers.values():
                text = render(pairs, body)
                total += 1
                for name, tokenizer in tokenizers.items():
                    n = len(tokenizer(question, text)["input_ids"])
                    if n > max_tokens:
                        excess[name].append(n - max_tokens)
                        hit[name] = True
        for name, overflowed in hit.items():
            if overflowed:
                ids[name].append(item_id)
    return {
        name: {
            "ids": ids[name],
            "renderings_over": len(excess[name]),
            "renderings": total,
            "excess_median": float(np.median(excess[name])) if excess[name] else 0.0,
            "excess_max": max(excess[name], default=0),
        }
        for name in tokenizers
    }


def record_items(records: Iterable[Mapping[str, Any]]) -> list[Item]:
    """Return the score-axis records as scan items, one passage each.

    Args:
        records: Records carrying question, pairs, and truncated_body.

    Returns:
        One item per record.
    """
    return [
        (r["id"], r["question"], [(r["pairs"], r["truncated_body"])]) for r in records
    ]


def query_items(prepared: Iterable[Mapping[str, Any]]) -> list[Item]:
    """Return the answer-axis queries as scan items, one passage per candidate.

    Args:
        prepared: Queries carrying id, question, and candidates.

    Returns:
        One item per query.
    """
    return [
        (
            q["id"],
            q["question"],
            [(c["pairs"], c["truncated_body"]) for c in q["candidates"]],
        )
        for q in prepared
    ]


def refit_body(
    question: str,
    pairs: Sequence[tuple[str, str]],
    body: str,
    tokenizers: Mapping[str, Tokenizer],
    *,
    max_tokens: int = MAX_TOKENS,
    formats: Mapping[str, Renderer] | None = None,
) -> str | None:
    """Truncate a body so that it fits every tokenizer, not only the tightest.

    Each tokenizer budgets and cuts the body on its own count, and the shortest
    cut is kept. A cut that fits one tokenizer is a prefix of any longer cut, so
    the shortest fits them all.

    Args:
        question: The query text.
        pairs: The metadata pairs of the passage.
        body: The full body prose.
        tokenizers: The tokenizers to fit, by name.
        max_tokens: The model context length.
        formats: The renderers to budget against. The default is FORMATS.

    Returns:
        The truncated body, or None if any tokenizer leaves too small a budget.
    """
    cuts = []
    for name, tokenizer in tokenizers.items():
        budget, _ = compute_body_budget(
            question,
            pairs,
            {name: tokenizer},
            max_tokens=max_tokens,
            formats=formats,
        )
        cut = truncate_body_semantic(body, budget, tokenizer)
        if cut is None:
            return None
        cuts.append(cut)
    return min(cuts, key=len)
