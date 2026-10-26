"""Scoring every candidate of every query under every format."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.scoring import score_batch
from fsr.standardisation.layout import candidates_path


def load_candidates(data_root: Path, split: str, limit: int = 0) -> list[dict]:
    """Read the candidate lists the answer axis built.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.
        limit: The record cap the lists were built under, or 0 for all.

    Returns:
        One entry per query, each holding its candidates with the gold first.

    Raises:
        SystemExit: If the lists are absent.
    """
    path = candidates_path(data_root, split, limit)
    if not path.exists():
        raise SystemExit(
            f"no candidate lists at {path}. The answer axis of the main "
            f"pipeline builds them, for example `scripts/within_query.py "
            f"--split {split} --models <slug> --mode with_body`."
        )
    return json.loads(path.read_text())["prepared"]


def query_ids(prepared: Sequence[dict]) -> list[str]:
    """Return the query of each candidate list, in order.

    Args:
        prepared: The candidate lists.

    Returns:
        The query identifiers.
    """
    return [str(query["id"]) for query in prepared]


def score_every_format(
    model: Any,
    tokenizer: Any,
    prepared: Sequence[dict],
    batch_size: int,
    device: str,
    on_format: Callable[[str], None] | None = None,
) -> np.ndarray:
    """Score every candidate of every query, once per format.

    Each candidate carries its budgeted body text, as the answer axis scores
    it.

    Args:
        model: The loaded cross-encoder, in evaluation mode.
        tokenizer: The tokenizer that matches the model.
        prepared: The candidate lists, with the gold passage first.
        batch_size: The number of pairs to score at once.
        device: The device to score on.
        on_format: Called with the format name after each one completes.

    Returns:
        The scores, shaped (queries, candidates, formats).

    Raises:
        ValueError: If the candidate lists hold nothing to score.
    """
    n_queries = len(prepared)
    n_candidates = len(prepared[0]["candidates"]) if n_queries else 0
    if not n_queries or not n_candidates:
        raise ValueError("the candidate lists hold no query to score")
    scores = np.empty((n_queries, n_candidates, len(FORMAT_NAMES)), dtype=np.float32)
    for index, name in enumerate(FORMAT_NAMES):
        render = FORMATS[name]
        pairs = [
            (query["question"], render(c["pairs"], c["truncated_body"]))
            for query in prepared
            for c in query["candidates"]
        ]
        flat = score_batch(model, tokenizer, pairs, batch_size, device)
        scores[:, :, index] = np.asarray(flat, dtype=np.float32).reshape(
            n_queries, n_candidates
        )
        if on_format is not None:
            on_format(name)
    return scores
