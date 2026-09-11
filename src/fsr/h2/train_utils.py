"""Batch construction, forward passes, and the composite training loss."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn.functional import logsigmoid

from fsr.common.rendering import FORMATS
from fsr.common.scoring import MAX_TOKENS, extract_score_from_logits


@dataclass
class TrainRecord:
    """One record with its positive passage and its hard negatives.

    Attributes:
        id: The record identifier.
        question: The query text.
        pairs: The metadata key-value pairs of the positive passage.
        truncated_body: The budgeted body of the positive passage.
        body_budget_tokens: The token budget the body was cut to.
        neg_pairs_list: The metadata pairs of each negative.
        neg_bodies_truncated: The budgeted body of each negative, in the order
            of neg_pairs_list.
    """

    id: str
    question: str
    pairs: list[tuple[str, str]]
    truncated_body: str
    body_budget_tokens: int
    neg_pairs_list: list[list[tuple[str, str]]]
    neg_bodies_truncated: list[str]


def render_positive(record: TrainRecord, format_name: str) -> str:
    """Render the positive passage of a record in one format."""
    return FORMATS[format_name](record.pairs, record.truncated_body)


def render_negative(record: TrainRecord, neg_idx: int, format_name: str) -> str:
    """Render one negative passage of a record in one format."""
    return FORMATS[format_name](
        record.neg_pairs_list[neg_idx], record.neg_bodies_truncated[neg_idx]
    )


def _var_across_formats(t: torch.Tensor) -> torch.Tensor:
    """Return the mean population variance across the format axis.

    Args:
        t: A tensor whose second dimension is the format axis.

    Returns:
        A scalar. The value is zero when there are fewer than two formats.
    """
    if t.shape[1] <= 1:
        return torch.zeros((), device=t.device, dtype=t.dtype)
    return t.var(dim=1, unbiased=False).mean()


def compute_loss(
    pos_scores: torch.Tensor,
    neg_scores: torch.Tensor,
    lambda_inv: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the composite ranking and invariance loss.

    L = L_rank + lambda_inv * L_inv

    Args:
        pos_scores: The positive scores, shape (B, F).
        neg_scores: The negative scores, shape (B, F, K).
        lambda_inv: The weight of the invariance term.

    Returns:
        The total loss, the ranking term, and the invariance term.
    """
    diff = pos_scores.unsqueeze(-1) - neg_scores
    l_rank = -logsigmoid(diff).mean()
    l_inv = _var_across_formats(pos_scores)
    return l_rank + lambda_inv * l_inv, l_rank, l_inv


def forward_batch(
    model: Any,
    tokenizer: Any,
    batch: list[TrainRecord],
    format_names: Sequence[str],
    device: str,
    max_tokens: int = MAX_TOKENS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Score every record of a batch under every training format.

    Args:
        model: The model to score with.
        tokenizer: The tokenizer that matches the model.
        batch: The records to score.
        format_names: The formats to render each passage in.
        device: The device to move each encoding to.
        max_tokens: The encoding length at which a pair is truncated.

    Returns:
        The positive scores of shape (B, F) and the negative scores of shape
        (B, F, K).
    """
    n_records = len(batch)
    n_formats = len(format_names)
    n_negatives = len(batch[0].neg_pairs_list)

    pos_scores = torch.zeros(n_records, n_formats, device=device)
    neg_scores = torch.zeros(n_records, n_formats, n_negatives, device=device)

    def encode(pairs: list[tuple[str, str]]) -> Any:
        """Tokenize query and passage pairs onto the device."""
        return tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)

    for f_idx, fmt in enumerate(format_names):
        pos_pairs = [(r.question, render_positive(r, fmt)) for r in batch]
        out = model(**encode(pos_pairs))
        pos_scores[:, f_idx] = extract_score_from_logits(out.logits)

        neg_pairs = [
            (r.question, render_negative(r, k, fmt))
            for r in batch
            for k in range(n_negatives)
        ]
        out = model(**encode(neg_pairs))
        flat = extract_score_from_logits(out.logits)
        neg_scores[:, f_idx, :] = flat.reshape(n_records, n_negatives)

    return pos_scores, neg_scores
