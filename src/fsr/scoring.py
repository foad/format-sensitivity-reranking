"""Cross-encoder scoring primitives.

Importing this module adds `create_position_ids_from_input_ids` to the
transformers XLM-RoBERTa module, where jina-reranker-v2's remote modeling code
expects it. In transformers 5.16.1 the helper exists only as a method on the
embeddings class, so the module attribute is absent. The patch runs once, on
import, and only when the attribute is missing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import transformers.models.xlm_roberta.modeling_xlm_roberta as _xlm_mod

MAX_TOKENS = 512

if not hasattr(  # pragma: no branch - decided once at import
    _xlm_mod, "create_position_ids_from_input_ids"
):

    def create_position_ids_from_input_ids(
        input_ids, padding_idx, past_key_values_length=0
    ):
        """Return position ids that skip padding, counting from padding_idx."""
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (
            torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
        ) * mask
        return incremental_indices.long() + padding_idx

    _xlm_mod.create_position_ids_from_input_ids = create_position_ids_from_input_ids


def extract_score_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Reduce a model's output logits to one relevance score per pair.

    Args:
        logits: The model output, of shape (B, 1), (B, 2), (B,), or (B, N).

    Returns:
        One score per pair. A single-logit head is squeezed, a binary head uses
        index 1, a one-dimensional output is used as is, and any other shape
        uses column 0.
    """
    if logits.dim() == 2 and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.dim() == 2 and logits.shape[-1] == 2:
        return logits[:, 1]
    if logits.dim() == 1:
        return logits
    return logits[:, 0]


@torch.no_grad()
def score_batch(
    model: Any,
    tokenizer: Any,
    pairs: Sequence[tuple[str, str]],
    batch_size: int,
    device: str,
    max_tokens: int = MAX_TOKENS,
) -> list[float]:
    """Score every query and passage pair.

    Args:
        model: The loaded cross-encoder, in evaluation mode.
        tokenizer: The tokenizer that matches the model.
        pairs: The query and passage pairs.
        batch_size: The number of pairs to encode at once.
        device: The device to move each batch to.
        max_tokens: The encoding length at which a pair is truncated.

    Returns:
        One score per pair, in the order given.
    """
    scores = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i : i + batch_size]
        enc = tokenizer(
            [p[0] for p in chunk],
            [p[1] for p in chunk],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(device)
        batch_scores = extract_score_from_logits(model(**enc).logits)
        scores.extend(batch_scores.float().cpu().tolist())
    return scores
