"""Loading and preparation of the records a training run reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fsr.corpus.layout import NEGATIVES_NAME, split_dir
from fsr.corpus.splitting import load_records
from fsr.passages import Tokenizer, compute_body_budget, truncate_body_semantic
from fsr.training.batch import TrainRecord

DEFAULT_TRAIN_NEGATIVES = 7


def load_negatives(data_root: Path) -> dict[str, list[str]]:
    """Return the mined negative identifiers of every query, by query identifier.

    Args:
        data_root: The corpus directory.

    Returns:
        The negative record identifiers, in mining order.
    """
    path = split_dir(data_root) / NEGATIVES_NAME
    return json.loads(path.read_text())["negatives"]


def load_corpus_index(
    data_root: Path,
    *,
    include_nq_val: bool = False,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Return the records a negative identifier can name.

    Args:
        data_root: The corpus directory.
        include_nq_val: Whether to add the independent validation split.
        verbose: Whether to name each file as it is read.

    Returns:
        The train, dev and test records.
    """
    index = {r["id"]: r for r in load_records(data_root, "all", verbose=verbose)}
    if include_nq_val:
        for record in load_records(data_root, "nq_val", verbose=verbose):
            index[record["id"]] = record
    return index


def truncate_to_budget(body: str, budget: int, tokenizer: Tokenizer) -> str:
    """Return a body cut to a token budget, or an empty string.

    Args:
        body: The passage body.
        budget: The token budget.
        tokenizer: The tokenizer that measures the body.

    Returns:
        The truncated body or an empty string.
    """
    return truncate_body_semantic(body, budget, tokenizer) or ""


def prepare_train_records(
    records: list[dict[str, Any]],
    negatives: dict[str, list[str]],
    corpus: dict[str, dict[str, Any]],
    tokenizer: Tokenizer,
    neg_k: int = DEFAULT_TRAIN_NEGATIVES,
) -> tuple[list[TrainRecord], int]:
    """Attach a budgeted body and hard negatives to each training record.

    Args:
        records: The split records, each with question, pairs and body.
        negatives: The mined negative identifiers, by query identifier.
        corpus: The records a negative identifier names, by identifier.
        tokenizer: The tokenizer of the model under training.
        neg_k: The negatives each record keeps.

    Returns:
        The prepared records, and the dropped count.
    """
    prepared: list[TrainRecord] = []
    dropped = 0
    for r in records:
        budget, _ = compute_body_budget(r["question"], r["pairs"], {"m": tokenizer})
        body = truncate_body_semantic(r["body"], budget, tokenizer)
        if body is None:
            dropped += 1
            continue

        neg_ids = negatives.get(r["id"], [])[:neg_k]
        if len(neg_ids) < neg_k:
            dropped += 1
            continue

        prepared.append(
            TrainRecord(
                id=r["id"],
                question=r["question"],
                pairs=r["pairs"],
                truncated_body=body,
                body_budget_tokens=budget,
                neg_pairs_list=[corpus[nid]["pairs"] for nid in neg_ids],
                neg_bodies_truncated=[
                    truncate_to_budget(corpus[nid]["body"], budget, tokenizer)
                    for nid in neg_ids
                ],
            )
        )
    return prepared, dropped


def prepare_dev_records(
    records: list[dict[str, Any]],
    tokenizer: Tokenizer,
) -> list[TrainRecord]:
    """Attach a budgeted body to each development record.

    Args:
        records: The split records, each with question, pairs and body.
        tokenizer: The tokenizer of the model under training.

    Returns:
        The prepared records.
    """
    prepared: list[TrainRecord] = []
    for r in records:
        budget, _ = compute_body_budget(r["question"], r["pairs"], {"m": tokenizer})
        body = truncate_body_semantic(r["body"], budget, tokenizer)
        if body is None:
            continue
        prepared.append(
            TrainRecord(
                id=r["id"],
                question=r["question"],
                pairs=r["pairs"],
                truncated_body=body,
                body_budget_tokens=budget,
                neg_pairs_list=[],
                neg_bodies_truncated=[],
            )
        )
    return prepared
