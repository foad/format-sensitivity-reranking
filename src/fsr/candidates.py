"""Candidate lists for the within-query measurement."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

from fsr.passages import Tokenizer, compute_body_budget, truncate_body_semantic

DEFAULT_NEGATIVES = 15
PROGRESS_EVERY = 25


def prepare_candidate(
    question: str,
    record: Mapping[str, Any],
    tokenizers: Mapping[str, Tokenizer],
) -> dict[str, Any] | None:
    """Budget and truncate one candidate against a query.

    Args:
        question: The query text.
        record: The candidate record, with its pairs and body.
        tokenizers: The tokenizers to budget across, by name.

    Returns:
        The candidate with its truncated body, or None.
    """
    budget, tightest = compute_body_budget(question, record["pairs"], tokenizers)
    body = truncate_body_semantic(record["body"], budget, tokenizers[tightest])
    if body is None:
        return None
    return {
        "id": record["id"],
        "pairs": record["pairs"],
        "truncated_body": body,
        "body_budget_tokens": budget,
        "tightest_tokeniser": tightest,
    }


def build_candidate_lists(
    records: Sequence[Mapping[str, Any]],
    negatives: Mapping[str, Sequence[str]],
    by_id: Mapping[str, Mapping[str, Any]],
    tokenizers: Mapping[str, Tokenizer],
    n_negatives: int = DEFAULT_NEGATIVES,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one candidate list per query, with the gold passage first.

    Args:
        records: The query records.
        negatives: The mined negative ids of each query, by query id.
        by_id: Every record that may be cited as a negative, by id.
        tokenizers: The tokenizers to budget across, by name.
        n_negatives: How many negatives each list holds.
        verbose: Whether to report progress while building.

    Returns:
        The prepared queries, and the counts of what was kept and dropped.
    """
    prepared: list[dict[str, Any]] = []
    dropped_gold_budget = 0
    dropped_no_negs = 0
    dropped_thin_negs = 0
    skipped_negs = 0
    tightest_counts: dict[str, int] = {}

    def count(candidate: Mapping[str, Any]) -> None:
        name = candidate["tightest_tokeniser"]
        tightest_counts[name] = tightest_counts.get(name, 0) + 1

    started = time.time()
    for n_seen, record in enumerate(records, start=1):
        if verbose and (n_seen % PROGRESS_EVERY == 0 or n_seen == len(records)):
            rate = n_seen / max(time.time() - started, 1e-9)
            eta = (len(records) - n_seen) / max(rate, 1e-9)
            print(
                f"    {n_seen}/{len(records)} queries  "
                f"({rate:.1f}/s, eta {eta / 60:.1f} min)",
                flush=True,
            )

        neg_ids = negatives.get(record["id"])
        if not neg_ids:
            dropped_no_negs += 1
            continue

        gold = prepare_candidate(record["question"], record, tokenizers)
        if gold is None:
            dropped_gold_budget += 1
            continue
        count(gold)

        candidates = [gold]
        for neg_id in neg_ids:
            if len(candidates) > n_negatives:
                break
            negative = by_id.get(neg_id)
            if negative is None:
                continue
            prepared_negative = prepare_candidate(
                record["question"], negative, tokenizers
            )
            if prepared_negative is None:
                skipped_negs += 1
                continue
            count(prepared_negative)
            candidates.append(prepared_negative)

        if len(candidates) < n_negatives + 1:
            dropped_thin_negs += 1
            continue

        prepared.append(
            {
                "id": record["id"],
                "question": record["question"],
                "candidates": candidates,
            }
        )

    stats = {
        "n_input": len(records),
        "n_kept": len(prepared),
        "dropped_gold_budget": dropped_gold_budget,
        "dropped_no_bm25_negs": dropped_no_negs,
        "dropped_too_few_eligible_negs": dropped_thin_negs,
        "negatives_skipped_for_budget": skipped_negs,
        "tightest_tokeniser_counts": tightest_counts,
    }
    return prepared, stats
