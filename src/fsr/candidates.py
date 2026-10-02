"""Candidate lists for the within-query measurement."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fsr.formats import FORMATS, Renderer
from fsr.passages import (
    MAX_TOKENS,
    Tokenizer,
    compute_body_budget,
    truncate_body_semantic,
)

DEFAULT_NEGATIVES = 15
PROGRESS_EVERY = 25
OVERFLOW_PLACES = 4


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


def measure_overflow(
    prepared: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Tokenizer],
    sample_size: int,
    seed: int = 0,
    formats: Mapping[str, Renderer] | None = None,
    max_tokens: int = MAX_TOKENS,
    verbose: bool = True,
) -> dict[str, Any] | None:
    """Measure how often a rendered candidate exceeds the model context.

    Args:
        prepared: The queries with their candidate lists.
        tokenizers: The tokenizers to measure, by name.
        sample_size: How many queries to sample. 0 or fewer measures nothing.
        seed: The seed for the sample.
        formats: The renderers to measure. The default is FORMATS.
        max_tokens: The model context length.
        verbose: Whether to report progress while measuring.

    Returns:
        The statistics of the overflow measurement, or None.
    """
    if sample_size <= 0:
        return None
    renderers = FORMATS if formats is None else formats
    rng = np.random.default_rng(seed)
    n = min(sample_size, len(prepared))
    sampled = rng.choice(len(prepared), size=n, replace=False)

    counts = dict.fromkeys(tokenizers, 0)
    total = 0
    for seen, index in enumerate(sampled, start=1):
        if verbose and (seen % PROGRESS_EVERY == 0 or seen == n):
            print(f"    overflow check {seen}/{n}", flush=True)
        query = prepared[int(index)]
        for candidate in query["candidates"]:
            for render in renderers.values():
                text = render(candidate["pairs"], candidate["truncated_body"])
                for name, tokenizer in tokenizers.items():
                    encoded = tokenizer(query["question"], text)["input_ids"]
                    if len(encoded) > max_tokens:
                        counts[name] += 1
                total += 1
    return {
        "queries_sampled": n,
        "renderings_per_model": total,
        "over_max_tokens_pct": {
            name: round(100.0 * count / total, OVERFLOW_PLACES)
            for name, count in counts.items()
        },
    }
