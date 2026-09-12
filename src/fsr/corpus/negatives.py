"""BM25 hard-negative mining over the training corpus."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from fsr.corpus.config import DEFAULT

TOKEN_RE = re.compile(r"\w+")
CANDIDATE_MULTIPLIER = 4
PROGRESS_EVERY = 1000


@dataclass
class NegativeIndex:
    """A BM25 index over the corpus, with the lookups mining needs.

    Attributes:
        bm25: The index.
        ids: The record id at each corpus position.
        titles: The article title at each corpus position.
        title_to_ids: The record ids of each article title.
    """

    bm25: BM25Okapi
    ids: list[str]
    titles: list[str]
    title_to_ids: dict[str, list[str]]


def tokenize(text: str) -> list[str]:
    """Split text into lowercase word tokens."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def record_to_doc(record: dict) -> str:
    """Return the indexed text of a record.

    The text is the title, the metadata pairs, and the body, with no format
    markup.

    Args:
        record: A parsed record.

    Returns:
        The text to index.
    """
    kv = " ".join(f"{k} {v}" for k, v in record["pairs"])
    return f"{record['title']} {kv} {record['body']}"


def build_index(corpus_records: list[dict]) -> NegativeIndex:
    """Index the corpus for retrieval.

    Args:
        corpus_records: The records to index.

    Returns:
        The index and its lookups.
    """
    ids = [r["id"] for r in corpus_records]
    titles = [r["title"] for r in corpus_records]
    title_to_ids: dict[str, list[str]] = {}
    for rid, title in zip(ids, titles, strict=True):
        title_to_ids.setdefault(title, []).append(rid)
    bm25 = BM25Okapi([tokenize(record_to_doc(r)) for r in corpus_records])
    return NegativeIndex(bm25, ids, titles, title_to_ids)


def mine_for_query(
    index: NegativeIndex,
    query: dict,
    rng: random.Random,
    cache_k: int = DEFAULT.cache_k,
) -> tuple[list[str], bool]:
    """Return the hard negatives of one query.

    A candidate sharing the query's title is skipped, and each title
    contributes at most one negative. When retrieval yields too few, the
    remainder is filled from titles picked at random.

    Args:
        index: The corpus index.
        query: The query record.
        rng: The source of randomness for the fill.
        cache_k: How many negatives to return.

    Returns:
        The negative record ids, and whether the fill was used.

    Raises:
        ValueError: If the corpus holds too few distinct titles.
    """
    scores = index.bm25.get_scores(tokenize(query["question"]))
    top_idxs = scores.argsort()[::-1][: cache_k * CANDIDATE_MULTIPLIER]
    gold_title = query["title"]

    picked_ids: list[str] = []
    picked_titles: set[str] = set()
    for idx in top_idxs:
        title = index.titles[idx]
        if title == gold_title or title in picked_titles:
            continue
        picked_ids.append(index.ids[idx])
        picked_titles.add(title)
        if len(picked_ids) == cache_k:
            break

    filled = len(picked_ids) < cache_k
    if filled:
        eligible = [
            t for t in index.title_to_ids if t != gold_title and t not in picked_titles
        ]
        rng.shuffle(eligible)
        for title in eligible:
            picked_ids.append(index.title_to_ids[title][0])
            picked_titles.add(title)
            if len(picked_ids) == cache_k:
                break

    if len(picked_ids) != cache_k:
        raise ValueError(
            f"only {len(picked_ids)} negatives available for {query['id']}, "
            f"need {cache_k}"
        )
    return picked_ids, filled


def build_negatives(
    corpus_records: list[dict],
    queries: list[dict],
    cache_k: int = DEFAULT.cache_k,
    seed: int = DEFAULT.negatives_seed,
    verbose: bool = True,
) -> tuple[dict[str, list[str]], int]:
    """Mine hard negatives for every query.

    Args:
        corpus_records: The records to retrieve from.
        queries: The records to mine negatives for.
        cache_k: How many negatives to cache per query.
        seed: The seed for the random fill.
        verbose: Whether to report progress.

    Returns:
        The negatives by query id, and how many queries needed the fill.
    """
    if verbose:
        print(f"Indexing {len(corpus_records):,} corpus records...")
    index = build_index(corpus_records)

    rng = random.Random(seed)
    negatives: dict[str, list[str]] = {}
    filled_count = 0
    for i, query in enumerate(queries):
        if verbose and i and i % PROGRESS_EVERY == 0:
            print(f"  mined {i:,} / {len(queries):,} queries")
        picked, filled = mine_for_query(index, query, rng, cache_k)
        negatives[query["id"]] = picked
        filled_count += filled
    return negatives, filled_count
