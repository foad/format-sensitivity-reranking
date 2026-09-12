"""Tests for fsr.corpus.negatives."""

from __future__ import annotations

import random

import pytest

from fsr.corpus.negatives import (
    build_index,
    build_negatives,
    mine_for_query,
    record_to_doc,
    tokenize,
)


def record(rec_id: str, title: str, question: str = "", body: str = "prose") -> dict:
    """Return a parsed record."""
    return {
        "id": rec_id,
        "title": title,
        "question": question or f"question about {title}",
        "pairs": [("Born", "1946"), ("Role", title)],
        "body": body,
    }


def corpus(n: int = 40) -> list[dict]:
    """Return records across distinct article titles."""
    return [record(str(i), f"Article {i}", body=f"body {i}") for i in range(n)]


class TestTokenize:
    def test_lowercases_and_splits_on_words(self):
        assert tokenize("Born In 1946!") == ["born", "in", "1946"]

    def test_drops_punctuation(self):
        assert tokenize("a-b, c.d") == ["a", "b", "c", "d"]

    def test_handles_empty_text(self):
        assert tokenize("") == []


class TestRecordToDoc:
    def test_joins_title_pairs_and_body(self):
        doc = record_to_doc(record("1", "Ada"))
        assert doc.startswith("Ada ")
        assert "Born 1946" in doc
        assert doc.endswith("prose")

    def test_carries_no_format_markup(self):
        doc = record_to_doc(record("1", "Ada"))
        assert ":" not in doc and "{" not in doc and "**" not in doc


class TestBuildIndex:
    def test_records_ids_and_titles_in_order(self):
        index = build_index(corpus(5))
        assert index.ids == [str(i) for i in range(5)]
        assert index.titles == [f"Article {i}" for i in range(5)]

    def test_groups_ids_by_title(self):
        records = [record("a", "Shared"), record("b", "Shared"), record("c", "Other")]
        assert build_index(records).title_to_ids["Shared"] == ["a", "b"]


class TestMineForQuery:
    def test_returns_the_requested_number(self):
        index = build_index(corpus(40))
        picked, _ = mine_for_query(index, record("q", "Query"), random.Random(0), 5)
        assert len(picked) == 5

    def test_never_returns_the_query_title(self):
        records = corpus(40)
        index = build_index(records)
        picked, _ = mine_for_query(index, records[3], random.Random(0), 5)
        assert "3" not in picked

    def test_returns_one_negative_per_title(self):
        records = [
            record(f"{i}-{j}", f"Article {i}") for i in range(40) for j in range(3)
        ]
        index = build_index(records)
        picked, _ = mine_for_query(index, record("q", "Query"), random.Random(0), 10)
        titles = [index.titles[index.ids.index(p)] for p in picked]
        assert len(set(titles)) == len(titles)

    def test_reports_when_the_fill_was_not_needed(self):
        index = build_index(corpus(40))
        _, filled = mine_for_query(index, record("q", "Query"), random.Random(0), 3)
        assert filled is False

    def test_fills_when_retrieval_returns_too_few_titles(self):
        # The top candidates are dominated by one repeated title, so the
        # retrieved set yields fewer distinct titles than requested.
        records = [record(f"dup{i}", "Repeated", body="shared text") for i in range(40)]
        records += [record(f"rare{i}", f"Rare {i}", body="unrelated") for i in range(5)]
        index = build_index(records)
        query = record("q", "Query", question="shared text")
        picked, filled = mine_for_query(index, query, random.Random(0), 4)
        assert len(picked) == 4
        assert filled is True

    def test_rejects_a_corpus_with_too_few_titles(self):
        index = build_index(corpus(3))
        with pytest.raises(ValueError, match="only 2 negatives available"):
            mine_for_query(index, corpus(3)[0], random.Random(0), 5)

    def test_is_deterministic_for_a_given_seed(self):
        index = build_index(corpus(40))
        query = record("q", "Query")
        first, _ = mine_for_query(index, query, random.Random(7), 5)
        second, _ = mine_for_query(index, query, random.Random(7), 5)
        assert first == second


class TestBuildNegatives:
    def test_mines_every_query(self):
        records = corpus(40)
        negatives, _ = build_negatives(records, records[:6], 4, verbose=False)
        assert set(negatives) == {r["id"] for r in records[:6]}
        assert all(len(v) == 4 for v in negatives.values())

    def test_counts_the_queries_that_needed_a_fill(self):
        records = [record(f"dup{i}", "Repeated", body="shared text") for i in range(40)]
        records += [record(f"rare{i}", f"Rare {i}", body="unrelated") for i in range(5)]
        queries = [record(f"q{i}", "Query", question="shared text") for i in range(3)]
        _, filled = build_negatives(records, queries, 4, verbose=False)
        assert filled == 3

    def test_no_query_is_its_own_negative(self):
        records = corpus(40)
        negatives, _ = build_negatives(records, records[:10], 5, verbose=False)
        assert all(qid not in negs for qid, negs in negatives.items())

    def test_is_deterministic_for_a_given_seed(self):
        records = corpus(40)
        first, _ = build_negatives(records, records[:8], 5, seed=3, verbose=False)
        second, _ = build_negatives(records, records[:8], 5, seed=3, verbose=False)
        assert first == second

    def test_handles_no_queries(self):
        negatives, filled = build_negatives(corpus(10), [], 3, verbose=False)
        assert negatives == {} and filled == 0

    def test_is_quiet_when_asked(self, capsys):
        records = corpus(20)
        build_negatives(records, records[:2], 3, verbose=False)
        assert capsys.readouterr().out == ""

    def test_reports_query_progress_at_the_interval(self, monkeypatch, capsys):
        import fsr.corpus.negatives as mod

        monkeypatch.setattr(mod, "PROGRESS_EVERY", 1)
        records = corpus(20)
        build_negatives(records, records[:3], 3, verbose=True)
        assert "mined 1 / 3 queries" in capsys.readouterr().out

    def test_reports_progress_when_verbose(self, capsys):
        records = corpus(30)
        build_negatives(records, records[:2], 3, verbose=True)
        assert "Indexing" in capsys.readouterr().out
