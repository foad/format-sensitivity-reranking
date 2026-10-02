"""Tests for fsr.candidates."""

from __future__ import annotations

import pytest
from tests.fakes import CharTokenizer, WordTokenizer

from fsr.candidates import DEFAULT_NEGATIVES, build_candidate_lists, prepare_candidate

SENTENCE = "The bridge opened in 1946 and carries the main road across the river. "
TOKENIZERS = {"word": WordTokenizer()}
# 500 words of metadata leaves under MIN_BODY_TOKENS of the 512-token context.
CROWDED_PAIRS = [["Key", "value " * 500]]


def record(rec_id, pairs=None, body=None):
    return {
        "id": str(rec_id),
        "title": f"Article {rec_id}",
        "question": f"who built bridge {rec_id}",
        "pairs": pairs
        if pairs is not None
        else [["Born", "1946"], ["Role", "Engineer"]],
        "body": body if body is not None else SENTENCE * 20,
    }


def corpus(n, start=0):
    return {str(i): record(i) for i in range(start, start + n)}


def negatives_for(query_ids, pool_ids, count=DEFAULT_NEGATIVES):
    return {q: [p for p in pool_ids if p != q][:count] for q in query_ids}


class TestPrepareCandidate:
    def test_returns_the_budgeted_candidate(self):
        out = prepare_candidate("who built it", record(1), TOKENIZERS)
        assert out["id"] == "1"
        assert out["pairs"] == record(1)["pairs"]
        assert out["tightest_tokeniser"] == "word"
        assert out["body_budget_tokens"] > 0
        assert out["truncated_body"]

    def test_truncates_the_body_to_the_budget(self):
        out = prepare_candidate("who built it", record(1), TOKENIZERS)
        assert len(out["truncated_body"].split()) <= out["body_budget_tokens"]

    def test_returns_none_when_the_body_does_not_fit(self):
        crowded = record(1, pairs=CROWDED_PAIRS)
        assert prepare_candidate("who built it", crowded, TOKENIZERS) is None

    def test_picks_the_tightest_tokenizer(self):
        out = prepare_candidate(
            "who built it",
            record(1),
            {"word": WordTokenizer(), "char": CharTokenizer()},
        )
        assert out["tightest_tokeniser"] == "char"


class TestBuildCandidateLists:
    def _build(self, n_queries=2, pool=20, n_negatives=3, **kwargs):
        by_id = corpus(pool)
        queries = [by_id[str(i)] for i in range(n_queries)]
        negs = negatives_for([q["id"] for q in queries], list(by_id), n_negatives)
        return build_candidate_lists(
            queries, negs, by_id, TOKENIZERS, n_negatives, verbose=False, **kwargs
        )

    def test_builds_one_list_per_query(self):
        prepared, stats = self._build()
        assert len(prepared) == 2
        assert stats["n_input"] == 2
        assert stats["n_kept"] == 2

    def test_each_list_holds_the_gold_and_the_negatives(self):
        prepared, _ = self._build(n_negatives=3)
        assert all(len(q["candidates"]) == 4 for q in prepared)

    def test_the_gold_comes_first(self):
        prepared, _ = self._build()
        assert all(q["candidates"][0]["id"] == q["id"] for q in prepared)

    def test_a_query_never_cites_itself_as_a_negative(self):
        prepared, _ = self._build()
        for query in prepared:
            assert query["id"] not in [c["id"] for c in query["candidates"][1:]]

    def test_carries_the_question(self):
        prepared, _ = self._build()
        assert prepared[0]["question"] == record(0)["question"]

    def test_counts_the_tightest_tokenizer_per_candidate(self):
        _, stats = self._build(n_queries=2, n_negatives=3)
        assert stats["tightest_tokeniser_counts"] == {"word": 8}

    def test_drops_a_query_with_no_negatives(self):
        by_id = corpus(20)
        queries = [by_id["0"], by_id["1"]]
        negs = {"0": [str(i) for i in range(2, 5)]}
        prepared, stats = build_candidate_lists(
            queries, negs, by_id, TOKENIZERS, 3, verbose=False
        )
        assert stats["dropped_no_bm25_negs"] == 1
        assert [q["id"] for q in prepared] == ["0"]

    def test_drops_a_query_with_an_empty_negative_list(self):
        by_id = corpus(20)
        prepared, stats = build_candidate_lists(
            [by_id["0"]], {"0": []}, by_id, TOKENIZERS, 3, verbose=False
        )
        assert stats["dropped_no_bm25_negs"] == 1
        assert prepared == []

    def test_drops_a_query_whose_gold_body_does_not_fit(self):
        by_id = corpus(20)
        crowded = {**by_id["0"], "pairs": CROWDED_PAIRS}
        prepared, stats = build_candidate_lists(
            [crowded],
            negatives_for(["0"], list(by_id), 3),
            by_id,
            TOKENIZERS,
            3,
            verbose=False,
        )
        assert stats["dropped_gold_budget"] == 1
        assert prepared == []

    def test_passes_over_a_negative_that_does_not_fit(self):
        by_id = corpus(20)
        by_id["2"] = {**by_id["2"], "pairs": CROWDED_PAIRS}
        prepared, stats = build_candidate_lists(
            [by_id["0"]],
            {"0": [str(i) for i in range(1, 10)]},
            by_id,
            TOKENIZERS,
            3,
            verbose=False,
        )
        assert stats["negatives_skipped_for_budget"] == 1
        assert [c["id"] for c in prepared[0]["candidates"]] == ["0", "1", "3", "4"]

    def test_skips_a_negative_absent_from_the_corpus(self):
        by_id = corpus(20)
        prepared, stats = build_candidate_lists(
            [by_id["0"]],
            {"0": ["999", "1", "2", "3"]},
            by_id,
            TOKENIZERS,
            3,
            verbose=False,
        )
        assert [c["id"] for c in prepared[0]["candidates"]] == ["0", "1", "2", "3"]
        assert stats["negatives_skipped_for_budget"] == 0

    def test_drops_a_query_with_too_few_eligible_negatives(self):
        by_id = corpus(20)
        prepared, stats = build_candidate_lists(
            [by_id["0"]], {"0": ["1", "2"]}, by_id, TOKENIZERS, 3, verbose=False
        )
        assert stats["dropped_too_few_eligible_negs"] == 1
        assert prepared == []

    def test_stops_at_the_requested_number_of_negatives(self):
        by_id = corpus(30)
        prepared, _ = build_candidate_lists(
            [by_id["0"]],
            {"0": [str(i) for i in range(1, 25)]},
            by_id,
            TOKENIZERS,
            5,
            verbose=False,
        )
        assert len(prepared[0]["candidates"]) == 6

    def test_defaults_to_fifteen_negatives(self):
        by_id = corpus(40)
        prepared, _ = build_candidate_lists(
            [by_id["0"]],
            {"0": [str(i) for i in range(1, 30)]},
            by_id,
            TOKENIZERS,
            verbose=False,
        )
        assert len(prepared[0]["candidates"]) == DEFAULT_NEGATIVES + 1

    def test_no_records_gives_empty_output(self):
        prepared, stats = build_candidate_lists([], {}, {}, TOKENIZERS, verbose=False)
        assert prepared == []
        assert stats["n_input"] == 0
        assert stats["tightest_tokeniser_counts"] == {}

    def test_reports_progress_when_asked(self, capsys):
        by_id = corpus(20)
        build_candidate_lists(
            [by_id["0"]], negatives_for(["0"], list(by_id), 3), by_id, TOKENIZERS, 3
        )
        assert "1/1 queries" in capsys.readouterr().out

    def test_stays_quiet_otherwise(self, capsys):
        self._build()
        assert capsys.readouterr().out == ""

    @pytest.mark.parametrize(
        "key",
        [
            "n_input",
            "n_kept",
            "dropped_gold_budget",
            "dropped_no_bm25_negs",
            "dropped_too_few_eligible_negs",
            "negatives_skipped_for_budget",
            "tightest_tokeniser_counts",
        ],
    )
    def test_reports_every_statistic(self, key):
        _, stats = self._build()
        assert key in stats
