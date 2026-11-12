"""Tests for fsr.overflow."""

from __future__ import annotations

from fsr.overflow import (
    query_items,
    record_items,
    refit_body,
    scan,
)
from tests.fakes import CharTokenizer, WordTokenizer

ONE_FORMAT = {"plain": lambda pairs, body: " ".join(v for _, v in pairs) + " " + body}


def item(item_id, question, *bodies):
    return (item_id, question, [([("k", "v")], body) for body in bodies])


class TestScan:
    def test_names_an_item_over_the_context(self):
        out = scan(
            [item("a", "q", "x" * 20)],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["ids"] == ["a"]

    def test_leaves_out_an_item_within_the_context(self):
        out = scan(
            [item("a", "q", "x")],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["ids"] == []

    def test_a_rendering_at_the_limit_does_not_overflow(self):
        out = scan(
            [item("a", "q", "xy")],
            {"char": CharTokenizer()},
            max_tokens=len("q v xy"),
            formats=ONE_FORMAT,
        )
        assert out["char"]["ids"] == []

    def test_judges_each_tokenizer_separately(self):
        out = scan(
            [item("a", "q", "x" * 20)],
            {"char": CharTokenizer(), "word": WordTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["ids"] == ["a"]
        assert out["word"]["ids"] == []

    def test_one_long_candidate_marks_the_whole_query(self):
        out = scan(
            [item("a", "q", "x", "x" * 20, "x")],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["ids"] == ["a"]
        assert out["char"]["renderings_over"] == 1
        assert out["char"]["renderings"] == 3

    def test_counts_every_format(self):
        two = {"a": ONE_FORMAT["plain"], "b": ONE_FORMAT["plain"]}
        out = scan(
            [item("a", "q", "x" * 20)],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=two,
        )
        assert out["char"]["renderings_over"] == 2
        assert out["char"]["renderings"] == 2

    def test_reports_the_excess(self):
        out = scan(
            [item("a", "q", "x" * 8), item("b", "q", "x" * 10)],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["excess_median"] == 3.0
        assert out["char"]["excess_max"] == 4

    def test_reports_no_excess_when_nothing_overflows(self):
        out = scan(
            [item("a", "q", "x")],
            {"char": CharTokenizer()},
            max_tokens=10,
            formats=ONE_FORMAT,
        )
        assert out["char"]["excess_median"] == 0.0
        assert out["char"]["excess_max"] == 0


class TestItems:
    def test_a_record_is_one_passage(self):
        record = {
            "id": "r",
            "question": "q",
            "pairs": [("k", "v")],
            "truncated_body": "b",
        }
        assert record_items([record]) == [("r", "q", [([("k", "v")], "b")])]

    def test_a_query_carries_every_candidate(self):
        query = {
            "id": "q1",
            "question": "q",
            "candidates": [
                {"pairs": [("k", "v")], "truncated_body": "gold"},
                {"pairs": [("k", "w")], "truncated_body": "neg"},
            ],
        }
        _, _, passages = query_items([query])[0]
        assert [body for _, body in passages] == ["gold", "neg"]


class TestRefitBody:
    def test_cuts_to_the_most_demanding_tokenizer(self):
        body = "aa bb cc dd ee ff gg hh ii jj kk ll mm nn oo pp qq rr ss tt"
        refit = refit_body(
            "q",
            [("k", "v")],
            body,
            {"word": WordTokenizer(), "char": CharTokenizer()},
            max_tokens=60,
            formats=ONE_FORMAT,
        )
        assert body.startswith(refit)
        assert len(refit) < len(body)

    def test_the_cut_fits_every_tokenizer(self):
        body = " ".join(["word"] * 40)
        tokenizers = {"word": WordTokenizer(), "char": CharTokenizer()}
        refit = refit_body(
            "q", [("k", "v")], body, tokenizers, max_tokens=60, formats=ONE_FORMAT
        )
        out = scan(
            [("a", "q", [([("k", "v")], refit)])],
            tokenizers,
            max_tokens=60,
            formats=ONE_FORMAT,
        )
        assert all(not v["ids"] for v in out.values())

    def test_a_short_body_is_unchanged(self):
        refit = refit_body(
            "q",
            [("k", "v")],
            "short",
            {"char": CharTokenizer()},
            max_tokens=60,
            formats=ONE_FORMAT,
        )
        assert refit == "short"

    def test_too_small_a_budget_gives_none(self):
        refit = refit_body(
            "q",
            [("k", "v")],
            "x" * 100,
            {"char": CharTokenizer()},
            max_tokens=20,
            formats=ONE_FORMAT,
        )
        assert refit is None
