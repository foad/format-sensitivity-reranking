"""Tests for fsr.answer_location."""

from __future__ import annotations

import pytest

from fsr.answer_location import (
    CATEGORIES,
    category_counts,
    classify_record,
    classify_records,
)


def record(rec_id="1", pairs=(("Born", "1946"),), body="prose", short_answers=None):
    rec = {"id": rec_id, "pairs": list(pairs), "body": body}
    if short_answers is not None:
        rec["short_answers"] = short_answers
    return rec


class TestClassifyRecord:
    def test_answer_in_metadata_only(self):
        rec = record(pairs=[("Born", "1946")], body="prose", short_answers=["1946"])
        assert classify_record(rec) == "metadata_only"

    def test_answer_in_body_only(self):
        rec = record(pairs=[("Born", "1946")], body="born 1834", short_answers=["1834"])
        assert classify_record(rec) == "body_only"

    def test_answer_in_both(self):
        rec = record(pairs=[("Born", "1946")], body="born 1946", short_answers=["1946"])
        assert classify_record(rec) == "both"

    def test_answer_in_neither(self):
        rec = record(short_answers=["1700"])
        assert classify_record(rec) == "neither"

    def test_missing_short_answers_field(self):
        assert classify_record(record()) == "no_answer"

    def test_empty_short_answers(self):
        assert classify_record(record(short_answers=[])) == "no_answer"

    def test_none_short_answers(self):
        assert classify_record(record(short_answers=None)) == "no_answer"

    @pytest.mark.parametrize("answer", ["", "   ", "\n"])
    def test_blank_answer_strings_do_not_count(self, answer):
        assert classify_record(record(short_answers=[answer])) == "no_answer"

    def test_non_string_answers_are_ignored(self):
        assert classify_record(record(short_answers=[1946, None])) == "no_answer"

    def test_any_answer_matching_is_enough(self):
        rec = record(short_answers=["absent", "1946"])
        assert classify_record(rec) == "metadata_only"

    def test_matching_ignores_case(self):
        rec = record(pairs=[("Role", "Engineer")], short_answers=["engineer"])
        assert classify_record(rec) == "metadata_only"

    def test_metadata_keys_are_not_searched(self):
        rec = record(pairs=[("Engineer", "1946")], short_answers=["Engineer"])
        assert classify_record(rec) == "neither"

    def test_empty_body_does_not_match(self):
        rec = record(pairs=[("Born", "1946")], body="", short_answers=["1946"])
        assert classify_record(rec) == "metadata_only"

    def test_empty_metadata_does_not_match(self):
        rec = record(pairs=[], body="born 1946", short_answers=["1946"])
        assert classify_record(rec) == "body_only"

    def test_pair_values_are_coerced_to_text(self):
        rec = record(pairs=[("Born", 1946)], body="prose", short_answers=["1946"])
        assert classify_record(rec) == "metadata_only"


class TestClassifyRecords:
    def test_classifies_each_record_by_id(self):
        records = [
            record("a", short_answers=["1946"]),
            record("b", body="born 1834", short_answers=["1834"]),
        ]
        assert classify_records(records) == {"a": "metadata_only", "b": "body_only"}

    def test_records_without_the_field_are_absent(self):
        records = [record("a", short_answers=[]), record("b")]
        assert classify_records(records) == {"a": "no_answer"}

    def test_empty_input(self):
        assert classify_records([]) == {}


class TestCategoryCounts:
    def test_totals_each_category(self):
        counts = category_counts(["metadata_only", "both", "both"])
        assert counts["metadata_only"] == 1
        assert counts["both"] == 2

    def test_reports_every_category_in_order(self):
        counts = category_counts([])
        assert list(counts) == list(CATEGORIES)
        assert set(counts.values()) == {0}

    def test_rejects_an_unknown_category(self):
        with pytest.raises(KeyError):
            category_counts(["unknown"])
