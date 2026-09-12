"""Tests for fsr.corpus.nq."""

from __future__ import annotations

import pytest

from fsr.corpus import nq as mod

OPEN = '<table class="infobox">'
CLOSE = "</table>"
PREFIX = "<html><body>"
INFOBOX = OPEN + "<tr><th>Born</th><td>1946</td></tr>" + CLOSE
DOC = PREFIX + INFOBOX + "<p>Lead prose.</p></body></html>"
IB_START = len(PREFIX)
IB_END = IB_START + len(INFOBOX)


def example(
    html=DOC,
    long_answer=None,
    short_answers=None,
    ex_id="42",
    title="Person",
    question="when was the person born",
):
    if long_answer is None:
        long_answer = [{"start_byte": IB_START, "end_byte": IB_END}]
    annotations = {"long_answer": long_answer}
    if short_answers is not None:
        annotations["short_answers"] = short_answers
    return {
        "id": ex_id,
        "document": {"html": html, "title": title},
        "question": {"text": question},
        "annotations": annotations,
    }


class TestShortAnswerTexts:
    def test_returns_the_first_annotators_strings(self):
        ann = {"short_answers": [{"text": ["1946", "in 1946"]}, {"text": ["other"]}]}
        assert mod.short_answer_texts(ann) == ["1946", "in 1946"]

    def test_drops_empty_strings(self):
        ann = {"short_answers": [{"text": ["1946", "", None]}]}
        assert mod.short_answer_texts(ann) == ["1946"]

    @pytest.mark.parametrize(
        "ann",
        [
            {},
            {"short_answers": []},
            {"short_answers": "not a list"},
            {"short_answers": ["not a dict"]},
            {"short_answers": [{"text": "not a list"}]},
            {"short_answers": [{}]},
        ],
    )
    def test_returns_nothing_for_a_missing_or_malformed_field(self, ann):
        assert mod.short_answer_texts(ann) == []


class TestMatchExample:
    def test_builds_a_record_for_a_matching_example(self):
        record = mod.match_example(example(short_answers=[{"text": ["1946"]}]))
        assert record == {
            "id": "42",
            "title": "Person",
            "question": "when was the person born",
            "infobox_html_raw": INFOBOX,
            "post_infobox_html_raw": "<p>Lead prose.</p></body></html>",
            "short_answers": ["1946"],
        }

    def test_rejects_a_document_with_no_infobox(self):
        assert mod.match_example(example(html="<html><p>No table.</p></html>")) is None

    def test_rejects_a_long_answer_outside_the_infobox(self):
        outside = [{"start_byte": IB_END + 1, "end_byte": IB_END + 5}]
        assert mod.match_example(example(long_answer=outside)) is None

    @pytest.mark.parametrize(
        "long_answer",
        [
            [],
            "not a list",
            ["not a dict"],
            [{"end_byte": 10}],
            [{"start_byte": -1, "end_byte": 10}],
        ],
    )
    def test_rejects_a_missing_or_malformed_long_answer(self, long_answer):
        assert mod.match_example(example(long_answer=long_answer)) is None

    def test_uses_the_first_annotators_long_answer_only(self):
        answers = [
            {"start_byte": IB_START, "end_byte": IB_END},
            {"start_byte": 0, "end_byte": 1},
        ]
        assert mod.match_example(example(long_answer=answers)) is not None

    def test_an_example_without_short_answers_still_matches(self):
        record = mod.match_example(example())
        assert record is not None
        assert record["short_answers"] == []

    def test_coerces_a_non_string_id(self):
        assert mod.match_example(example(ex_id=7))["id"] == "7"

    def test_supplies_an_empty_id_when_absent(self):
        ex = example()
        del ex["id"]
        assert mod.match_example(ex)["id"] == ""

    def test_supplies_an_empty_title_when_absent(self):
        ex = example()
        del ex["document"]["title"]
        assert mod.match_example(ex)["title"] == ""

    def test_carries_no_parsed_pairs_or_body(self):
        record = mod.match_example(example())
        assert "pairs" not in record
        assert "body" not in record


class TestIterMatched:
    def _stream(self, monkeypatch, examples):
        monkeypatch.setattr(mod, "load_dataset", lambda *_a, **_k: iter(examples))

    def test_yields_only_matching_records(self, monkeypatch):
        self._stream(monkeypatch, [example(), example(html="<p>none</p>"), example()])
        assert len(list(mod.iter_matched("train", verbose=False))) == 2

    def test_counts_scanned_and_matched(self, monkeypatch):
        self._stream(monkeypatch, [example(), example(html="<p>none</p>")])
        stats = mod.ScanStats()
        list(mod.iter_matched("train", stats=stats, verbose=False))
        assert (stats.scanned, stats.matched) == (2, 1)

    def test_stops_at_the_scan_limit(self, monkeypatch):
        self._stream(monkeypatch, [example(ex_id=str(i)) for i in range(10)])
        assert len(list(mod.iter_matched("train", n_limit=4, verbose=False))) == 4

    def test_the_scan_count_includes_the_example_that_hit_the_limit(self, monkeypatch):
        self._stream(monkeypatch, [example(ex_id=str(i)) for i in range(10)])
        stats = mod.ScanStats()
        list(mod.iter_matched("train", n_limit=4, stats=stats, verbose=False))
        assert stats.scanned == 5

    def test_is_quiet_when_asked(self, monkeypatch, capsys):
        self._stream(monkeypatch, [example()])
        list(mod.iter_matched("train", verbose=False))
        assert capsys.readouterr().out == ""

    def test_reports_progress_and_completion(self, monkeypatch, capsys):
        self._stream(monkeypatch, [example(html="<p>none</p>")] * mod.PROGRESS_EVERY)
        list(mod.iter_matched("train"))
        out = capsys.readouterr().out
        assert f"scanned {mod.PROGRESS_EVERY:,}" in out
        assert "Done:" in out

    def test_passes_the_dataset_and_revision(self, monkeypatch):
        seen = {}

        def loader(name, **kwargs):
            seen["name"] = name
            seen.update(kwargs)
            return iter([])

        monkeypatch.setattr(mod, "load_dataset", loader)
        list(mod.iter_matched("train", dataset="v/d", revision="abc", verbose=False))
        assert (seen["name"], seen["revision"], seen["streaming"]) == (
            "v/d",
            "abc",
            True,
        )
