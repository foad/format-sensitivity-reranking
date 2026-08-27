"""Tests for scripts.h1.build_nq_cache."""

from __future__ import annotations

import json

import pytest
from scripts.h1 import build_nq_cache as mod

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


class TestBuildCache:
    def _run(self, monkeypatch, examples, tmp_path, n_limit=0):
        monkeypatch.setattr(mod, "load_dataset", lambda *a, **k: iter(examples))
        out = tmp_path / "matched_train.json"
        mod.build_cache("train", out, n_limit)
        return json.loads(out.read_text())

    def test_writes_only_matching_examples(self, monkeypatch, tmp_path):
        examples = [example(ex_id="a"), example(html="<p>none</p>"), example(ex_id="b")]
        data = self._run(monkeypatch, examples, tmp_path)
        assert data["n_scanned"] == 3
        assert data["n_matched"] == 2
        assert [r["id"] for r in data["records"]] == ["a", "b"]

    def test_records_the_split_name(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, [example()], tmp_path)["split"] == "train"

    def test_stops_at_the_scan_limit(self, monkeypatch, tmp_path):
        examples = [example(ex_id=str(i)) for i in range(10)]
        data = self._run(monkeypatch, examples, tmp_path, n_limit=4)
        assert data["n_matched"] == 4

    def test_a_zero_limit_scans_everything(self, monkeypatch, tmp_path):
        examples = [example(ex_id=str(i)) for i in range(6)]
        assert self._run(monkeypatch, examples, tmp_path, n_limit=0)["n_matched"] == 6

    def test_creates_the_output_directory(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "load_dataset", lambda *a, **k: iter([example()]))
        out = tmp_path / "nested" / "dir" / "matched_train.json"
        mod.build_cache("train", out, 0)
        assert out.exists()

    def test_writes_an_empty_record_list_when_nothing_matches(
        self, monkeypatch, tmp_path
    ):
        data = self._run(monkeypatch, [example(html="<p>none</p>")], tmp_path)
        assert data["n_matched"] == 0
        assert data["records"] == []

    def test_reports_progress_on_the_thousandth_example(
        self, monkeypatch, tmp_path, capsys
    ):
        examples = [example(html="<p>none</p>") for _ in range(1000)]
        self._run(monkeypatch, examples, tmp_path)
        assert "scanned 1,000" in capsys.readouterr().out


class TestMain:
    def test_builds_each_requested_split(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "load_dataset", lambda *a, **k: iter([example()]))
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--splits", "train", "validation", "--out-dir", str(tmp_path)],
        )
        mod.main()
        assert (tmp_path / "matched_train.json").exists()
        assert (tmp_path / "matched_validation.json").exists()

    def test_skips_a_split_whose_cache_exists(self, monkeypatch, tmp_path, capsys):
        existing = tmp_path / "matched_train.json"
        existing.write_text("{}")

        def fail(*args, **kwargs):
            raise AssertionError("load_dataset must not be called")

        monkeypatch.setattr(mod, "load_dataset", fail)
        monkeypatch.setattr(
            "sys.argv", ["prog", "--splits", "train", "--out-dir", str(tmp_path)]
        )
        mod.main()
        assert "Skipping train" in capsys.readouterr().out
        assert existing.read_text() == "{}"
