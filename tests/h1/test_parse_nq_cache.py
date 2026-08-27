"""Tests for scripts.h1.parse_nq_cache."""

from __future__ import annotations

import json

import pytest
from scripts.h1 import parse_nq_cache as mod

PARA = "A sufficiently long paragraph of real article prose goes here for tests."
# Long enough to clear the 200-character min_infobox gate.
INFOBOX = (
    '<table class="infobox">'
    "<tr><th>Born</th><td>1946</td></tr>"
    "<tr><th>Died</th><td>2011</td></tr>"
    "<tr><th>Role</th><td>Engineer</td></tr>"
    "<tr><th>Known for</th><td>Designing several notable bridges</td></tr>"
    "</table>"
)
BODY_HTML = f"<p>{PARA}</p><p>{PARA}</p><p>{PARA}</p>"


def cache_record(rec_id="1", infobox=INFOBOX, body=BODY_HTML, short_answers=None):
    rec = {
        "id": rec_id,
        "title": f"Title {rec_id}",
        "question": f"question {rec_id}",
        "infobox_html_raw": infobox,
        "post_infobox_html_raw": body,
    }
    if short_answers is not None:
        rec["short_answers"] = short_answers
    return rec


class TestParseRecord:
    def test_builds_a_parsed_record(self):
        parsed, _ = mod.parse_record(cache_record())
        assert parsed["id"] == "1"
        assert parsed["title"] == "Title 1"
        assert parsed["question"] == "question 1"
        assert parsed["pairs"][:3] == [
            ("Born", "1946"),
            ("Died", "2011"),
            ("Role", "Engineer"),
        ]
        assert parsed["body"].startswith(PARA)
        assert parsed["quality_flags"] == []

    def test_returns_the_parser_drop_counts(self):
        infobox = '<table class="infobox"><tr><th></th><td>x</td></tr></table>'
        _, stats = mod.parse_record(cache_record(infobox=infobox))
        assert stats["empty_dropped"] == 1

    def test_flags_a_record_that_fails_the_gate(self):
        parsed, _ = mod.parse_record(cache_record(infobox="<table></table>", body=""))
        assert "too_few_pairs" in parsed["quality_flags"]
        assert "body_too_short" in parsed["quality_flags"]

    def test_applies_the_body_length_target(self):
        parsed, _ = mod.parse_record(cache_record(), target_body_chars=80)
        assert len(parsed["body"]) == 80

    def test_applies_custom_gate_thresholds(self):
        parsed, _ = mod.parse_record(cache_record(), min_pairs=99)
        assert parsed["quality_flags"] == ["too_few_pairs"]


class TestParseRecords:
    def test_parses_every_record(self):
        parsed, _, _ = mod.parse_records([cache_record("1"), cache_record("2")])
        assert [r["id"] for r in parsed] == ["1", "2"]

    def test_totals_the_parser_counts(self):
        bad = '<table class="infobox"><tr><th></th><td>x</td></tr></table>'
        _, parser_totals, _ = mod.parse_records(
            [cache_record(infobox=bad), cache_record(infobox=bad)]
        )
        assert parser_totals["empty_dropped"] == 2

    def test_totals_the_quality_flags(self):
        _, _, quality_totals = mod.parse_records(
            [cache_record(infobox="<table></table>", body="")] * 3
        )
        assert quality_totals["too_few_pairs"] == 3
        assert quality_totals["body_too_short"] == 3

    def test_handles_an_empty_input(self):
        parsed, parser_totals, quality_totals = mod.parse_records([])
        assert parsed == []
        assert parser_totals == {}
        assert quality_totals == {}


class TestStrictGateAvailable:
    def test_reports_true_when_short_answers_are_present(self):
        assert mod.strict_gate_available([cache_record(short_answers=["1946"])])

    def test_reports_false_when_short_answers_are_absent(self):
        assert not mod.strict_gate_available([cache_record()])

    def test_reports_false_for_no_records(self):
        assert not mod.strict_gate_available([])

    def test_only_samples_the_leading_records(self):
        records = [cache_record() for _ in range(mod.STRICT_GATE_SAMPLE)]
        records.append(cache_record(short_answers=["1946"]))
        assert not mod.strict_gate_available(records)


class TestWriteDryRun:
    def test_writes_one_block_per_sample(self, tmp_path):
        parsed, _, _ = mod.parse_records([cache_record("1"), cache_record("2")])
        out = tmp_path / "dry.txt"
        mod.write_dry_run(out, parsed, [cache_record("1"), cache_record("2")], 2)
        text = out.read_text()
        assert "1/2" in text and "2/2" in text
        assert "'Born': '1946'" in text

    def test_marks_a_passing_record(self, tmp_path):
        parsed, _, _ = mod.parse_records([cache_record()])
        out = tmp_path / "dry.txt"
        mod.write_dry_run(out, parsed, [cache_record()], 1)
        assert "(passed)" in out.read_text()

    def test_notes_a_truncated_body(self, tmp_path):
        long_body = "".join(f"<p>{PARA}</p>" for _ in range(60))
        rec = cache_record(body=long_body)
        parsed, _, _ = mod.parse_records([rec])
        out = tmp_path / "dry.txt"
        mod.write_dry_run(out, parsed, [rec], 1)
        assert "more chars truncated in this dump" in out.read_text()

    def test_creates_the_output_directory(self, tmp_path):
        parsed, _, _ = mod.parse_records([cache_record()])
        out = tmp_path / "nested" / "dry.txt"
        mod.write_dry_run(out, parsed, [cache_record()], 1)
        assert out.exists()


class TestMain:
    def _cache(self, tmp_path, split="train", records=None):
        path = tmp_path / f"matched_{split}.json"
        if records is None:
            records = [cache_record()]
        path.write_text(json.dumps({"records": records}))
        return path

    def _argv(self, monkeypatch, tmp_path, *extra):
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--splits",
                "train",
                "--in-dir",
                str(tmp_path),
                "--out-dir",
                str(tmp_path),
                *extra,
            ],
        )

    def test_writes_a_parsed_cache(self, monkeypatch, tmp_path):
        self._cache(tmp_path)
        self._argv(monkeypatch, tmp_path)
        mod.main()
        data = json.loads((tmp_path / "parsed_train.json").read_text())
        assert data["split"] == "train"
        assert data["source_cache"] == "matched_train.json"
        assert data["n_records"] == 1
        assert data["n_passed_quality"] == 1
        assert data["records"][0]["pairs"][0] == ["Born", "1946"]

    def test_records_the_gate_settings(self, monkeypatch, tmp_path):
        self._cache(tmp_path)
        self._argv(monkeypatch, tmp_path, "--min-pairs", "2", "--min-body", "50")
        mod.main()
        data = json.loads((tmp_path / "parsed_train.json").read_text())
        assert data["min_pairs"] == 2
        assert data["min_body"] == 50

    def test_reports_whether_the_strict_gate_is_active(self, monkeypatch, tmp_path):
        self._cache(tmp_path, records=[cache_record(short_answers=["1946"])])
        self._argv(monkeypatch, tmp_path)
        mod.main()
        data = json.loads((tmp_path / "parsed_train.json").read_text())
        assert data["strict_gate_active"] is True

    def test_skips_a_split_with_no_input(self, monkeypatch, tmp_path, capsys):
        self._argv(monkeypatch, tmp_path)
        mod.main()
        assert "does not exist" in capsys.readouterr().out
        assert not (tmp_path / "parsed_train.json").exists()

    def test_a_dry_run_writes_samples_and_stops(self, monkeypatch, tmp_path):
        self._cache(tmp_path)
        dry = tmp_path / "dry.txt"
        self._argv(monkeypatch, tmp_path, "--dry-run", "1", "--dry-run-out", str(dry))
        mod.main()
        assert dry.exists()
        assert not (tmp_path / "parsed_train.json").exists()

    def test_prints_the_parser_and_quality_summaries(
        self, monkeypatch, tmp_path, capsys
    ):
        flawed = cache_record(
            infobox='<table class="infobox"><tr><th></th><td>x</td></tr></table>',
            body="",
        )
        self._cache(tmp_path, records=[flawed])
        self._argv(monkeypatch, tmp_path)
        mod.main()
        out = capsys.readouterr().out
        assert "empty_dropped: 1" in out
        assert "flagged too_few_pairs: 1" in out
        assert "flagged body_too_short: 1" in out

    @pytest.mark.parametrize("n_records", [0, 1])
    def test_handles_a_cache_of_any_size(self, monkeypatch, tmp_path, n_records):
        self._cache(tmp_path, records=[cache_record() for _ in range(n_records)])
        self._argv(monkeypatch, tmp_path)
        mod.main()
        data = json.loads((tmp_path / "parsed_train.json").read_text())
        assert data["n_records"] == n_records
