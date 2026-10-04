from __future__ import annotations

import re

import pytest

from fsr.reporting import (
    NAME_WIDTH,
    ProgressCounter,
    eta_seconds,
    failures_block,
    format_duration,
    heading,
    model_heading,
    parse_progress,
    progress_line,
    report_elapsed,
    report_saved,
    shorten,
    stamp,
    table,
)

STAMP_RE = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


class TestStamp:
    def test_reads_as_a_date_and_a_time(self):
        assert re.fullmatch(STAMP_RE, stamp())

    def test_formats_a_given_moment(self):
        assert re.fullmatch(STAMP_RE, stamp(0))

    def test_two_moments_differ(self):
        assert stamp(0) != stamp(86_400)


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [(0, "0:00:00"), (9, "0:00:09"), (90, "0:01:30"), (3661, "1:01:01")],
    )
    def test_formats_the_duration(self, seconds, expected):
        assert format_duration(seconds) == expected


class TestEtaSeconds:
    def test_estimates_at_the_rate_so_far(self):
        assert eta_seconds(10, 110, 20.0) == 200.0

    def test_is_zero_before_the_first_unit(self):
        assert eta_seconds(0, 100, 0.0) == 0.0

    def test_is_zero_past_the_last_unit(self):
        assert eta_seconds(120, 100, 60.0) == 0.0


class TestShorten:
    def test_leaves_a_short_name_alone(self):
        assert shorten("bge-base") == "bge-base"

    def test_keeps_the_end_of_a_long_name(self):
        out = shorten("x" * 60 + "tail", width=10)
        assert out.endswith("tail")
        assert len(out) == 10

    def test_marks_a_cut_name(self):
        assert shorten("y" * 60).startswith("...")

    def test_keeps_a_name_of_exactly_the_width(self):
        assert shorten("z" * NAME_WIDTH) == "z" * NAME_WIDTH


class TestHeading:
    def test_prints_the_title_and_the_time(self, capsys):
        heading("CROSS-MODEL SUMMARY")
        out = capsys.readouterr().out
        assert "CROSS-MODEL SUMMARY" in out
        assert re.search(STAMP_RE, out)

    def test_rules_above_and_below(self, capsys):
        heading("T", width=12)
        lines = capsys.readouterr().out.strip().split("\n")
        assert lines[0] == "=" * 12
        assert lines[2] == "=" * 12


class TestModelHeading:
    def test_prints_the_name_and_the_time(self, capsys):
        model_heading("BAAI/bge-reranker-base")
        out = capsys.readouterr().out
        assert "BAAI/bge-reranker-base" in out
        assert re.search(STAMP_RE, out)

    def test_rules_above_and_below(self, capsys):
        model_heading("m", width=8)
        lines = capsys.readouterr().out.strip().split("\n")
        assert lines[0] == "-" * 8
        assert lines[2] == "-" * 8


class TestProgressLine:
    def test_reports_every_field(self):
        line = progress_line(3, 10, "with_body/yaml", 60.0)
        assert parse_progress(line) == {
            "done": "3",
            "total": "10",
            "elapsed": "60.0",
            "eta": "140.0",
            "failed": "0",
            "label": "with_body/yaml",
        }

    def test_reports_no_time_left_at_the_end(self):
        assert parse_progress(progress_line(10, 10, "a", 50.0))["eta"] == "0.0"

    def test_marks_a_failure(self):
        assert (
            parse_progress(progress_line(3, 10, "a", 1.0, failed=True))["failed"] == "1"
        )

    def test_ends_without_a_newline(self):
        assert not progress_line(1, 2, "a", 1.0).endswith("\n")


class TestProgressCounter:
    def test_counts_from_one(self, tmp_path):
        counter = ProgressCounter(4, tmp_path / "j.progress")
        assert parse_progress(counter.step("a"))["done"] == "1"
        assert parse_progress(counter.step("b"))["done"] == "2"

    def test_writes_its_opening_state(self, tmp_path):
        path = tmp_path / "j.progress"
        ProgressCounter(4, path)
        assert parse_progress(path.read_text())["done"] == "0"

    def test_replaces_the_file_at_each_unit(self, tmp_path):
        path = tmp_path / "j.progress"
        counter = ProgressCounter(4, path)
        counter.step("a")
        counter.step("b")
        assert path.read_text().count("\n") == 1
        assert parse_progress(path.read_text())["label"] == "b"

    def test_creates_the_directory(self, tmp_path):
        path = tmp_path / "runs" / "j.progress"
        ProgressCounter(2, path)
        assert path.exists()

    def test_keeps_no_file_without_a_path(self, tmp_path):
        counter = ProgressCounter(2)
        assert parse_progress(counter.step("a"))["done"] == "1"
        assert list(tmp_path.iterdir()) == []

    def test_carries_the_total(self):
        assert parse_progress(ProgressCounter(7).step("a"))["total"] == "7"

    def test_labels_each_unit(self):
        assert parse_progress(ProgressCounter(2).step("x/y"))["label"] == "x/y"

    def test_measures_from_the_given_start(self, monkeypatch):
        monkeypatch.setattr("fsr.reporting.time.time", lambda: 60.0)
        counter = ProgressCounter(2, started=0.0)
        assert parse_progress(counter.step("a"))["elapsed"] == "60.0"

    def test_reports_the_time_since_it_began(self, monkeypatch):
        monkeypatch.setattr("fsr.reporting.time.time", lambda: 90.0)
        assert ProgressCounter(2, started=30.0).elapsed == pytest.approx(60.0)

    def test_marks_a_failure_and_keeps_the_count(self, tmp_path):
        path = tmp_path / "j.progress"
        counter = ProgressCounter(4, path)
        counter.step("a")
        counter.fail()
        fields = parse_progress(path.read_text())
        assert fields["failed"] == "1"
        assert fields["done"] == "1"

    def test_spans_the_phases_of_one_job(self, tmp_path):
        path = tmp_path / "j.progress"
        counter = ProgressCounter(4, path)
        for mode in ("with_body", "metadata_only"):
            for fmt in ("yaml", "json"):
                counter.step(f"{mode}/{fmt}")
        assert counter.done == 4
        assert parse_progress(path.read_text())["done"] == "4"


class TestParseProgress:
    def test_reads_a_state_line(self):
        assert parse_progress("done=7 total=10 label=x/y")["label"] == "x/y"

    def test_tolerates_a_trailing_newline(self):
        assert parse_progress("done=7 total=10\n")["total"] == "10"

    def test_reports_nothing_for_an_empty_file(self):
        assert parse_progress("") is None

    def test_reports_nothing_for_a_line_with_no_field(self):
        assert parse_progress("  scored yaml  ") is None

    def test_skips_a_token_with_no_value(self):
        assert parse_progress("done=1 stray total=2") == {"done": "1", "total": "2"}


COLUMNS = (("model", 10), ("max |d|", 8), ("ranking", 0))


class TestTable:
    def test_prints_a_header_and_a_rule(self, capsys):
        table(COLUMNS, [])
        lines = capsys.readouterr().out.strip().split("\n")
        assert lines[0].startswith("model")
        assert set(lines[1]) == {"-"}

    def test_left_aligns_the_first_column(self, capsys):
        table(COLUMNS, [["bge", "0.383", "yaml > json"]])
        row = capsys.readouterr().out.strip().split("\n")[-1]
        assert row.startswith("bge       ")

    def test_right_aligns_a_numeric_column(self, capsys):
        table(COLUMNS, [["bge", "0.383", ""]])
        row = capsys.readouterr().out.strip().split("\n")[-1]
        assert row.endswith("0.383")

    def test_does_not_pad_a_trailing_column(self, capsys):
        table(COLUMNS, [["bge", "0.383", "yaml > json"]])
        assert capsys.readouterr().out.rstrip().endswith("yaml > json")

    def test_prints_one_line_per_row(self, capsys):
        table(COLUMNS, [["a", "1", ""], ["b", "2", ""]])
        assert len(capsys.readouterr().out.strip().split("\n")) == 4

    def test_rejects_a_row_of_the_wrong_width(self, capsys):
        with pytest.raises(ValueError, match="zip"):
            table(COLUMNS, [["a", "1"]])
        capsys.readouterr()


FAILURE = {"model": "m/a", "stage": "scoring", "error": "ValueError: bad"}


class TestFailuresBlock:
    def test_prints_nothing_when_none_failed(self, capsys):
        failures_block([])
        assert capsys.readouterr().out == ""

    def test_counts_the_failures(self, capsys):
        failures_block([FAILURE, FAILURE])
        assert "FAILURES (2)" in capsys.readouterr().out

    def test_names_the_model_the_stage_and_the_error(self, capsys):
        failures_block([FAILURE])
        out = capsys.readouterr().out
        assert "m/a" in out
        assert "[scoring]" in out
        assert "ValueError: bad" in out


class TestReportSaved:
    def test_names_the_file_and_its_size(self, tmp_path, capsys):
        path = tmp_path / "out.json"
        path.write_text("x" * 2_000_000)
        report_saved(path)
        out = capsys.readouterr().out
        assert str(path) in out
        assert "2.0 MB" in out

    def test_names_what_the_file_holds(self, tmp_path, capsys):
        path = tmp_path / "out.json"
        path.write_text("{}")
        report_saved(path, "with_body")
        assert "with_body results" in capsys.readouterr().out


class TestReportElapsed:
    def test_returns_and_prints_the_duration(self, capsys, monkeypatch):
        monkeypatch.setattr("fsr.reporting.time.time", lambda: 100.0)
        assert report_elapsed("scored", 40.0) == pytest.approx(60.0)
        assert "scored in 0:01:00" in capsys.readouterr().out


class TestResumedCounter:
    def test_starts_from_the_units_already_done(self, tmp_path):
        counter = ProgressCounter(10, tmp_path / "job.progress", done=4)
        assert counter.done == 4

    def test_writes_the_resumed_state_at_once(self, tmp_path):
        path = tmp_path / "job.progress"
        ProgressCounter(10, path, done=4)
        assert parse_progress(path.read_text())["done"] == "4"

    def test_counts_on_from_there(self, tmp_path):
        counter = ProgressCounter(10, tmp_path / "job.progress", done=4)
        counter.step("next")
        assert counter.done == 5

    def test_starts_at_zero_by_default(self, tmp_path):
        assert ProgressCounter(10, tmp_path / "job.progress").done == 0
