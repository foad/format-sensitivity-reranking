"""Tests for fsr.cli."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from fsr import cli
from fsr.cli import resolve_model


def parser():
    ap = argparse.ArgumentParser()
    cli.add_common_args(ap, "Cap on records read")
    return ap


class TestAddCommonArgs:
    def test_defaults(self):
        args = parser().parse_args([])
        assert args.data_root == cli.DEFAULT_DATA_ROOT
        assert args.limit == 0
        assert args.force is False

    def test_parses_each_argument(self):
        args = parser().parse_args(["--data-root", "/tmp/x", "--limit", "5", "--force"])
        assert args.data_root == Path("/tmp/x")
        assert args.limit == 5
        assert args.force is True

    def test_limit_help_is_the_caller_text(self):
        assert "Cap on records read" in parser().format_help()


class TestTake:
    def test_no_limit_returns_every_record(self):
        assert cli.take([1, 2, 3], 0) == [1, 2, 3]

    def test_limit_truncates(self):
        assert cli.take([1, 2, 3], 2) == [1, 2]

    def test_limit_above_the_length_returns_every_record(self):
        assert cli.take([1, 2], 5) == [1, 2]

    def test_reads_an_iterator(self):
        assert cli.take(iter(range(10)), 3) == [0, 1, 2]

    def test_a_limit_stops_the_stream_early(self):
        seen = []

        def stream():
            for i in range(10):
                seen.append(i)
                yield i

        assert cli.take(stream(), 2) == [0, 1]
        assert seen == [0, 1]


class TestSkipExisting:
    def test_skips_when_every_output_is_present(self, tmp_path, capsys):
        out = tmp_path / "a.json"
        out.write_text("{}")
        assert cli.skip_existing("parse", [out], force=False) is True
        assert "Skipping parse" in capsys.readouterr().out

    def test_runs_when_one_output_is_missing(self, tmp_path):
        present = tmp_path / "a.json"
        present.write_text("{}")
        assert (
            cli.skip_existing("parse", [present, tmp_path / "b.json"], False) is False
        )

    def test_force_overrides_present_outputs(self, tmp_path):
        out = tmp_path / "a.json"
        out.write_text("{}")
        assert cli.skip_existing("parse", [out], force=True) is False

    def test_a_stage_with_no_outputs_never_skips(self):
        assert cli.skip_existing("parse", [], force=False) is False

    def test_names_a_single_output(self, tmp_path, capsys):
        out = tmp_path / "a.json"
        out.write_text("{}")
        cli.skip_existing("parse", [out], force=False)
        printed = capsys.readouterr().out
        assert f"{out} is already present" in printed
        assert "--force" in printed

    def test_counts_several_outputs(self, tmp_path, capsys):
        outs = []
        for name in ("a.json", "b.json"):
            p = tmp_path / name
            p.write_text("{}")
            outs.append(p)
        cli.skip_existing("split", outs, force=False)
        printed = capsys.readouterr().out
        assert f"2 outputs in {tmp_path} are already present" in printed
        assert "--force" in printed


class TestReportWritten:
    def test_reports_the_size(self, tmp_path, capsys):
        out = tmp_path / "a.json"
        out.write_text("x" * 2_000_000)
        cli.report_written(out)
        printed = capsys.readouterr().out
        assert "2.0 MB" in printed
        assert "records" not in printed

    def test_reports_the_record_count(self, tmp_path, capsys):
        out = tmp_path / "a.json"
        out.write_text("{}")
        cli.report_written(out, 8188)
        assert "8,188 records" in capsys.readouterr().out


class TestResolveModel:
    def test_accepts_a_slug(self):
        assert resolve_model("bge_base").slug == "bge_base"

    def test_accepts_an_identifier(self):
        assert resolve_model("BAAI/bge-reranker-base").slug == "bge_base"

    def test_rejects_an_unknown_name(self):
        with pytest.raises(SystemExit, match="unknown model 'nope'"):
            resolve_model("nope")

    def test_rejects_an_unregistered_identifier(self):
        with pytest.raises(SystemExit, match="unknown model"):
            resolve_model("some/other-model")

    def test_names_the_known_slugs_when_it_refuses(self):
        with pytest.raises(SystemExit, match="minilm_l6"):
            resolve_model("nope")
