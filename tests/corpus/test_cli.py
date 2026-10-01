"""Tests for fsr.corpus.cli."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from fsr.corpus import cli
from fsr.corpus.config import DEFAULT


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


class TestSplitDir:
    def test_sits_under_the_data_root(self):
        assert cli.split_dir(Path("data/nq")) == Path("data/nq") / cli.SPLIT_SUBDIR


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


class TestRecordStage:
    def write(self, tmp_path, name="parsed_train.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"records": []}')
        return path

    def test_writes_a_manifest_holding_the_stage(self, tmp_path):
        out = self.write(tmp_path)
        manifest_path = cli.record_stage(tmp_path, "parse", {out: 12}, {})
        assert manifest_path == tmp_path / cli.MANIFEST_NAME
        data = json.loads(manifest_path.read_text())
        assert [s["name"] for s in data["stages"]] == ["parse"]
        assert data["stages"][0]["outputs"][0]["records"] == 12

    def test_paths_are_relative_to_the_data_root(self, tmp_path):
        out = self.write(tmp_path / cli.SPLIT_SUBDIR, "train.json")
        cli.record_stage(tmp_path, "split", {out: 3}, {})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert (
            data["stages"][0]["outputs"][0]["path"] == f"{cli.SPLIT_SUBDIR}/train.json"
        )

    def test_a_new_manifest_starts_from_the_published_config(self, tmp_path):
        out = self.write(tmp_path)
        cli.record_stage(tmp_path, "parse", {out: None}, {})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert data["config"] == DEFAULT.as_dict()

    def test_the_stage_config_replaces_published_values(self, tmp_path):
        out = self.write(tmp_path)
        cli.record_stage(tmp_path, "split", {out: None}, {"split_seed": 7})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert data["config"]["split_seed"] == 7
        assert data["config"]["cache_k"] == DEFAULT.cache_k

    def test_a_second_stage_is_appended(self, tmp_path):
        out = self.write(tmp_path)
        cli.record_stage(tmp_path, "parse", {out: None}, {})
        cli.record_stage(tmp_path, "split", {out: None}, {})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert [s["name"] for s in data["stages"]] == ["parse", "split"]

    def test_rerunning_a_stage_replaces_it(self, tmp_path):
        out = self.write(tmp_path)
        cli.record_stage(tmp_path, "parse", {out: 1}, {})
        cli.record_stage(tmp_path, "parse", {out: 2}, {})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert len(data["stages"]) == 1
        assert data["stages"][0]["outputs"][0]["records"] == 2

    def test_records_a_file_holding_no_records(self, tmp_path):
        out = self.write(tmp_path, "meta.json")
        cli.record_stage(tmp_path, "split", {out: None}, {})
        data = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        assert data["stages"][0]["outputs"][0]["records"] is None

    def test_rejects_an_output_outside_the_data_root(self, tmp_path):
        out = self.write(tmp_path.parent, "stray.json")
        with pytest.raises(ValueError):
            cli.record_stage(tmp_path, "parse", {out: None}, {})
