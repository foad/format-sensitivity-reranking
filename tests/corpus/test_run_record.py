"""Tests for fsr.corpus.run_record."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from fsr.corpus import run_record as mod


def completed(returncode=0, stdout=""):
    return subprocess.CompletedProcess([], returncode, stdout=stdout, stderr="")


class TestNow:
    def test_returns_an_iso_string_in_utc(self):
        assert mod.now().endswith("+00:00")


class TestGitState:
    def test_reports_a_clean_tree(self, monkeypatch):
        outputs = iter([completed(stdout="abc123\n"), completed(stdout="")])
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: next(outputs))
        assert mod.git_state(Path(".")) == {"revision": "abc123", "dirty": False}

    def test_reports_a_dirty_tree(self, monkeypatch):
        outputs = iter([completed(stdout="abc123\n"), completed(stdout=" M a.py\n")])
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: next(outputs))
        assert mod.git_state(Path("."))["dirty"] is True

    def test_returns_none_outside_a_repository(self, monkeypatch):
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: completed(128))
        assert mod.git_state(Path(".")) is None

    def test_returns_none_when_git_is_absent(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(subprocess, "run", fail)
        assert mod.git_state(Path(".")) is None

    def test_returns_none_when_git_times_out(self, monkeypatch):
        def fail(*_args, **_kwargs):
            raise subprocess.TimeoutExpired("git", 10)

        monkeypatch.setattr(subprocess, "run", fail)
        assert mod.git_state(Path(".")) is None

    def test_runs_git_against_the_given_directory(self, monkeypatch):
        seen = []
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda command, **_k: seen.append(command) or completed(stdout="a\n"),
        )
        mod.git_state(Path("/tmp/repo"))
        assert seen[0][:3] == ["git", "-C", "/tmp/repo"]
        assert seen[0][3:] == ["rev-parse", "HEAD"]


class TestBegin:
    def test_records_the_command_and_revision(self, monkeypatch):
        monkeypatch.setattr(mod, "git_state", lambda _p: {"revision": "a", "dirty": 0})
        record = mod.RunRecord.begin(["prog", "--force"], Path("."))
        assert record.command == ["prog", "--force"]
        assert record.git == {"revision": "a", "dirty": 0}
        assert record.finished is None
        assert record.stages == []

    def test_copies_the_command(self, monkeypatch):
        monkeypatch.setattr(mod, "git_state", lambda _p: None)
        argv = ["prog"]
        record = mod.RunRecord.begin(argv, Path("."))
        argv.append("--force")
        assert record.command == ["prog"]

    def test_tolerates_no_repository(self, monkeypatch):
        monkeypatch.setattr(mod, "git_state", lambda _p: None)
        assert mod.RunRecord.begin(["prog"], Path(".")).git is None


class TestAdd:
    def _record(self):
        return mod.RunRecord(command=["prog"], started="t0", git=None)

    def test_adds_a_stage(self):
        record = self._record()
        record.add("parse", 0, "t1", 1.23456)
        assert [s.name for s in record.stages] == ["parse"]
        assert record.stages[0].exit_code == 0
        assert record.stages[0].started == "t1"

    def test_rounds_the_duration(self):
        record = self._record()
        record.add("parse", 0, "t1", 1.23456)
        assert record.stages[0].seconds == 1.235

    def test_keeps_the_build_order(self):
        record = self._record()
        record.add("parse", 0, "t1", 1.0)
        record.add("split", 2, "t2", 2.0)
        assert [(s.name, s.exit_code) for s in record.stages] == [
            ("parse", 0),
            ("split", 2),
        ]

    def test_complete_sets_the_finish_time(self):
        record = self._record()
        assert record.finished is None
        record.complete()
        assert record.finished is not None


class TestSave:
    def test_writes_every_field(self, tmp_path):
        record = mod.RunRecord(
            command=["prog", "--force"],
            started="t0",
            fsr_version="1.0.0",
            git={"revision": "abc", "dirty": False},
        )
        record.add("parse", 0, "t1", 1.0)
        record.complete()
        path = tmp_path / mod.RUN_NAME
        record.save(path)

        data = json.loads(path.read_text())
        assert data["command"] == ["prog", "--force"]
        assert data["fsr_version"] == "1.0.0"
        assert data["git"] == {"revision": "abc", "dirty": False}
        assert data["started"] == "t0"
        assert data["finished"] is not None
        assert data["stages"] == [
            {
                "name": "parse",
                "exit_code": 0,
                "started": "t1",
                "finished": data["stages"][0]["finished"],
                "seconds": 1.0,
            }
        ]

    def test_an_unfinished_build_saves(self, tmp_path):
        record = mod.RunRecord(command=["prog"], started="t0")
        record.add("parse", 1, "t1", 0.5)
        record.save(tmp_path / mod.RUN_NAME)
        data = json.loads((tmp_path / mod.RUN_NAME).read_text())
        assert data["finished"] is None
        assert data["stages"][0]["exit_code"] == 1

    def test_replaces_an_earlier_record(self, tmp_path):
        path = tmp_path / mod.RUN_NAME
        mod.RunRecord(command=["one"], started="t0").save(path)
        mod.RunRecord(command=["two"], started="t0").save(path)
        assert json.loads(path.read_text())["command"] == ["two"]
