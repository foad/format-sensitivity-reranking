"""Tests for scripts.corpus.build_corpus."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.corpus import build_corpus as mod

from fsr.corpus.config import DEFAULT
from fsr.corpus.run_record import RUN_NAME

DATA_ROOT = Path("data") / "nq"


def names(stages):
    return [s.name for s in stages]


def args_of(stages, name):
    return next(s.args for s in stages if s.name == name)


class TestPlan:
    def test_runs_four_stages_by_default(self):
        assert names(mod.plan(DATA_ROOT)) == ["parse", "split", "negatives", "validate"]

    def test_fetch_is_added_first(self):
        stages = mod.plan(DATA_ROOT, fetch=True)
        assert names(stages) == ["fetch", "parse", "split", "negatives", "validate"]

    def test_parse_reads_the_cache_when_fetch_runs(self):
        stages = mod.plan(DATA_ROOT, fetch=True)
        assert "--source" in args_of(stages, "parse")
        assert "cache" in args_of(stages, "parse")

    def test_parse_streams_by_default(self):
        assert "--source" not in args_of(mod.plan(DATA_ROOT), "parse")

    def test_every_stage_takes_the_data_root(self):
        for stage in mod.plan(DATA_ROOT, fetch=True):
            assert stage.args[:2] == ["--data-root", str(DATA_ROOT)]

    def test_the_limit_reaches_the_scanning_stages_only(self):
        stages = mod.plan(DATA_ROOT, limit=50, fetch=True)
        assert "--limit" in args_of(stages, "fetch")
        assert "--limit" not in args_of(stages, "parse")
        assert "--limit" not in args_of(stages, "split")

    def test_the_limit_reaches_parse_when_it_streams(self):
        stages = mod.plan(DATA_ROOT, limit=50)
        assert args_of(stages, "parse")[-2:] == ["--limit", "50"]

    def test_no_limit_argument_without_a_limit(self):
        for stage in mod.plan(DATA_ROOT):
            assert "--limit" not in stage.args

    def test_force_reaches_every_derived_stage(self):
        stages = mod.plan(DATA_ROOT, force=True, fetch=True)
        for name in ("parse", "split", "negatives"):
            assert "--force" in args_of(stages, name)

    def test_force_never_refetches_the_raw_cache(self):
        stages = mod.plan(DATA_ROOT, force=True, fetch=True)
        assert "--force" not in args_of(stages, "fetch")

    def test_validate_never_takes_force(self):
        stages = mod.plan(DATA_ROOT, force=True)
        assert args_of(stages, "validate") == ["--data-root", str(DATA_ROOT)]

    def test_cache_k_reaches_the_negatives_stage(self):
        stages = mod.plan(DATA_ROOT, cache_k=4)
        assert args_of(stages, "negatives")[2:4] == ["--cache-k", "4"]

    def test_cache_k_defaults_to_the_published_value(self):
        stages = mod.plan(DATA_ROOT)
        assert args_of(stages, "negatives")[3] == str(DEFAULT.cache_k)

    def test_each_stage_names_its_script(self):
        for stage in mod.plan(DATA_ROOT, fetch=True):
            assert (mod.SCRIPT_DIR / stage.script).exists()


class TestProseStage:
    def test_is_off_by_default(self):
        assert "prose" not in names(mod.plan(DATA_ROOT))

    def test_runs_before_validation(self):
        stages = names(mod.plan(DATA_ROOT, prose=True))
        assert stages.index("prose") == stages.index("validate") - 1

    def test_takes_the_data_root(self):
        assert "--data-root" in args_of(mod.plan(DATA_ROOT, prose=True), "prose")

    def test_takes_the_limit(self):
        args = args_of(mod.plan(DATA_ROOT, limit=50, prose=True), "prose")
        assert args[args.index("--limit") + 1] == "50"

    def test_takes_force(self):
        args = args_of(mod.plan(DATA_ROOT, force=True, prose=True), "prose")
        assert "--force" in args

    def test_names_its_script(self):
        stage = next(s for s in mod.plan(DATA_ROOT, prose=True) if s.name == "prose")
        assert stage.script == "prose.py"


class TestRequireClean:
    def test_passes_a_clean_tree(self):
        assert mod.require_clean({"revision": "abc", "dirty": False}) is None

    def test_rejects_a_dirty_tree(self):
        with pytest.raises(SystemExit, match="abc has uncommitted changes"):
            mod.require_clean({"revision": "abc", "dirty": True})

    def test_rejects_an_unknown_revision(self):
        with pytest.raises(SystemExit, match="could not be read"):
            mod.require_clean(None)


class TestRunStage:
    def test_runs_the_script_in_its_own_process(self, monkeypatch, capsys):
        seen = {}

        def fake_run(command, check):
            seen["command"] = command
            seen["check"] = check
            return subprocess.CompletedProcess(command, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        assert mod.run_stage(mod.Stage("split", "split.py", ["--force"])) == 0
        assert seen["command"][0] == sys.executable
        assert seen["command"][1] == str(mod.SCRIPT_DIR / "split.py")
        assert seen["command"][2] == "--force"
        assert seen["check"] is False
        assert "=== split" in capsys.readouterr().out

    def test_returns_the_exit_code(self, monkeypatch):
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda command, **_kwargs: subprocess.CompletedProcess(command, 3),
        )
        assert mod.run_stage(mod.Stage("parse", "parse.py")) == 3


class TestMain:
    @pytest.fixture(autouse=True)
    def _no_git(self, monkeypatch):
        monkeypatch.setattr(
            "fsr.corpus.run_record.git_state",
            lambda _p: {"revision": "a", "dirty": False},
        )

    def _argv(self, monkeypatch, tmp_path, *extra):
        monkeypatch.setattr("sys.argv", ["prog", "--data-root", str(tmp_path), *extra])

    def _codes(self, monkeypatch, codes):
        ran = []

        def fake_run_stage(stage):
            ran.append(stage.name)
            return codes.get(stage.name, 0)

        monkeypatch.setattr(mod, "run_stage", fake_run_stage)
        return ran

    def test_runs_every_stage(self, monkeypatch, tmp_path, capsys):
        ran = self._codes(monkeypatch, {})
        self._argv(monkeypatch, tmp_path)
        mod.main()
        assert ran == ["parse", "split", "negatives", "validate"]
        assert "Corpus build complete" in capsys.readouterr().out

    def test_stops_at_the_first_failure(self, monkeypatch, tmp_path):
        ran = self._codes(monkeypatch, {"split": 2})
        self._argv(monkeypatch, tmp_path)
        with pytest.raises(SystemExit, match="split failed with exit code 2"):
            mod.main()
        assert ran == ["parse", "split"]

    def test_passes_the_arguments_through(self, monkeypatch, tmp_path):
        seen = []
        monkeypatch.setattr(mod, "run_stage", lambda stage: seen.append(stage) or 0)
        self._argv(monkeypatch, tmp_path, "--fetch", "--force", "--cache-k", "5")
        mod.main()
        assert names(seen)[0] == "fetch"
        assert "--force" in args_of(seen, "parse")
        assert "5" in args_of(seen, "negatives")

    def test_writes_a_run_record(self, monkeypatch, tmp_path):
        self._codes(monkeypatch, {})
        self._argv(monkeypatch, tmp_path)
        mod.main()
        data = json.loads((tmp_path / RUN_NAME).read_text())
        assert [s["name"] for s in data["stages"]] == [
            "parse",
            "split",
            "negatives",
            "validate",
        ]
        assert data["command"] == ["prog", "--data-root", str(tmp_path)]
        assert data["git"] == {"revision": "a", "dirty": False}
        assert data["finished"] is not None
        assert all(s["exit_code"] == 0 for s in data["stages"])
        assert all(s["seconds"] >= 0 for s in data["stages"])

    def test_a_failed_build_leaves_its_record(self, monkeypatch, tmp_path):
        self._codes(monkeypatch, {"split": 2})
        self._argv(monkeypatch, tmp_path)
        with pytest.raises(SystemExit):
            mod.main()
        data = json.loads((tmp_path / RUN_NAME).read_text())
        assert [(s["name"], s["exit_code"]) for s in data["stages"]] == [
            ("parse", 0),
            ("split", 2),
        ]
        assert data["finished"] is None

    def test_require_clean_stops_before_any_stage(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "fsr.corpus.run_record.git_state",
            lambda _p: {"revision": "abc", "dirty": True},
        )
        ran = self._codes(monkeypatch, {})
        self._argv(monkeypatch, tmp_path, "--require-clean")
        with pytest.raises(SystemExit, match="uncommitted changes"):
            mod.main()
        assert ran == []
        assert not (tmp_path / RUN_NAME).exists()

    def test_require_clean_allows_a_clean_tree(self, monkeypatch, tmp_path):
        ran = self._codes(monkeypatch, {})
        self._argv(monkeypatch, tmp_path, "--require-clean")
        mod.main()
        assert ran == ["parse", "split", "negatives", "validate"]
