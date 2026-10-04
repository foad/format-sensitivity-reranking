"""Tests for scripts.corpus.prose."""

from __future__ import annotations

import json

import pytest
from scripts.corpus import prose as mod

from fsr.corpus.layout import prose_path
from fsr.corpus.manifest import MANIFEST_NAME, Manifest
from fsr.corpus.prose import ANSWER_IN_INFOBOX

RECORDS = [
    {
        "id": str(i),
        "title": f"Article {i}",
        "question": f"who built bridge {i}",
        "long_answer_text": f"The bridge {i} opened in 1946.",
        "long_answer_chars": 29,
        "short_answers": ["1946"],
    }
    for i in range(3)
]


@pytest.fixture
def streamed(monkeypatch):
    """Stand in for the dataset stream, counting one rejection."""

    def fake_iter(split, n_limit, min_body, body_chars, revision=None, stats=None):
        seen = {
            "split": split,
            "limit": n_limit,
            "min_body": min_body,
            "body_chars": body_chars,
            "revision": revision,
        }
        calls.append(seen)
        if stats is not None:
            stats.scanned = 10
            stats.matched = len(RECORDS)
            stats.skip(ANSWER_IN_INFOBOX)
        yield from RECORDS

    calls: list[dict] = []
    monkeypatch.setattr(mod, "iter_prose", fake_iter)
    return calls


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr("sys.argv", ["prose.py", "--data-root", str(data_root), *extra])
    mod.main()


def payload(data_root):
    return json.loads(prose_path(data_root).read_text())


class TestRun:
    @pytest.mark.usefixtures("streamed")
    def test_writes_the_records(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["n_records"] == 3

    @pytest.mark.usefixtures("streamed")
    def test_names_the_split_it_read(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["split"] == "validation"

    @pytest.mark.usefixtures("streamed")
    def test_records_what_it_scanned(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["n_scanned"] == 10

    @pytest.mark.usefixtures("streamed")
    def test_records_why_examples_were_rejected(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["skipped"][ANSWER_IN_INFOBOX] == 1

    @pytest.mark.usefixtures("streamed")
    def test_keeps_the_record_fields(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["records"][0]["long_answer_text"]

    def test_reads_the_validation_split(self, monkeypatch, tmp_path, streamed):
        run(monkeypatch, tmp_path)
        assert streamed[0]["split"] == "validation"

    def test_passes_the_gates_through(self, monkeypatch, tmp_path, streamed):
        run(monkeypatch, tmp_path, "--min-body-chars", "50", "--body-chars", "900")
        assert streamed[0]["min_body"] == 50
        assert streamed[0]["body_chars"] == 900

    def test_passes_the_cap_through(self, monkeypatch, tmp_path, streamed):
        run(monkeypatch, tmp_path, "--limit", "25")
        assert streamed[0]["limit"] == 25

    def test_pins_the_revision(self, monkeypatch, tmp_path, streamed):
        run(monkeypatch, tmp_path, "--revision", "abc123")
        assert streamed[0]["revision"] == "abc123"

    @pytest.mark.usefixtures("streamed")
    def test_reports_the_counts(self, monkeypatch, tmp_path, capsys):
        run(monkeypatch, tmp_path)
        printed = capsys.readouterr().out
        assert "kept 3 of 10 scanned" in printed
        assert ANSWER_IN_INFOBOX in printed


class TestManifest:
    def loaded(self, data_root):
        return Manifest.load(data_root / MANIFEST_NAME)

    @pytest.mark.usefixtures("streamed")
    def test_records_the_stage(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        stage = self.loaded(tmp_path).stage("prose")
        assert stage is not None
        assert stage.outputs[0].records == 3

    @pytest.mark.usefixtures("streamed")
    def test_records_the_file_it_wrote(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path)
        stage = self.loaded(tmp_path).stage("prose")
        assert stage.outputs[0].path == "prose.json"
        assert stage.outputs[0].sha256

    @pytest.mark.usefixtures("streamed")
    def test_records_the_parameters(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path, "--min-body-chars", "50")
        config = self.loaded(tmp_path).config
        assert config["min_body_chars"] == 50
        assert config["split"] == "validation"

    @pytest.mark.usefixtures("streamed")
    def test_records_the_revision(self, monkeypatch, tmp_path):
        run(monkeypatch, tmp_path, "--revision", "abc123")
        assert self.loaded(tmp_path).config["dataset_revision"] == "abc123"


class TestSkipAndForce:
    def test_skips_a_finished_stage(self, monkeypatch, tmp_path, streamed, capsys):
        run(monkeypatch, tmp_path)
        capsys.readouterr()
        run(monkeypatch, tmp_path)
        assert "Skipping prose" in capsys.readouterr().out
        assert len(streamed) == 1

    def test_force_collects_again(self, monkeypatch, tmp_path, streamed):
        run(monkeypatch, tmp_path)
        run(monkeypatch, tmp_path, "--force")
        assert len(streamed) == 2


class TestEmptyResult:
    def test_stops_when_nothing_passes(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "iter_prose", lambda *_a, **_k: iter(()))
        with pytest.raises(SystemExit, match="nothing to write"):
            run(monkeypatch, tmp_path)

    def test_writes_no_file(self, monkeypatch, tmp_path):
        monkeypatch.setattr(mod, "iter_prose", lambda *_a, **_k: iter(()))
        with pytest.raises(SystemExit):
            run(monkeypatch, tmp_path)
        assert not prose_path(tmp_path).exists()
