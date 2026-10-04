"""Tests for scripts.corpus.prose_eval."""

from __future__ import annotations

import json

import pytest
from scripts.corpus import prose_eval as mod

from fsr.corpus.layout import prose_eval_path, prose_path
from fsr.corpus.manifest import MANIFEST_NAME, Manifest
from fsr.corpus.prose import PROSE_NEGATIVES


def prose_file(data_root, n_articles=20):
    records = [
        {
            "id": str(a),
            "title": f"Article {a}",
            "question": f"who built bridge {a}",
            "long_answer_text": f"The bridge {a} opened in 194{a % 10} at Place{a}.",
            "long_answer_chars": 40,
            "short_answers": ["1946"],
        }
        for a in range(n_articles)
    ]
    path = prose_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"n_records": len(records), "records": records}))
    return records


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr(
        "sys.argv",
        ["prose_eval.py", "--data-root", str(data_root), "--negatives", "5", *extra],
    )
    mod.main()


def payload(data_root):
    return json.loads(prose_eval_path(data_root).read_text())


class TestRun:
    def test_writes_one_entry_per_record(self, monkeypatch, tmp_path):
        records = prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["n_records"] == len(records)

    def test_gives_each_entry_its_negatives(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        assert all(len(r["negatives"]) == 5 for r in payload(tmp_path)["records"])

    def test_records_the_negative_count(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["k_negatives"] == 5

    def test_records_the_source(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["source"] == "prose.json"

    def test_records_the_article_count(self, monkeypatch, tmp_path):
        prose_file(tmp_path, n_articles=12)
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["n_distinct_articles"] == 12

    def test_records_how_often_the_fill_ran(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        assert payload(tmp_path)["n_random_fill"] >= 0

    def test_resolves_the_passage_of_every_candidate(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        entry = payload(tmp_path)["records"][0]
        assert entry["positive"]["text"]
        assert all(n["text"] for n in entry["negatives"])

    def test_defaults_to_the_published_negative_count(self, monkeypatch, tmp_path):
        prose_file(tmp_path, n_articles=30)
        monkeypatch.setattr("sys.argv", ["prose_eval.py", "--data-root", str(tmp_path)])
        mod.main()
        assert payload(tmp_path)["k_negatives"] == PROSE_NEGATIVES == 15

    def test_caps_the_records_on_request(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path, "--limit", "8")
        assert payload(tmp_path)["n_records"] == 8

    def test_reports_the_counts(self, monkeypatch, tmp_path, capsys):
        prose_file(tmp_path, n_articles=12)
        run(monkeypatch, tmp_path)
        printed = capsys.readouterr().out
        assert "12 records over 12 articles" in printed
        assert "random fill" in printed


class TestManifest:
    def test_records_the_stage(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        stage = Manifest.load(tmp_path / MANIFEST_NAME).stage("prose_eval")
        assert stage.outputs[0].path == "prose_eval.json"

    def test_records_the_parameters(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path, "--seed", "7")
        config = Manifest.load(tmp_path / MANIFEST_NAME).config
        assert config["prose_negatives"] == 5
        assert config["prose_negatives_seed"] == 7


class TestMissingSource:
    def test_refuses_without_the_prose_records(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="run the prose stage first"):
            run(monkeypatch, tmp_path)


class TestSkipAndForce:
    def test_skips_a_finished_stage(self, monkeypatch, tmp_path, capsys):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        capsys.readouterr()
        run(monkeypatch, tmp_path)
        assert "Skipping prose_eval" in capsys.readouterr().out

    def test_force_mines_again(self, monkeypatch, tmp_path, capsys):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        capsys.readouterr()
        run(monkeypatch, tmp_path, "--force")
        assert "Skipping" not in capsys.readouterr().out


class TestDeterminism:
    def test_two_runs_agree(self, monkeypatch, tmp_path):
        prose_file(tmp_path)
        run(monkeypatch, tmp_path)
        first = payload(tmp_path)
        run(monkeypatch, tmp_path, "--force")
        assert payload(tmp_path) == first
