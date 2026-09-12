"""Tests for scripts.corpus.split."""

from __future__ import annotations

import json

import pytest
from scripts.corpus import split as mod


def parsed_file(path, n_titles=100, n_flagged=0, report_passed=True):
    """Write a parsed corpus file and return its path."""
    records = [
        {"id": str(t), "title": f"Article {t}", "quality_flags": []}
        for t in range(n_titles)
    ]
    records += [
        {"id": f"bad{i}", "title": f"Bad {i}", "quality_flags": ["too_few_pairs"]}
        for i in range(n_flagged)
    ]
    payload = {"records": records}
    if report_passed:
        payload["n_passed_quality"] = n_titles
    path.write_text(json.dumps(payload))
    return path


def corpus_dir(tmp_path, n_titles=100):
    """Write both parsed corpus files into a directory."""
    parsed_file(tmp_path / "parsed_train.json", n_titles, n_flagged=3)
    parsed_file(tmp_path / "parsed_validation.json", 12, n_flagged=1)
    return tmp_path


class TestLoadPassed:
    def test_drops_flagged_records(self, tmp_path):
        path = parsed_file(tmp_path / "parsed_train.json", 10, n_flagged=4)
        assert len(mod.load_passed(path)) == 10

    def test_rejects_a_count_mismatch(self, tmp_path):
        path = tmp_path / "parsed_train.json"
        path.write_text(json.dumps({"records": [], "n_passed_quality": 5}))
        with pytest.raises(ValueError, match="counted 0 passed records"):
            mod.load_passed(path)

    def test_tolerates_a_file_without_the_total(self, tmp_path):
        path = parsed_file(tmp_path / "p.json", 6, report_passed=False)
        assert len(mod.load_passed(path)) == 6


class TestMain:
    def _run(self, monkeypatch, tmp_path, *extra):
        in_dir = corpus_dir(tmp_path)
        out_dir = tmp_path / "splits"
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--in-dir", str(in_dir), "--out-dir", str(out_dir), *extra],
        )
        mod.main()
        return out_dir

    def test_writes_every_split_file(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        for name in ("train", "dev", "test", "nq_val", "meta"):
            assert (out / f"{name}.json").exists()

    def test_a_split_file_carries_its_name_and_seed(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        data = json.loads((out / "dev.json").read_text())
        assert data["split"] == "dev"
        assert data["seed"] == mod.DEFAULT.split_seed
        assert data["n_records"] == len(data["records"])

    def test_the_validation_split_is_kept_whole(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        assert json.loads((out / "nq_val.json").read_text())["n_records"] == 12

    def test_meta_records_the_shares_and_counts(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        meta = json.loads((out / "meta.json").read_text())
        assert meta["seed"] == mod.DEFAULT.split_seed
        assert meta["frac_train"] == mod.DEFAULT.frac_train
        assert meta["n_articles_total"] == 100
        assert meta["nq_val_n_records"] == 12

    def test_honours_a_custom_seed(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path, "--seed", "7")
        assert json.loads((out / "meta.json").read_text())["seed"] == 7

    def test_honours_custom_shares(self, monkeypatch, tmp_path):
        out = self._run(
            monkeypatch, tmp_path, "--frac-train", "0.5", "--frac-dev", "0.25"
        )
        meta = json.loads((out / "meta.json").read_text())
        assert meta["n_articles_train"] == 50
        assert meta["n_articles_dev"] == 25

    def test_creates_the_output_directory(self, monkeypatch, tmp_path):
        assert self._run(monkeypatch, tmp_path).is_dir()

    def test_excludes_flagged_records_from_the_splits(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        total = sum(
            json.loads((out / f"{n}.json").read_text())["n_records"]
            for n in ("train", "dev", "test")
        )
        assert total == 100
