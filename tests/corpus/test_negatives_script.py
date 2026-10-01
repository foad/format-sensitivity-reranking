"""Tests for scripts.corpus.negatives."""

from __future__ import annotations

import json

import pytest
from scripts.corpus import negatives as mod

from fsr.corpus import cli
from fsr.corpus.cli import split_dir

SPLIT_SIZES = {"train": 4, "dev": 2, "test": 2, "nq_val": 2}


def record(idx):
    return {
        "id": str(idx),
        "title": f"Article {idx}",
        "question": f"who built bridge {idx}",
        "pairs": [["Built", str(1900 + idx)]],
        "body": f"Article {idx} describes a bridge built in {1900 + idx}.",
        "quality_flags": [],
    }


def corpus_dir(tmp_path, n_records=10, n_flagged=1):
    """Write the parsed corpus and every split file, and return the data root."""
    records = [record(i) for i in range(n_records)]
    flagged = [
        {**record(900 + i), "quality_flags": ["too_few_pairs"]}
        for i in range(n_flagged)
    ]
    (tmp_path / "parsed_train.json").write_text(
        json.dumps({"records": records + flagged})
    )
    out_dir = split_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, size in SPLIT_SIZES.items():
        chunk = records[start : start + size]
        start += size
        (out_dir / f"{name}.json").write_text(json.dumps({"records": chunk}))
    return tmp_path


class TestLoadQueries:
    def test_reads_every_split_in_order(self, tmp_path):
        data_root = corpus_dir(tmp_path)
        queries = mod.load_queries(split_dir(data_root))
        assert len(queries) == sum(SPLIT_SIZES.values())
        assert queries[0]["id"] == "0"

    def test_reads_only_the_named_splits(self, tmp_path):
        data_root = corpus_dir(tmp_path)
        queries = mod.load_queries(split_dir(data_root), ("dev",))
        assert len(queries) == SPLIT_SIZES["dev"]


class TestCheckCorpusSize:
    def test_accepts_a_corpus_with_enough_titles(self):
        mod.check_corpus_size([record(i) for i in range(5)], cache_k=4)

    def test_rejects_a_corpus_with_too_few_titles(self):
        with pytest.raises(SystemExit, match="at most 2 negatives per query"):
            mod.check_corpus_size([record(i) for i in range(3)], cache_k=4)

    def test_counts_distinct_titles_not_records(self):
        duplicated = [record(0), {**record(1), "title": "Article 0"}]
        with pytest.raises(SystemExit, match="1 distinct titles"):
            mod.check_corpus_size(duplicated, cache_k=2)


class TestMain:
    def _run(self, monkeypatch, tmp_path, *extra):
        data_root = corpus_dir(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            ["prog", "--data-root", str(data_root), "--cache-k", "3", *extra],
        )
        mod.main()
        return split_dir(data_root) / cli.NEGATIVES_NAME

    def test_writes_the_negatives_cache(self, monkeypatch, tmp_path):
        data = json.loads(self._run(monkeypatch, tmp_path).read_text())
        assert data["cache_k"] == 3
        assert data["n_queries"] == sum(SPLIT_SIZES.values())
        assert all(len(v) == 3 for v in data["negatives"].values())

    def test_excludes_flagged_records_from_the_corpus(self, monkeypatch, tmp_path):
        data = json.loads(self._run(monkeypatch, tmp_path).read_text())
        assert data["n_corpus"] == 10

    def test_a_query_never_takes_a_negative_from_its_own_article(
        self, monkeypatch, tmp_path
    ):
        data = json.loads(self._run(monkeypatch, tmp_path).read_text())
        for query_id, negative_ids in data["negatives"].items():
            assert query_id not in negative_ids

    def test_honours_the_seed(self, monkeypatch, tmp_path):
        data = json.loads(self._run(monkeypatch, tmp_path, "--seed", "7").read_text())
        assert data["seed"] == 7

    def test_limits_the_queries_mined(self, monkeypatch, tmp_path):
        data = json.loads(self._run(monkeypatch, tmp_path, "--limit", "3").read_text())
        assert data["n_queries"] == 3

    def test_skips_when_the_cache_exists(self, monkeypatch, tmp_path, capsys):
        out = self._run(monkeypatch, tmp_path)
        out.write_text("{}")
        self._run(monkeypatch, tmp_path)
        assert "Skipping negatives" in capsys.readouterr().out
        assert out.read_text() == "{}"

    def test_force_rebuilds_the_cache(self, monkeypatch, tmp_path):
        out = self._run(monkeypatch, tmp_path)
        out.write_text("{}")
        self._run(monkeypatch, tmp_path, "--force")
        assert json.loads(out.read_text())["cache_k"] == 3

    def test_records_the_stage_in_the_manifest(self, monkeypatch, tmp_path):
        self._run(monkeypatch, tmp_path, "--seed", "7")
        manifest = json.loads((tmp_path / cli.MANIFEST_NAME).read_text())
        stage = manifest["stages"][0]
        assert stage["name"] == "negatives"
        assert stage["outputs"][0]["path"] == f"{cli.SPLIT_SUBDIR}/{cli.NEGATIVES_NAME}"
        assert stage["outputs"][0]["records"] == sum(SPLIT_SIZES.values())
        assert manifest["config"]["negatives_seed"] == 7
        assert manifest["config"]["cache_k"] == 3

    def test_rejects_a_cache_k_the_corpus_cannot_meet(self, monkeypatch, tmp_path):
        data_root = corpus_dir(tmp_path)
        monkeypatch.setattr(
            "sys.argv", ["prog", "--data-root", str(data_root), "--cache-k", "20"]
        )
        with pytest.raises(SystemExit, match="short of the 20 requested"):
            mod.main()

    def test_reports_the_queries_needing_a_fill(self, monkeypatch, tmp_path, capsys):
        self._run(monkeypatch, tmp_path)
        assert "random fill" in capsys.readouterr().out
