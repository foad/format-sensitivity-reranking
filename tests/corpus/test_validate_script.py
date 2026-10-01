"""Tests for scripts.corpus.validate."""

from __future__ import annotations

import json

import pytest
from scripts.corpus import validate as mod

from fsr.corpus import cli
from fsr.corpus.cli import split_dir

N_CORPUS = 6
SPLIT_SIZES = {"train": 2, "dev": 1, "test": 1, "nq_val": 1}


def record(idx, flags=None):
    return {
        "id": str(idx),
        "title": f"Article {idx}",
        "quality_flags": flags or [],
    }


def build_corpus(tmp_path, cache_k=2):
    """Write a consistent corpus under tmp_path and return the data root."""
    corpus = [record(i) for i in range(N_CORPUS)]
    (tmp_path / "parsed_train.json").write_text(json.dumps({"records": corpus}))

    out_dir = split_dir(tmp_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries = []
    start = 0
    for name, size in SPLIT_SIZES.items():
        chunk = corpus[start : start + size]
        start += size
        queries += chunk
        (out_dir / f"{name}.json").write_text(json.dumps({"records": chunk}))

    (out_dir / cli.META_NAME).write_text(
        json.dumps(
            {
                "n_records_train": SPLIT_SIZES["train"],
                "n_records_dev": SPLIT_SIZES["dev"],
                "n_records_test": SPLIT_SIZES["test"],
                "nq_val_n_records": SPLIT_SIZES["nq_val"],
            }
        )
    )

    negatives = {
        q["id"]: [r["id"] for r in corpus if r["id"] != q["id"]][:cache_k]
        for q in queries
    }
    (out_dir / cli.NEGATIVES_NAME).write_text(
        json.dumps({"cache_k": cache_k, "negatives": negatives})
    )

    outputs = {out_dir / f"{name}.json": None for name in SPLIT_SIZES}
    outputs[tmp_path / "parsed_train.json"] = None
    outputs[out_dir / cli.META_NAME] = None
    outputs[out_dir / cli.NEGATIVES_NAME] = None
    cli.record_stage(tmp_path, "build", outputs, {})
    return tmp_path


class TestLoadJson:
    def test_reads_a_file(self, tmp_path):
        path = tmp_path / "a.json"
        path.write_text('{"x": 1}')
        problems = []
        assert mod.load_json(path, problems) == {"x": 1}
        assert problems == []

    def test_records_a_missing_file(self, tmp_path):
        problems = []
        assert mod.load_json(tmp_path / "a.json", problems) is None
        assert len(problems) == 1
        assert "is missing" in problems[0]


class TestCollectProblems:
    def test_a_consistent_corpus_has_no_problems(self, tmp_path):
        assert mod.collect_problems(build_corpus(tmp_path)) == []

    @pytest.mark.parametrize(
        "name",
        ["parsed_train.json", "splits/train.json", "splits/meta.json", "manifest.json"],
    )
    def test_reports_a_missing_file(self, tmp_path, name):
        data_root = build_corpus(tmp_path)
        (data_root / name).unlink()
        problems = mod.collect_problems(data_root)
        assert len(problems) == 1
        assert "is missing" in problems[0]

    def test_a_missing_file_stops_the_later_checks(self, tmp_path):
        data_root = build_corpus(tmp_path)
        (data_root / "manifest.json").unlink()
        flagged = [record(0, flags=["body_too_short"]), record(1)]
        (split_dir(data_root) / "train.json").write_text(
            json.dumps({"records": flagged})
        )
        assert mod.collect_problems(data_root) == [
            f"{data_root / 'manifest.json'} is missing"
        ]

    def test_reports_a_flagged_record(self, tmp_path):
        data_root = build_corpus(tmp_path)
        path = split_dir(data_root) / "train.json"
        data = json.loads(path.read_text())
        data["records"][0]["quality_flags"] = ["body_too_short"]
        path.write_text(json.dumps(data))
        problems = mod.collect_problems(data_root)
        assert any("quality flag" in p for p in problems)

    def test_reports_a_changed_file(self, tmp_path):
        data_root = build_corpus(tmp_path)
        path = split_dir(data_root) / cli.META_NAME
        path.write_text(json.dumps({**json.loads(path.read_text()), "extra": 1}))
        problems = mod.collect_problems(data_root)
        assert problems == ["build: splits/meta.json changed after the build"]

    def test_reports_a_split_overlap(self, tmp_path):
        data_root = build_corpus(tmp_path)
        train = json.loads((split_dir(data_root) / "train.json").read_text())
        path = split_dir(data_root) / "dev.json"
        path.write_text(json.dumps({"records": train["records"][:1]}))
        problems = mod.collect_problems(data_root)
        assert any("share 1 titles" in p for p in problems)


class TestMain:
    def test_reports_a_consistent_corpus(self, monkeypatch, tmp_path, capsys):
        data_root = build_corpus(tmp_path)
        monkeypatch.setattr("sys.argv", ["prog", "--data-root", str(data_root)])
        mod.main()
        assert "every check passed" in capsys.readouterr().out

    def test_exits_non_zero_when_a_check_fails(self, monkeypatch, tmp_path, capsys):
        data_root = build_corpus(tmp_path)
        (data_root / "manifest.json").unlink()
        monkeypatch.setattr("sys.argv", ["prog", "--data-root", str(data_root)])
        with pytest.raises(SystemExit, match="1 problems found"):
            mod.main()
        assert "is missing" in capsys.readouterr().out
