"""Tests for scripts.h2.compare."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h2 import compare as mod

from fsr.comparison import IN_TRAINING, OOD, SUBSET_NAMES
from fsr.corpus.layout import split_dir
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, comparison_path, result_path

SLUG = "minilm_l6"
HELD_OUT = "yaml"
ARM = f"{HELD_OUT}_lam0.1"
N = 40


def scores(offset, seed):
    rng = np.random.default_rng(seed)
    return {
        name: (
            rng.normal(0, 1.0, size=N) + (offset if name == FORMAT_NAMES[0] else 0.0)
        ).tolist()
        for name in FORMAT_NAMES
    }


def write_eval(root, arm_name, offset, seed, rr=0.5, ids=None):
    record_ids = ids or [f"r{i}" for i in range(N)]
    path = result_path(root, "test", "cross", SLUG, arm_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "n_records_kept": N,
                "scores_per_fmt": scores(offset, seed),
                "record_ids": record_ids,
                "mrr_guardrail": {
                    "per_format_reciprocal_ranks": {f: [rr] * N for f in FORMAT_NAMES},
                    "record_ids": record_ids,
                },
            }
        )
    )
    return path


def record(rid, answer, in_metadata, in_body):
    return {
        "id": rid,
        "question": f"q {rid}",
        "pairs": [["Born", answer if in_metadata else "1800"]],
        "body": f"The text says {answer}." if in_body else "The text says nothing.",
        "short_answers": [answer],
        "quality_flags": [],
    }


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    splits = split_dir(root)
    splits.mkdir(parents=True)
    records = [
        record(f"r{i}", f"answer{i}", in_metadata=True, in_body=i >= 25)
        for i in range(N)
    ]
    (splits / "test.json").write_text(json.dumps({"records": records}))
    write_eval(root, BASE_ARM, 3.0, 0)
    write_eval(root, ARM, 0.0, 1, rr=0.7)
    return root


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare.py",
            "--model",
            SLUG,
            "--data-root",
            str(data_root),
            "--held-out-format",
            HELD_OUT,
            "--lambda-inv",
            "0.1",
            "--n-boot",
            "50",
            *extra,
        ],
    )
    mod.main()


def result(data_root, arm_name=ARM):
    return json.loads(comparison_path(data_root, SLUG, arm_name).read_text())


class TestMetadataOnlyIds:
    def test_finds_the_records_whose_answer_is_only_in_the_metadata(self, data_root):
        found = mod.metadata_only_ids(data_root, "test")
        assert found == {f"r{i}" for i in range(25)}

    def test_leaves_out_a_record_whose_answer_is_also_in_the_body(self, data_root):
        assert "r30" not in mod.metadata_only_ids(data_root, "test")


class TestRun:
    def test_writes_the_comparison(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert comparison_path(data_root, SLUG, ARM).exists()

    def test_names_the_model_and_the_arm(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = result(data_root)
        assert out["model"] == SLUG
        assert out["arm"] == ARM
        assert out["split"] == "test"

    def test_records_both_sources(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = result(data_root)
        assert out["baseline_source"].endswith(f"{SLUG}_{BASE_ARM}.json")
        assert out["trained_source"].endswith(f"{SLUG}_{ARM}.json")

    def test_covers_every_subset(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert set(result(data_root)["point_estimates_max_d"]) == set(SUBSET_NAMES)

    def test_carries_the_held_out_format(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert result(data_root)["held_out_format"] == HELD_OUT

    def test_measures_the_metadata_only_subset(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert result(data_root)["metadata_only_subset"]["n_records"] == 25

    def test_skips_the_subset_on_request(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--no-metadata-subset")
        assert result(data_root)["metadata_only_subset"] is None

    def test_covers_a_rank_arm(self, monkeypatch, data_root):
        write_eval(data_root, f"{HELD_OUT}_lam0.1_r8", 0.0, 2)
        run(monkeypatch, data_root, "--rank-tag", "8")
        assert comparison_path(data_root, SLUG, f"{HELD_OUT}_lam0.1_r8").exists()


class TestPrinting:
    def test_prints_one_line_per_pair(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        printed = capsys.readouterr().out
        for f1, f2 in [(FORMAT_NAMES[0], FORMAT_NAMES[1])]:
            assert f"{f1} - {f2}" in printed

    def test_marks_each_pair_with_its_subset(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        printed = capsys.readouterr().out
        assert OOD in printed
        assert IN_TRAINING in printed

    def test_reports_the_threshold_verdict(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        assert "In-training thresholds" in capsys.readouterr().out

    def test_omits_the_relative_change_at_a_zero_baseline(
        self, monkeypatch, data_root, capsys
    ):
        flat = {f: [1.0] * N for f in FORMAT_NAMES}
        for arm_name in (BASE_ARM, ARM):
            path = result_path(data_root, "test", "cross", SLUG, arm_name)
            payload = json.loads(path.read_text())
            payload["scores_per_fmt"] = flat
            path.write_text(json.dumps(payload))
        run(monkeypatch, data_root)
        assert "relative change" not in capsys.readouterr().out

    def test_reports_the_guardrail(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        assert "Ranking guardrail" in capsys.readouterr().out

    def test_reports_the_transfer(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        assert f"Transfer to {HELD_OUT}" in capsys.readouterr().out

    def test_says_when_the_subset_is_too_small(self, monkeypatch, tmp_path, capsys):
        root = tmp_path / "nq"
        splits = split_dir(root)
        splits.mkdir(parents=True)
        records = [
            record(f"r{i}", f"answer{i}", in_metadata=True, in_body=True)
            for i in range(N)
        ]
        (splits / "test.json").write_text(json.dumps({"records": records}))
        write_eval(root, BASE_ARM, 3.0, 0)
        write_eval(root, ARM, 0.0, 1)
        run(monkeypatch, root)
        assert "Too few metadata-only records" in capsys.readouterr().out

    def test_says_when_a_guardrail_is_absent(self, monkeypatch, data_root, capsys):
        path = result_path(data_root, "test", "cross", SLUG, ARM)
        payload = json.loads(path.read_text())
        payload["mrr_guardrail"] = None
        path.write_text(json.dumps(payload))
        run(monkeypatch, data_root)
        printed = capsys.readouterr().out
        assert "no guardrail in one of the evaluations" in printed


class TestMissingInputs:
    def test_refuses_without_a_baseline(self, monkeypatch, data_root):
        result_path(data_root, "test", "cross", SLUG, BASE_ARM).unlink()
        with pytest.raises(SystemExit, match="no evaluation of base"):
            run(monkeypatch, data_root)

    def test_refuses_without_the_trained_arm(self, monkeypatch, data_root):
        result_path(data_root, "test", "cross", SLUG, ARM).unlink()
        with pytest.raises(SystemExit, match=f"no evaluation of {ARM}"):
            run(monkeypatch, data_root)


class TestSkipAndForce:
    def test_skips_a_finished_comparison(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        capsys.readouterr()
        run(monkeypatch, data_root)
        assert "Skipping" in capsys.readouterr().out

    def test_force_compares_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        capsys.readouterr()
        run(monkeypatch, data_root, "--force")
        assert "Skipping" not in capsys.readouterr().out


class TestMark:
    def test_names_a_pass(self):
        assert mod._mark(True) == "pass"

    def test_names_a_failure(self):
        assert mod._mark(False) == "fail"
