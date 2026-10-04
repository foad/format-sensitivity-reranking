"""Tests for scripts.h2.compare_prose."""

from __future__ import annotations

import json

import pytest
from scripts.h2 import compare_prose as mod

from fsr.comparison import NI_MARGIN
from fsr.h2_layout import BASE_ARM, comparison_path, result_path

SLUG = "minilm_l6"
ARM = "5fmt_lam0.1"
N = 40


def write_ranking(data_root, arm, ranks, ids=None, adapter=None):
    record_ids = ids or [f"r{i}" for i in range(len(ranks))]
    path = result_path(data_root, "prose", "mrr", SLUG, arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": SLUG,
                "arm": arm,
                "adapter": adapter,
                "mrr": sum(ranks) / len(ranks),
                "reciprocal_ranks": list(ranks),
                "record_ids": record_ids,
            }
        )
    )
    return path


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    write_ranking(root, BASE_ARM, [0.5] * N)
    write_ranking(root, ARM, [0.6] * N, adapter="some/adapter")
    return root


def run(monkeypatch, data_root, *extra, arm=ARM):
    monkeypatch.setattr(
        "sys.argv",
        [
            "compare_prose.py",
            "--model",
            SLUG,
            "--data-root",
            str(data_root),
            "--arm",
            arm,
            "--n-boot",
            "100",
            *extra,
        ],
    )
    mod.main()


def payload(data_root, arm=ARM):
    return json.loads(comparison_path(data_root, SLUG, f"prose_{arm}").read_text())


class TestRun:
    def test_writes_the_comparison(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert comparison_path(data_root, SLUG, f"prose_{ARM}").exists()

    def test_names_the_model_and_the_arm(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = payload(data_root)
        assert out["model"] == SLUG
        assert out["arm"] == ARM
        assert out["split"] == "prose"

    def test_reports_both_rankings(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = payload(data_root)
        assert out["baseline_mrr"] == pytest.approx(0.5)
        assert out["trained_mrr"] == pytest.approx(0.6)

    def test_reports_the_change(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert payload(data_root)["delta_mrr_mean"] == pytest.approx(0.1)

    def test_the_interval_brackets_the_change(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = payload(data_root)
        low, high = out["delta_mrr_ci"]
        assert low <= out["delta_mrr_mean"] <= high

    def test_records_the_change_of_every_record(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = payload(data_root)
        assert len(out["per_record_delta_rr"]) == N
        assert len(out["record_ids"]) == N

    def test_records_the_adapter_of_the_arm(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert payload(data_root)["adapter"] == "some/adapter"

    def test_records_the_resample_count(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert payload(data_root)["n_boot"] == 100

    def test_names_both_sources(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        out = payload(data_root)
        assert out["baseline_source"].endswith(f"{SLUG}_{BASE_ARM}.json")
        assert out["trained_source"].endswith(f"{SLUG}_{ARM}.json")

    def test_keeps_the_score_axis_comparison_apart(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert not comparison_path(data_root, SLUG, ARM).exists()


class TestVerdict:
    def test_passes_an_arm_that_keeps_its_ranking(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert payload(data_root)["ni_pass"] is True

    def test_fails_an_arm_that_loses_its_ranking(self, monkeypatch, tmp_path):
        root = tmp_path / "nq"
        write_ranking(root, BASE_ARM, [0.9] * N)
        write_ranking(root, ARM, [0.2] * N)
        run(monkeypatch, root)
        assert payload(root)["ni_pass"] is False

    def test_carries_the_published_margin(self, monkeypatch, data_root):
        run(monkeypatch, data_root)
        assert payload(data_root)["ni_margin"] == NI_MARGIN == 0.03

    def test_reports_the_verdict(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        assert "kept, margin 0.03: yes" in capsys.readouterr().out


class TestPartialOverlap:
    def test_pairs_only_the_shared_records(self, monkeypatch, tmp_path):
        root = tmp_path / "nq"
        write_ranking(root, BASE_ARM, [0.5] * N)
        write_ranking(root, ARM, [0.6] * 10, ids=[f"r{i}" for i in range(10)])
        run(monkeypatch, root)
        assert payload(root)["n_paired_records"] == 10

    def test_says_the_counts_differ(self, monkeypatch, tmp_path, capsys):
        root = tmp_path / "nq"
        write_ranking(root, BASE_ARM, [0.5] * N)
        write_ranking(root, ARM, [0.6] * 10, ids=[f"r{i}" for i in range(10)])
        run(monkeypatch, root)
        assert "10 are shared" in capsys.readouterr().out

    def test_stays_quiet_when_they_match(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root)
        assert "are shared" not in capsys.readouterr().out

    def test_refuses_two_rankings_with_nothing_in_common(self, monkeypatch, tmp_path):
        root = tmp_path / "nq"
        write_ranking(root, BASE_ARM, [0.5] * 5, ids=[f"a{i}" for i in range(5)])
        write_ranking(root, ARM, [0.6] * 5, ids=[f"b{i}" for i in range(5)])
        with pytest.raises(ValueError, match="share no record"):
            run(monkeypatch, root)


class TestMissingInputs:
    def test_refuses_without_the_baseline(self, monkeypatch, data_root):
        result_path(data_root, "prose", "mrr", SLUG, BASE_ARM).unlink()
        with pytest.raises(SystemExit, match="no prose ranking of base"):
            run(monkeypatch, data_root)

    def test_refuses_without_the_arm(self, monkeypatch, data_root):
        result_path(data_root, "prose", "mrr", SLUG, ARM).unlink()
        with pytest.raises(SystemExit, match=f"no prose ranking of {ARM}"):
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


class TestDefaults:
    def test_defaults_to_the_published_resample_count(self):
        from fsr.metrics import DEFAULT_N_BOOT

        args = mod.build_parser().parse_args(["--model", SLUG, "--arm", ARM])
        assert args.n_boot == DEFAULT_N_BOOT == 10_000

    def test_the_seed_fixes_the_interval(self, monkeypatch, tmp_path):
        import numpy as np

        rng = np.random.default_rng(0)
        base = rng.random(N).tolist()
        trained = rng.random(N).tolist()
        intervals = []
        for _ in range(2):
            root = tmp_path / f"nq{len(intervals)}"
            write_ranking(root, BASE_ARM, base)
            write_ranking(root, ARM, trained)
            run(monkeypatch, root, "--seed", "5")
            intervals.append(payload(root)["delta_mrr_ci"])
        assert intervals[0] == intervals[1]
