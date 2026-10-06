"""Tests for scripts.h2.fit_heads."""

from __future__ import annotations

import json

import numpy as np
import pytest
from scripts.h2 import fit_heads as mod

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import (
    feature_meta_path,
    feature_path,
    feature_score_path,
    frontier_path,
)
from fsr.head_probe.heads import HEAD_NAMES, LINEAR, TANH
from fsr.reporting import parse_progress

SLUG = "minilm_l6"
N, C, D = 24, 5, 4


def write_cache(root, split, seed):
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1.0, size=(N, C, len(FORMAT_NAMES), D)).astype(np.float32)
    features[:, 0, :, 0] += 3.0
    path = feature_path(root, split, SLUG)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, features)
    np.save(feature_score_path(root, split, SLUG), features.sum(axis=-1))
    feature_meta_path(root, split, SLUG).write_text(
        json.dumps(
            {
                "model": SLUG,
                "split": split,
                "formats": list(FORMAT_NAMES),
                "record_ids": [f"r{i}" for i in range(N)],
            }
        )
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    write_cache(root, "train", 0)
    write_cache(root, "dev", 1)
    return root


def run(monkeypatch, data_root, *extra):
    monkeypatch.setattr(
        "sys.argv",
        [
            "fit_heads.py",
            "--model",
            SLUG,
            "--data-root",
            str(data_root),
            "--max-steps",
            "12",
            "--warmup-steps",
            "2",
            "--batch-size",
            "8",
            "--neg-k",
            "2",
            *extra,
        ],
    )
    mod.main()


def frontier(data_root):
    return json.loads(frontier_path(data_root, SLUG).read_text())


class TestRun:
    def test_writes_one_point_per_configuration(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--heads", LINEAR, TANH, "--lambdas", "0", "1")
        points = frontier(data_root)["points"]
        assert len(points) == 2 * 2 * len(mod.DEFAULT_SEEDS)

    def test_each_point_carries_both_axes(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        point = frontier(data_root)["points"][0]
        assert "max_abs_cohen_d" in point
        assert "mean_mrr" in point

    def test_records_the_parameter_count_of_each_head(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--lambdas", "0", "--seeds", "0")
        counts = {p["head"]: p["parameters"] for p in frontier(data_root)["points"]}
        assert counts["wide_linear"] == counts["tanh"]
        assert counts[LINEAR] < counts[TANH]

    def test_keeps_the_loss_history_of_every_fit(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        assert frontier(data_root)["points"][0]["history"]

    def test_describes_the_sweep(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        payload = frontier(data_root)
        assert payload["model"] == SLUG
        assert payload["fit_split"] == "train"
        assert payload["eval_split"] == "dev"
        assert payload["n_fit_records"] == N
        assert payload["feature_dim"] == D

    def test_sweeps_every_head_by_default(self, monkeypatch, data_root):
        run(monkeypatch, data_root, "--lambdas", "0", "--seeds", "0")
        assert {p["head"] for p in frontier(data_root)["points"]} == set(HEAD_NAMES)

    def test_a_seed_changes_the_point(self, monkeypatch, data_root):
        run(
            monkeypatch,
            data_root,
            "--heads",
            TANH,
            "--lambdas",
            "1",
            "--seeds",
            "0",
            "1",
        )
        points = frontier(data_root)["points"]
        assert points[0]["max_abs_cohen_d"] != points[1]["max_abs_cohen_d"]


class TestFailures:
    def test_one_bad_point_does_not_stop_the_sweep(self, monkeypatch, data_root):
        """A sweep of sixty points must not die on one of them."""
        calls = {"n": 0}
        original = mod.train_head

        def flaky(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("one bad fit")
            return original(*args, **kwargs)

        monkeypatch.setattr(mod, "train_head", flaky)
        run(
            monkeypatch,
            data_root,
            "--heads",
            LINEAR,
            "--lambdas",
            "0",
            "--seeds",
            "0",
            "1",
        )
        payload = frontier(data_root)
        assert len(payload["failures"]) == 1
        assert len(payload["points"]) == 1

    def test_the_failure_names_the_point(self, monkeypatch, data_root):
        monkeypatch.setattr(
            mod,
            "train_head",
            lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("x")),
        )
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "1", "--seeds", "0")
        assert frontier(data_root)["failures"][0]["point"] == "linear/lam1/seed0"


class TestProgress:
    def test_counts_every_configuration(self, monkeypatch, data_root, tmp_path):
        path = tmp_path / "p.progress"
        run(
            monkeypatch,
            data_root,
            "--heads",
            LINEAR,
            "--lambdas",
            "0",
            "1",
            "--seeds",
            "0",
            "--progress-file",
            str(path),
        )
        state = parse_progress(path.read_text())
        assert state["total"] == "2"
        assert state["done"] == "2"


class TestSkipAndForce:
    def test_skips_a_present_frontier(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        assert "is present" in capsys.readouterr().out

    def test_force_sweeps_again(self, monkeypatch, data_root, capsys):
        run(monkeypatch, data_root, "--heads", LINEAR, "--lambdas", "0", "--seeds", "0")
        run(
            monkeypatch,
            data_root,
            "--heads",
            LINEAR,
            "--lambdas",
            "0",
            "--seeds",
            "0",
            "--force",
        )
        assert "is present" not in capsys.readouterr().out


class TestMissingCache:
    def test_refuses_when_the_features_are_not_cached(self, monkeypatch, tmp_path):
        with pytest.raises(SystemExit, match="no cached representations"):
            run(monkeypatch, tmp_path / "nq")
