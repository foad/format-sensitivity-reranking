"""Tests for fsr.representation.separation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from fsr.representation import separation as mod


def write_separation(root, slug, raw, whitened):
    path = mod.pairs_path(root, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "formats": ["a", "b", "c"],
                "adapters": {"untrained": {"raw": raw, "whitened": whitened}},
            }
        )
    )


@pytest.fixture
def root(tmp_path, monkeypatch):
    raw = [[0, 1.0, 2.0], [1.0, 0, 3.0], [2.0, 3.0, 0]]
    whitened = [[0, 3.0, 2.0], [3.0, 0, 1.0], [2.0, 1.0, 0]]
    write_separation(tmp_path, "minilm_l6", raw, whitened)
    h1 = pd.DataFrame(
        {
            "model": ["MiniLM-L6"] * 3,
            "left": ["a", "a", "b"],
            "right": ["b", "c", "c"],
            "abs_d": [0.1, 0.2, 0.3],
        }
    )
    monkeypatch.setattr(mod.h1_results, "load_axis", lambda *_a, **_k: {})
    monkeypatch.setattr(mod.h1_results, "pairwise_table", lambda _e: h1)
    return tmp_path


class TestPairFrame:
    def test_lines_up_each_pair_with_h1(self, root):
        got = mod.pair_frame(root, ["minilm_l6"])
        row = got[got["pair"] == "b - c"].iloc[0]
        assert row["raw"] == 3.0
        assert row["whitened"] == 1.0
        assert row["h1_abs_d"] == 0.3


class TestFullSpaceSeparation:
    def test_averages_the_upper_triangle(self, root):
        got = mod.full_space_separation(root, ["minilm_l6"])
        assert got["MiniLM-L6"] == pytest.approx(2.0)


class TestWithinModelRanks:
    def test_ranks_each_measure_within_the_model(self, root):
        got = mod.within_model_ranks(mod.pair_frame(root, ["minilm_l6"]))
        assert got["h1_abs_d"].tolist() == [1.0, 2.0, 3.0]
        assert got["raw"].tolist() == [1.0, 2.0, 3.0]
        assert got["whitened"].tolist() == [3.0, 2.0, 1.0]


class TestRankFrame:
    def test_reports_each_model_and_the_pool(self, root):
        got = mod.rank_frame(mod.pair_frame(root, ["minilm_l6"]))
        assert list(got.index) == ["MiniLM-L6", mod.POOLED]
        assert got.loc["MiniLM-L6", "raw_rho"] == pytest.approx(1.0)
        assert got.loc["MiniLM-L6", "whitened_rho"] == pytest.approx(-1.0)


class TestLoadRobustness:
    def test_refuses_missing_results(self, tmp_path):
        with pytest.raises(SystemExit, match="no robustness results"):
            mod.load_robustness(tmp_path)


class TestMitigationFrame:
    def test_joins_each_seeds_score_change_and_separation_change(
        self, tmp_path, monkeypatch
    ):
        path = mod.pairs_path(tmp_path, "minilm_l6")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "seeds": [0, 1],
                    "summary": {
                        mod.WINNER_OVER_CONTROL: {
                            "whitened": {"training": {"per_seed": [1.1, 0.95]}}
                        }
                    },
                }
            )
        )
        scores = pd.DataFrame({"seed 0": [-0.2], "seed 1": [-0.1]}, index=["MiniLM-L6"])
        monkeypatch.setattr(
            mod.h2_results, "seed_score_table", lambda *_a, **_k: scores
        )
        got = mod.mitigation_frame(tmp_path, ["minilm_l6"], [0, 1])
        assert got["score_delta"].tolist() == [-0.2, -0.1]
        assert got["separation_change"].tolist() == pytest.approx([10.0, -5.0])


FORMATS = ["a", "b", "c", "d"]


class TestSplitPairs:
    def test_separates_training_and_held_out_pairs(self):
        effects = np.ones((4, 4)) - np.eye(4)
        effects[3, :3] = effects[:3, 3] = 5.0
        got = mod.split_pairs(effects, FORMATS, "d")
        assert got == {"training": 1.0, "held_out": 5.0}

    def test_uses_every_pair(self):
        effects = np.arange(16.0).reshape(4, 4)
        effects = effects + effects.T
        got = mod.split_pairs(effects, FORMATS, "a")
        assert got["held_out"] == pytest.approx(
            np.mean([effects[0, i] for i in (1, 2, 3)])
        )
        assert got["training"] == pytest.approx(
            np.mean([effects[1, 2], effects[1, 3], effects[2, 3]])
        )


def spread(values):
    return {"mean": sum(values) / len(values), "per_seed": values}


class TestRatioFrame:
    @pytest.fixture
    def results(self, tmp_path):
        path = mod.pairs_path(tmp_path, "minilm_l6")
        path.parent.mkdir(parents=True, exist_ok=True)
        summary = {
            mod.WINNER_OVER_CONTROL: {
                "whitened": {
                    "training": spread([0.6, 0.8]),
                    "held_out": spread([0.7, 0.9]),
                }
            }
        }
        path.write_text(json.dumps({"seeds": [0, 1], "summary": summary}))
        return tmp_path

    def test_reports_each_set_over_seeds(self, results):
        got = mod.ratio_frame(results, ["minilm_l6"], "whitened").loc["MiniLM-L6"]
        assert got["training_mean"] == pytest.approx(0.7)
        assert got["held_out_mean"] == pytest.approx(0.8)
        assert got["training_sd"] == pytest.approx(0.1414, rel=1e-3)
        assert got["difference"] == pytest.approx(0.1)

    def test_keeps_each_seed(self, results):
        got = mod.ratio_frame(results, ["minilm_l6"], "whitened").loc["MiniLM-L6"]
        assert got["held_out seed 1"] == pytest.approx(0.9)

    def test_refuses_missing_results(self, tmp_path):
        with pytest.raises(SystemExit, match="no pair results"):
            mod.ratio_frame(tmp_path, ["minilm_l6"], "whitened")
