"""Tests for fsr.representation.robustness."""

from __future__ import annotations

import pytest

from fsr.representation import robustness as mod


def payload(shrinkage=1e-3, seed=0):
    return {
        "shrinkage": shrinkage,
        "split_seed": seed,
        "separation_index": {"bge_base": 0.05, "a": 0.6, "b": 0.9},
        "pooled_rho": 0.4,
        "ratios": {
            "a": {"training": [1.0, 1.2], "held_out": [0.9, 1.0]},
            "b": {"training": [1.3, 1.3], "held_out": [1.1, 1.3]},
        },
    }


class TestSettings:
    def test_covers_every_shrinkage_and_split_seed(self):
        grid = mod.settings()
        assert {s.shrinkage for s in grid} == set(mod.SHRINKAGES)
        assert {s.seed for s in grid} == set(mod.SPLIT_SEEDS)

    def test_other_splits_use_the_default_shrinkage(self):
        assert all(s.shrinkage == 1e-3 for s in mod.settings() if s.seed != 0)

    def test_tags_are_unique(self):
        tags = [s.tag for s in mod.settings()]
        assert len(tags) == len(set(tags))


class TestSummaries:
    def test_seed_mean_ratios(self):
        got = mod.seed_mean_ratios(payload()["ratios"])
        assert got["a"]["training"] == pytest.approx(1.1)
        assert got["a"]["held_out"] == pytest.approx(0.95)

    def test_setting_row(self):
        got = mod.setting_row(payload())
        assert got["bge_base_separation"] == 0.05
        assert got["other_lowest_separation"] == 0.6
        assert got["lowest_ratio"] == pytest.approx(0.95)
        assert got["highest_ratio"] == pytest.approx(1.3)

    def test_range_table_has_one_row_per_setting(self):
        got = mod.range_table([payload(1e-4), payload(1e-3, 1)])
        assert list(got.index) == [
            "shrinkage 0.0001, split 0",
            "shrinkage 0.001, split 1",
        ]
