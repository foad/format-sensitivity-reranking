"""Tests for fsr.representation.seeds."""

from __future__ import annotations

import pytest

from fsr.representation import seeds as mod

FOLDS = ("a", "b")
SPACES = ("raw", "whitened")


def folds(training, held_out):
    one = {space: {"training": training, "held_out": held_out} for space in SPACES}
    return {f: one for f in FOLDS}


class TestSummarise:
    def test_summarises_only_the_spaces_named(self):
        seeds = {0: {0.0: folds(2.0, 4.0), 1.0: folds(1.0, 2.0)}}
        got = mod.summarise(folds(4.0, 4.0), seeds, SPACES)
        assert set(got["lambda 1"]) == set(SPACES)

    def test_ratio_of_the_winner_to_weight_zero(self):
        seeds = {
            0: {0.0: folds(2.0, 4.0), 1.0: folds(1.0, 2.0)},
            1: {0.0: folds(4.0, 4.0), 1.0: folds(1.0, 4.0)},
        }
        got = mod.summarise(folds(4.0, 4.0), seeds, SPACES)
        ratio = got["winner / lambda 0"]["whitened"]
        assert ratio["training"]["per_seed"] == pytest.approx([0.5, 0.25])
        assert ratio["held_out"]["per_seed"] == pytest.approx([0.5, 1.0])

    def test_spread_reports_the_seed_values(self):
        got = mod.spread([1.0, 3.0])
        assert got["mean"] == 2.0
        assert got["min"] == 1.0
        assert got["max"] == 3.0
