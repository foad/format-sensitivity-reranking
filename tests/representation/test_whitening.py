"""Tests for fsr.representation.whitening."""

from __future__ import annotations

import numpy as np

from fsr.representation import whitening as mod

WIDTH = 6


def axis(i):
    vector = np.zeros(WIDTH)
    vector[i] = 1.0
    return vector


class TestHalves:
    def test_partition_the_indices(self):
        a, b = mod.halves(9)
        assert sorted(np.concatenate([a, b]).tolist()) == list(range(9))
        assert len(a) == 4

    def test_are_reproducible(self):
        assert np.array_equal(mod.halves(10, seed=3)[0], mod.halves(10, seed=3)[0])


class TestContentDifferences:
    def test_hold_the_format_fixed(self):
        content = np.random.default_rng(5).normal(size=(6, 1, WIDTH))
        offsets = np.stack([axis(0), 5 * axis(1), axis(2), axis(3)])[None]
        diffs = mod.content_differences(content + offsets, [0, 1, 2])
        assert np.allclose(diffs[:, 0], diffs[:, 1])

    def test_pair_distinct_passages(self):
        passages = np.arange(8, dtype=float)[:, None, None] * np.ones((1, 2, WIDTH))
        diffs = mod.content_differences(passages, [0, 1])
        assert diffs.shape == (4, 2, WIDTH)
        assert np.all(diffs != 0)


class TestWhitener:
    def test_equalises_the_differences(self):
        rng = np.random.default_rng(8)
        diffs = rng.normal(size=(5000, WIDTH)) * np.arange(1, WIDTH + 1)
        white = mod.whitener(diffs, shrinkage=0.0)
        flat = diffs @ white
        assert np.allclose(flat.T @ flat / len(flat), np.eye(WIDTH), atol=1e-8)

    def test_shrinkage_keeps_an_unvarying_direction_finite(self):
        rng = np.random.default_rng(9)
        diffs = rng.normal(size=(500, WIDTH))
        diffs[:, 0] = 0.0
        assert np.all(np.isfinite(mod.whitener(diffs)))
