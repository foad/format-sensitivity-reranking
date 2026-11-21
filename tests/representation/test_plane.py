"""Tests for fsr.representation.plane."""

from __future__ import annotations

import numpy as np
import pytest

from fsr.representation import plane as mod

WIDTH = 6


def axis(i):
    vector = np.zeros(WIDTH)
    vector[i] = 1.0
    return vector


def cache(offsets, noise=0.0, records=200, candidates=2, seed=0):
    rng = np.random.default_rng(seed)
    content = rng.normal(size=(records, candidates, 1, WIDTH)) * 10
    features = content + np.stack(offsets)[None, None]
    return features + noise * rng.normal(size=features.shape)


class TestFormatShifts:
    def test_content_cancels(self):
        offsets = [axis(0), axis(1), -axis(0)]
        shifts = mod.format_shifts(cache(offsets))
        expected = np.stack(offsets) - np.mean(offsets, axis=0)
        assert np.allclose(shifts, expected[None])

    def test_each_passages_shifts_sum_to_zero(self):
        shifts = mod.format_shifts(cache([axis(i) for i in range(4)], noise=0.3))
        assert np.allclose(shifts.sum(axis=1), 0.0)


class TestPlane:
    def test_holds_all_of_a_planar_arrangement(self):
        offsets = [axis(0), axis(1), -axis(0), -axis(1)]
        basis, share = mod.plane(mod.format_shifts(cache(offsets)))
        assert share == pytest.approx(1.0)
        assert np.allclose(basis @ basis.T, np.eye(2))

    def test_spans_the_format_directions(self):
        offsets = [axis(2) * 3, -axis(2) * 3, axis(4), -axis(4)]
        basis, _ = mod.plane(mod.format_shifts(cache(offsets)))
        captured = (basis[:, [2, 4]] ** 2).sum()
        assert captured == pytest.approx(2.0)


class TestPlaneView:
    def test_separates_formats_on_unseen_records(self):
        offsets = [axis(0), axis(1), -axis(0), -axis(1)]
        got = mod.plane_view(cache(offsets, noise=0.05))
        assert got["points"].shape[1:] == (4, 2)
        distances = np.linalg.norm(got["means"][:, None] - got["means"][None], axis=-1)
        assert distances[np.triu_indices(4, 1)].min() > 1.0

    def test_reports_how_much_the_plane_holds(self):
        offsets = [axis(0), axis(1), -axis(0), -axis(1)]
        clean = mod.plane_view(cache(offsets, noise=0.0))
        noisy = mod.plane_view(cache(offsets, noise=1.0))
        assert clean["passage_share"] == pytest.approx(1.0)
        assert noisy["passage_share"] < clean["passage_share"]

    def test_whitening_discounts_a_direction_content_dominates(self):
        rng = np.random.default_rng(7)
        content = rng.normal(size=(400, 2, 1, WIDTH))
        content[..., 0] *= 50
        offsets = np.stack([axis(1), -axis(1), axis(2), -axis(2)])
        features = content + offsets[None, None]
        features[..., 0] += rng.normal(size=features.shape[:-1]) * 5
        raw = mod.plane_view(features)
        white = mod.plane_view(features, whiten=True)
        assert white["passage_share"] > raw["passage_share"]


class TestSeparation:
    def test_is_one_when_each_format_is_a_point(self):
        points = np.tile(np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 2.0]]), (10, 1, 1))
        assert mod.separation(points) == pytest.approx(1.0)

    def test_is_near_zero_when_the_formats_overlap(self):
        points = np.random.default_rng(8).normal(size=(5000, 3, 2))
        assert mod.separation(points) < 0.01

    def test_falls_as_the_scatter_grows(self):
        rng = np.random.default_rng(9)
        centres = np.array([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0]])
        tight = centres + 0.1 * rng.normal(size=(500, 3, 2))
        loose = centres + 2.0 * rng.normal(size=(500, 3, 2))
        assert mod.separation(tight) > mod.separation(loose)


class TestPairSeparation:
    def test_matches_a_paired_d_in_one_dimension(self):
        rng = np.random.default_rng(10)
        features = rng.normal(size=(400, 2, 2, 1))
        features[:, 0, 1, 0] = features[:, 0, 0, 0] + 1.0 + 0.5 * rng.normal(size=400)
        got = mod.pair_separation(features)
        _, show = mod.halves(400, 0)
        diff = features[show, 0, 0, 0] - features[show, 0, 1, 0]
        assert got[0, 1] == pytest.approx(abs(diff.mean()) / diff.std())

    def test_reads_only_the_gold(self):
        rng = np.random.default_rng(11)
        features = rng.normal(size=(100, 3, 3, WIDTH))
        shifted = features.copy()
        shifted[:, 1:, 0] += 9.0
        assert np.allclose(mod.pair_separation(features), mod.pair_separation(shifted))

    def test_a_fixed_offset_separates_more_than_noise(self):
        rng = np.random.default_rng(12)
        content = rng.normal(size=(300, 1, 1, WIDTH)) * 5
        offsets = np.stack([axis(0), axis(0) * 0, axis(0) * 0])[None, None]
        features = content + offsets + 0.2 * rng.normal(size=(300, 1, 3, WIDTH))
        got = mod.pair_separation(features, whiten=True)
        assert got[0, 1] > 3 * got[1, 2]


class TestShrinkage:
    def test_heavy_shrinkage_approaches_the_raw_separation(self):
        rng = np.random.default_rng(13)
        scales = np.array([20.0, 5.0, 1.0, 1.0, 0.5, 0.2])
        features = rng.normal(size=(300, 2, 3, WIDTH)) * scales
        features[:, :, 0, 1] += 0.5
        raw = mod.pair_separation(features)
        heavy = mod.pair_separation(features, whiten=True, shrinkage=1e6)
        light = mod.pair_separation(features, whiten=True, shrinkage=1e-3)
        assert np.allclose(heavy, raw, rtol=1e-3)
        assert not np.allclose(light, raw, rtol=1e-2)

    def test_plane_view_passes_the_shrinkage_on(self):
        features = cache([axis(0), axis(1), -axis(0), -axis(1)], noise=0.3)
        a = mod.plane_view(features, whiten=True, shrinkage=1e-3)
        b = mod.plane_view(features, whiten=True, shrinkage=10.0)
        assert a["separation"] != pytest.approx(b["separation"])


class TestDiscriminantPlane:
    def test_finds_the_directions_that_separate_formats(self):
        rng = np.random.default_rng(16)
        shifts = rng.normal(size=(500, 4, WIDTH)) * np.array([5, 5, 0.2, 0.2, 5, 5])
        shifts[:, :, 2] += np.array([1.0, -1.0, 0.0, 0.0])
        shifts[:, :, 3] += np.array([0.0, 0.0, 1.0, -1.0])
        basis, _ = mod.discriminant_plane(shifts)
        assert (basis[:, [2, 3]] ** 2).sum() == pytest.approx(2.0, abs=0.05)
        assert np.allclose(basis @ basis.T, np.eye(2), atol=1e-8)

    def test_plane_view_can_use_it(self):
        features = cache([axis(0), axis(1), -axis(0), -axis(1)], noise=0.3)
        got = mod.plane_view(features, whiten=True, method="discriminant")
        assert got["points"].shape[1:] == (4, 2)

    def test_gold_only_keeps_one_candidate(self):
        features = cache([axis(0), axis(1), -axis(0), -axis(1)], noise=0.3)
        got = mod.plane_view(features, gold_only=True)
        assert got["points"].shape[0] == features.shape[0] - features.shape[0] // 2
