"""Tests for fsr.head_probe.frontier."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from fsr.formats import FORMAT_NAMES
from fsr.head_probe.frontier import frontier_point, mrr_per_format, score_store
from fsr.head_probe.heads import LINEAR, build_head
from fsr.head_probe.store import FeatureStore

FORMATS = list(FORMAT_NAMES)
N, C, D = 30, 6, 4


def store(seed: int = 0) -> FeatureStore:
    """Build a cache whose gold rows outscore its negatives."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1.0, size=(N, C, len(FORMATS), D)).astype(np.float32)
    features[:, 0, :, 0] += 4.0
    return FeatureStore(
        features=features,
        scores=features.sum(axis=-1),
        record_ids=[f"r{i}" for i in range(N)],
        formats=list(FORMATS),
        model="minilm_l6",
        split="dev",
    )


class SumHead(nn.Module):
    """A head that scores the first element of a representation."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return the leading element, with a trailing axis."""
        return features[..., :1]


class TestScoreStore:
    def test_returns_a_gold_score_per_record_and_format(self):
        gold, _ = score_store(SumHead(), store(), FORMATS, 3)
        assert set(gold) == set(FORMATS)
        assert len(gold[FORMATS[0]]) == N

    def test_returns_negative_scores_with_a_format_axis(self):
        _, negatives = score_store(SumHead(), store(), FORMATS, 3)
        assert negatives.shape == (N, len(FORMATS), 3)

    def test_chunking_does_not_change_the_scores(self):
        one, _ = score_store(SumHead(), store(), FORMATS, 3, chunk=1)
        many, _ = score_store(SumHead(), store(), FORMATS, 3, chunk=N)
        assert one == many

    def test_the_head_sees_the_representation_it_was_given(self):
        built = store()
        gold, _ = score_store(SumHead(), built, FORMATS, 3)
        assert np.allclose(gold[FORMATS[0]], built.features[:, 0, 0, 0])

    def test_scores_under_no_gradient(self):
        head = build_head(LINEAR, D)
        score_store(head, store(), FORMATS, 3)
        assert all(p.grad is None for p in head.parameters())


class TestMrrPerFormat:
    def test_a_gold_that_leads_everywhere_scores_one(self):
        gold = {name: [5.0, 5.0] for name in FORMATS}
        negatives = np.zeros((2, len(FORMATS), 3))
        assert mrr_per_format(gold, negatives, FORMATS) == dict.fromkeys(FORMATS, 1.0)

    def test_a_gold_behind_every_negative_scores_the_last_rank(self):
        gold = {name: [0.0] for name in FORMATS}
        negatives = np.ones((1, len(FORMATS), 3))
        assert mrr_per_format(gold, negatives, FORMATS)[FORMATS[0]] == 0.25

    def test_each_format_is_measured_on_its_own_column(self):
        gold = {name: [5.0] for name in FORMATS}
        gold[FORMATS[1]] = [0.0]
        negatives = np.ones((1, len(FORMATS), 3))
        scored = mrr_per_format(gold, negatives, FORMATS)
        assert scored[FORMATS[0]] == 1.0
        assert scored[FORMATS[1]] == 0.25


class TestFrontierPoint:
    def test_reports_both_axes(self):
        point = frontier_point(SumHead(), store(), FORMATS, 3)
        assert "max_abs_cohen_d" in point
        assert "mean_mrr" in point

    def test_a_separable_cache_gives_a_high_mrr(self):
        assert frontier_point(SumHead(), store(), FORMATS, 3)["mean_mrr"] > 0.9

    def test_keeps_the_full_sensitivity_summary(self):
        point = frontier_point(SumHead(), store(), FORMATS, 3)
        assert set(point["format_sensitivity"]) == {
            "per_format",
            "pairwise",
            "rank",
            "summary",
        }

    def test_reports_the_worst_format_separately(self):
        point = frontier_point(SumHead(), store(), FORMATS, 3)
        assert point["min_mrr"] <= point["mean_mrr"]

    def test_counts_the_records_it_measured(self):
        assert frontier_point(SumHead(), store(), FORMATS, 3)["n_records"] == N

    def test_the_seed_reaches_the_rank_sample(self):
        """A different seed has to change the sampled rank statistics."""
        first = frontier_point(SumHead(), store(), FORMATS, 3, seed=1)
        second = frontier_point(SumHead(), store(), FORMATS, 3, seed=2)
        assert first["max_abs_cohen_d"] == second["max_abs_cohen_d"]
        assert (
            first["format_sensitivity"]["rank"] != second["format_sensitivity"]["rank"]
        )

    def test_identical_formats_report_no_sensitivity(self):
        """Cohen's d is scale invariant, so only equal formats give zero."""
        rng = np.random.default_rng(5)
        one = rng.normal(0, 1.0, size=(N, C, 1, D)).astype(np.float32)
        features = np.repeat(one, len(FORMATS), axis=2)
        flat = FeatureStore(
            features=features,
            scores=features.sum(axis=-1),
            record_ids=[f"r{i}" for i in range(N)],
            formats=list(FORMATS),
            model="minilm_l6",
            split="dev",
        )
        point = frontier_point(SumHead(), flat, FORMATS, 3)
        assert point["max_abs_cohen_d"] == 0.0
        assert len(set(point["mrr_per_format"].values())) == 1


class TestCollapseGuard:
    def test_refuses_a_head_that_gives_every_record_one_score(self):
        """A collapsed head scores max|d| 0 and MRR 1, the dominating corner."""

        class Constant(nn.Module):
            def forward(self, features):
                return torch.ones((*features.shape[:-1], 1))

        with pytest.raises(ValueError, match="same score"):
            frontier_point(Constant(), store(), FORMATS, 3)

    def test_allows_a_head_whose_scores_still_vary(self):
        point = frontier_point(SumHead(), store(), FORMATS, 3)
        assert point["mean_mrr"] > 0.0


class TestDevice:
    def test_scores_on_the_device_it_is_given(self):
        point = frontier_point(SumHead(), store(), FORMATS, 3, device="cpu")
        assert point["n_records"] == N
