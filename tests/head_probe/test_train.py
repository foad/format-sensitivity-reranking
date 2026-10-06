"""Tests for fsr.head_probe.train."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fsr.head_probe.heads import LINEAR, TANH, build_head
from fsr.head_probe.store import FeatureStore
from fsr.head_probe.train import (
    HeadProbeConfig,
    record_order,
    score_features,
    train_head,
)

FORMATS = ["yaml", "json", "toml"]
N, C, D = 40, 8, 5


def store(seed: int = 0, separable: bool = True) -> FeatureStore:
    """Build a cache whose gold rows are separable from its negatives."""
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 1.0, size=(N, C, len(FORMATS), D)).astype(np.float32)
    if separable:
        features[:, 0, :, 0] += 3.0
    return FeatureStore(
        features=features,
        scores=features.sum(axis=-1),
        record_ids=[f"r{i}" for i in range(N)],
        formats=list(FORMATS),
        model="minilm_l6",
        split="train",
    )


def config(**kwargs) -> HeadProbeConfig:
    base = {
        "lambda_inv": 0.0,
        "max_steps": 60,
        "batch_size": 8,
        "n_negatives": 3,
        "warmup_steps": 5,
        "log_every": 20,
    }
    base.update(kwargs)
    return HeadProbeConfig(**base)


class TestScoreFeatures:
    def test_drops_the_representation_axis(self):
        head = build_head(LINEAR, D)
        assert score_features(head, torch.zeros(4, 3, D)).shape == (4, 3)

    def test_handles_the_negative_axis(self):
        head = build_head(LINEAR, D)
        assert score_features(head, torch.zeros(4, 3, 2, D)).shape == (4, 3, 2)


class TestRecordOrder:
    def test_gives_one_row_per_step_and_batch_position(self):
        order = record_order(N, config(max_steps=7, batch_size=3))
        assert order.shape == (7, 3)

    def test_draws_every_record_before_repeating_one(self):
        order = record_order(10, config(max_steps=1, batch_size=10))
        assert sorted(order[0].tolist()) == list(range(10))

    def test_covers_more_steps_than_the_cache_holds_records(self):
        order = record_order(4, config(max_steps=10, batch_size=4))
        assert order.shape == (10, 4)

    def test_the_same_seed_gives_the_same_order(self):
        assert np.array_equal(
            record_order(N, config(seed=3)), record_order(N, config(seed=3))
        )

    def test_a_different_seed_gives_a_different_order(self):
        assert not np.array_equal(
            record_order(N, config(seed=1)), record_order(N, config(seed=2))
        )


class TestTrainHead:
    def test_returns_the_steps_it_took(self):
        result = train_head(build_head(LINEAR, D), store(), config(), FORMATS)
        assert result.steps == 60

    def test_the_ranking_loss_falls_on_separable_features(self):
        result = train_head(
            build_head(LINEAR, D),
            store(),
            config(max_steps=400, log_every=100),
            FORMATS,
        )
        assert result.history[-1]["rank"] < result.history[0]["rank"]

    def test_the_invariance_term_falls_when_it_is_weighted(self):
        """A weighted invariance term has to reduce the spread it measures."""
        free = train_head(
            build_head(TANH, D, seed=1), store(), config(lambda_inv=0.0), FORMATS
        )
        held = train_head(
            build_head(TANH, D, seed=1), store(), config(lambda_inv=10.0), FORMATS
        )
        assert held.inv_loss < free.inv_loss

    def test_logs_a_history_entry_every_log_every_steps(self):
        result = train_head(
            build_head(LINEAR, D), store(), config(max_steps=60, log_every=20), FORMATS
        )
        assert [h["step"] for h in result.history] == [20, 40, 60]

    def test_always_logs_the_last_step(self):
        result = train_head(
            build_head(LINEAR, D), store(), config(max_steps=25, log_every=20), FORMATS
        )
        assert result.history[-1]["step"] == 25

    def test_the_same_seed_gives_the_same_result(self):
        first = train_head(build_head(LINEAR, D, seed=4), store(), config(), FORMATS)
        second = train_head(build_head(LINEAR, D, seed=4), store(), config(), FORMATS)
        assert first.loss == second.loss

    def test_leaves_the_head_in_evaluation_mode(self):
        head = build_head(TANH, D)
        train_head(head, store(), config(), FORMATS)
        assert not head.training

    def test_reports_each_step(self):
        seen = []
        train_head(
            build_head(LINEAR, D),
            store(),
            config(max_steps=5),
            FORMATS,
            on_step=seen.append,
        )
        assert seen == [1, 2, 3, 4, 5]

    def test_a_subset_of_formats_narrows_the_invariance_term(self):
        result = train_head(build_head(LINEAR, D), store(), config(), ["yaml"])
        assert result.inv_loss == 0.0

    def test_refuses_a_batch_larger_than_the_cache(self):
        with pytest.raises(ValueError, match="holds 40 records"):
            train_head(build_head(LINEAR, D), store(), config(batch_size=99), FORMATS)

    def test_updates_every_parameter(self):
        head = build_head(TANH, D, seed=2)
        before = [p.detach().clone() for p in head.parameters()]
        train_head(head, store(), config(), FORMATS)
        assert all(
            not torch.equal(a, b)
            for a, b in zip(before, head.parameters(), strict=True)
        )
