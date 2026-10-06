"""Tests for fsr.features."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from fsr.features import (
    capture_encoder_output,
    encoder_module,
    features_and_scores,
)
from tests.fakes import FakeEncoder, HashingPairTokenizer, ScoringModel

PAIRS = [("who", "a passage"), ("what", "another"), ("when", "a third")]


class PooledModel(nn.Module):
    """A scorer with a pooler between the encoder and the head, as mxbai has."""

    def __init__(self, width: int = 4) -> None:
        """Build the encoder, the pooler and the head."""
        super().__init__()
        self.encoder = FakeEncoder()
        self.pooler = nn.Sequential(nn.Linear(width, width), nn.GELU())
        self.classifier = nn.Linear(width, 1)

    @property
    def base_model(self) -> nn.Module:
        """Return the encoder below the pooler."""
        return self.encoder

    def forward(self, input_ids=None, **_kwargs):
        """Return one logit per row, through the pooler."""
        hidden = self.encoder(input_ids=input_ids)[0]
        return types.SimpleNamespace(logits=self.classifier(self.pooler(hidden[:, 0])))


class TestEncoderModule:
    def test_returns_the_base_transformer(self):
        model = ScoringModel(4)
        assert encoder_module(model) is model.encoder

    def test_returns_the_encoder_below_a_pooler(self):
        model = PooledModel(4)
        assert encoder_module(model) is model.encoder

    def test_refuses_a_model_with_no_base_transformer(self):
        with pytest.raises(RuntimeError, match="no base transformer"):
            encoder_module(nn.Linear(4, 1))


class TestCaptureEncoderOutput:
    def test_collects_one_tensor_per_forward(self):
        model = ScoringModel(4)
        with capture_encoder_output(model) as captured:
            model(input_ids=torch.ones(2, 4, dtype=torch.long))
            model(input_ids=torch.ones(3, 4, dtype=torch.long))
        assert [t.shape[0] for t in captured] == [2, 3]

    def test_takes_the_first_position_of_the_sequence(self):
        model = ScoringModel(4)
        ids = torch.arange(8, dtype=torch.long).reshape(2, 4)
        with capture_encoder_output(model) as captured:
            model(input_ids=ids)
        assert torch.allclose(captured[0], ids.to(torch.float32) / 1000.0)

    def test_removes_the_hook_on_exit(self):
        model = ScoringModel(4)
        with capture_encoder_output(model) as captured:
            pass
        model(input_ids=torch.ones(2, 4, dtype=torch.long))
        assert captured == []

    def test_removes_the_hook_when_the_body_raises(self):
        model = ScoringModel(4)
        with pytest.raises(ValueError), capture_encoder_output(model):
            raise ValueError("stop")
        model(input_ids=torch.ones(2, 4, dtype=torch.long))
        assert not model.encoder._forward_hooks


class TestFeaturesAndScores:
    def test_returns_one_row_per_pair(self):
        features, scores = features_and_scores(
            ScoringModel(4), HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        assert features.shape == (len(PAIRS), 4)
        assert len(scores) == len(PAIRS)

    def test_the_representation_reproduces_the_score(self):
        """A head replayed on the captured vector gives the model's own score."""
        model = ScoringModel(4)
        features, scores = features_and_scores(
            model, HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        replayed = model.classifier(torch.tensor(features)).squeeze(-1)
        assert np.allclose(replayed.detach().numpy(), scores, atol=1e-5)

    def test_cuts_below_a_pooler_rather_than_above_it(self):
        """The cut must not vary with where a vendor draws the head boundary."""
        model = PooledModel(4)
        features, scores = features_and_scores(
            model, HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        pooled = model.pooler(torch.tensor(features))
        replayed = model.classifier(pooled).squeeze(-1)
        assert np.allclose(replayed.detach().numpy(), scores, atol=1e-5)
        assert not np.allclose(features, pooled.detach().numpy())

    def test_batching_does_not_change_the_representation(self):
        model = ScoringModel(4)
        one, _ = features_and_scores(model, HashingPairTokenizer(4), PAIRS, 1, "cpu")
        many, _ = features_and_scores(model, HashingPairTokenizer(4), PAIRS, 8, "cpu")
        assert np.allclose(one, many, atol=1e-6)

    def test_reports_progress(self):
        seen = []
        features_and_scores(
            ScoringModel(4),
            HashingPairTokenizer(4),
            PAIRS,
            2,
            "cpu",
            on_batch=seen.append,
        )
        assert seen == [2, 3]

    def test_handles_no_pairs(self):
        features, scores = features_and_scores(
            ScoringModel(4), HashingPairTokenizer(4), [], 2, "cpu"
        )
        assert features.shape == (0, 0)
        assert scores == []

    def test_refuses_a_representation_that_is_not_one_vector_per_pair(self):
        class CubeEncoder(nn.Module):
            def forward(self, input_ids=None, **_kwargs):
                scaled = input_ids.to(torch.float32)
                return (torch.stack([torch.stack([scaled] * 2, 1)] * 2, 1),)

        class Cube(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = CubeEncoder()
                self.classifier = nn.Linear(4, 1)

            @property
            def base_model(self):
                return self.encoder

            def forward(self, input_ids=None, **_kwargs):
                hidden = self.encoder(input_ids=input_ids)[0]
                return types.SimpleNamespace(
                    logits=self.classifier(hidden[:, 0].mean(dim=1))
                )

        with pytest.raises(RuntimeError, match="3-dimensional"):
            features_and_scores(Cube(), HashingPairTokenizer(4), PAIRS, 2, "cpu")

    def test_refuses_a_count_that_does_not_match_the_pairs(self):
        """An encoder called twice per forward yields more rows than scores."""

        class TwiceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = FakeEncoder()
                self.classifier = nn.Linear(4, 1)

            @property
            def base_model(self):
                return self.encoder

            def forward(self, input_ids=None, **_kwargs):
                self.encoder(input_ids=input_ids)
                hidden = self.encoder(input_ids=input_ids)[0]
                return types.SimpleNamespace(logits=self.classifier(hidden[:, 0]))

        with pytest.raises(RuntimeError, match="representations for"):
            features_and_scores(TwiceModel(), HashingPairTokenizer(4), PAIRS, 2, "cpu")
