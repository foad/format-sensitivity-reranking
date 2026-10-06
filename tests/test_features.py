"""Tests for fsr.features."""

from __future__ import annotations

import types

import numpy as np
import pytest
import torch
import torch.nn as nn

from fsr.features import (
    capture_classifier_input,
    classifier_input_module,
    features_and_scores,
)
from tests.fakes import HashingPairTokenizer, ScoringModel

PAIRS = [("who", "a passage"), ("what", "another"), ("when", "a third")]


class TwoLayerHead(nn.Module):
    """A head shaped like jina's: dense, tanh, then a projection."""

    def __init__(self, width: int = 4) -> None:
        """Build the two-layer head."""
        super().__init__()
        self.dense = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Score the first token of each row."""
        x = features[:, 0, :] if features.ndim == 3 else features
        return self.out_proj(torch.tanh(self.dense(x)))


class SequenceModel(nn.Module):
    """A scorer whose head receives the whole sequence, as jina's does."""

    def __init__(self, width: int = 4) -> None:
        """Build the sequence scorer."""
        super().__init__()
        self.classifier = TwoLayerHead(width)

    def forward(self, input_ids=None, **_kwargs):
        """Return one logit per row, from a two-token sequence."""
        scaled = input_ids.to(torch.float32) / 1000.0
        sequence = torch.stack([scaled, scaled * 2], dim=1)
        return types.SimpleNamespace(logits=self.classifier(sequence))


class TestClassifierInputModule:
    def test_returns_a_single_linear_classifier(self):
        model = ScoringModel(4)
        assert classifier_input_module(model) is model.classifier

    def test_returns_the_first_linear_of_a_deeper_head(self):
        model = SequenceModel(4)
        assert classifier_input_module(model) is model.classifier.dense

    def test_refuses_a_model_without_a_classifier(self):
        with pytest.raises(RuntimeError, match="no `classifier`"):
            classifier_input_module(nn.Module())

    def test_refuses_a_classifier_with_no_linear(self):
        model = nn.Module()
        model.classifier = nn.Sequential(nn.Tanh())
        with pytest.raises(RuntimeError, match=r"no.*nn\.Linear"):
            classifier_input_module(model)


class TestCaptureClassifierInput:
    def test_collects_one_tensor_per_forward(self):
        model = ScoringModel(4)
        with capture_classifier_input(model) as captured:
            model(input_ids=torch.ones(2, 4, dtype=torch.long))
            model(input_ids=torch.ones(3, 4, dtype=torch.long))
        assert [t.shape[0] for t in captured] == [2, 3]

    def test_removes_the_hook_on_exit(self):
        model = ScoringModel(4)
        with capture_classifier_input(model) as captured:
            pass
        model(input_ids=torch.ones(2, 4, dtype=torch.long))
        assert captured == []

    def test_removes_the_hook_when_the_body_raises(self):
        model = ScoringModel(4)
        with pytest.raises(ValueError), capture_classifier_input(model):
            raise ValueError("stop")
        model(input_ids=torch.ones(2, 4, dtype=torch.long))
        assert not model.classifier._forward_hooks


class TestFeaturesAndScores:
    def test_returns_one_row_per_pair(self):
        features, scores = features_and_scores(
            ScoringModel(4), HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        assert features.shape == (len(PAIRS), 4)
        assert len(scores) == len(PAIRS)

    def test_the_representation_reproduces_the_score(self):
        """The captured vector is what the head turned into the score."""
        model = ScoringModel(4)
        features, scores = features_and_scores(
            model, HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        replayed = model.classifier(torch.tensor(features)).squeeze(-1)
        assert np.allclose(replayed.detach().numpy(), scores, atol=1e-5)

    def test_captures_the_pooled_row_of_a_sequence_head(self):
        model = SequenceModel(4)
        features, scores = features_and_scores(
            model, HashingPairTokenizer(4), PAIRS, 2, "cpu"
        )
        assert features.shape == (len(PAIRS), 4)
        replayed = model.classifier(torch.tensor(features)).squeeze(-1)
        assert np.allclose(replayed.detach().numpy(), scores, atol=1e-5)

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
        class CubeHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(4, 1)

            def forward(self, features):
                return self.inner(features).mean(dim=1)

        model = nn.Module()
        model.classifier = CubeHead()
        model.forward = lambda input_ids=None, **_k: types.SimpleNamespace(
            logits=model.classifier(
                torch.stack(
                    [input_ids.to(torch.float32)] * 2,
                    dim=1,
                )
            )
        )
        with pytest.raises(RuntimeError, match="3-dimensional"):
            features_and_scores(model, HashingPairTokenizer(4), PAIRS, 2, "cpu")

    def test_refuses_a_count_that_does_not_match_the_pairs(self):
        """A head called twice per forward yields more rows than scores."""

        class TwiceModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(4, 1)

            def forward(self, input_ids=None, **_kwargs):
                scaled = input_ids.to(torch.float32)
                self.classifier(scaled)
                return types.SimpleNamespace(logits=self.classifier(scaled))

        with pytest.raises(RuntimeError, match="representations for"):
            features_and_scores(TwiceModel(), HashingPairTokenizer(4), PAIRS, 2, "cpu")
