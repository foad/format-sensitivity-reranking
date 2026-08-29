"""Tests for fsr.common.scoring."""

from __future__ import annotations

import pytest
import torch
import transformers.models.xlm_roberta.modeling_xlm_roberta as xlm_mod
from tests.fakes import LogitModel, PairTokenizer

from fsr.common.scoring import MAX_TOKENS, extract_score_from_logits, score_batch

PAIRS = [(f"q{i}", f"p{i}") for i in range(7)]


class TestXlmRobertaPatch:
    def test_the_position_id_helper_is_present(self):
        assert hasattr(xlm_mod, "create_position_ids_from_input_ids")

    def test_the_helper_skips_padding(self):
        ids = torch.tensor([[5, 6, 7, 1, 1]])
        result = xlm_mod.create_position_ids_from_input_ids(ids, padding_idx=1)
        assert result.tolist() == [[2, 3, 4, 1, 1]]


class TestExtractScoreFromLogits:
    def test_squeezes_a_single_logit_head(self):
        logits = torch.tensor([[0.5], [1.5]])
        assert extract_score_from_logits(logits).tolist() == [0.5, 1.5]

    def test_uses_index_one_of_a_binary_head(self):
        logits = torch.tensor([[0.1, 0.9], [0.3, 0.7]])
        assert extract_score_from_logits(logits).tolist() == pytest.approx([0.9, 0.7])

    def test_passes_a_one_dimensional_output_through(self):
        logits = torch.tensor([0.2, 0.4])
        assert extract_score_from_logits(logits).tolist() == pytest.approx([0.2, 0.4])

    def test_uses_column_zero_for_any_other_shape(self):
        logits = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        assert extract_score_from_logits(logits).tolist() == pytest.approx([0.1, 0.4])

    def test_preserves_the_batch_size(self):
        assert extract_score_from_logits(torch.zeros(7, 1)).shape == (7,)


class TestScoreBatch:
    def test_returns_one_score_per_pair(self):
        scores = score_batch(LogitModel(), PairTokenizer(), PAIRS, 3, "cpu")
        assert len(scores) == len(PAIRS)

    def test_returns_plain_floats(self):
        scores = score_batch(LogitModel(), PairTokenizer(), PAIRS, 3, "cpu")
        assert all(isinstance(s, float) for s in scores)

    def test_handles_a_batch_size_larger_than_the_input(self):
        scores = score_batch(LogitModel(), PairTokenizer(), PAIRS, 99, "cpu")
        assert len(scores) == len(PAIRS)

    def test_handles_no_pairs(self):
        assert score_batch(LogitModel(), PairTokenizer(), [], 4, "cpu") == []

    def test_accepts_every_logit_shape(self):
        for shape in [(1,), (2,), (), (3,)]:
            scores = score_batch(LogitModel(shape), PairTokenizer(), PAIRS, 4, "cpu")
            assert len(scores) == len(PAIRS)

    def test_passes_the_token_limit_to_the_tokenizer(self):
        seen = {}

        class RecordingTokenizer(PairTokenizer):
            def __call__(self, queries, passages=None, **kwargs):
                seen.update(kwargs)
                return super().__call__(queries, passages, **kwargs)

        score_batch(LogitModel(), RecordingTokenizer(), PAIRS, 4, "cpu")
        assert seen["max_length"] == MAX_TOKENS
        assert seen["truncation"] is True
        assert seen["padding"] is True

    def test_accepts_a_custom_token_limit(self):
        seen = {}

        class RecordingTokenizer(PairTokenizer):
            def __call__(self, queries, passages=None, **kwargs):
                seen.update(kwargs)
                return super().__call__(queries, passages, **kwargs)

        score_batch(LogitModel(), RecordingTokenizer(), PAIRS, 4, "cpu", max_tokens=128)
        assert seen["max_length"] == 128

    def test_batching_does_not_change_the_pair_count(self):
        tok, model = PairTokenizer(), LogitModel()
        assert len(score_batch(model, tok, PAIRS, 1, "cpu")) == len(
            score_batch(model, tok, PAIRS, 7, "cpu")
        )
