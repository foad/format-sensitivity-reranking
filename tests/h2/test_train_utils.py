"""Tests for fsr.h2.train_utils."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from fsr.h2.train_utils import (
    TrainRecord,
    _var_across_formats,
    compute_loss,
    forward_batch,
    render_negative,
    render_positive,
)

PAIRS = [("Born", "1946"), ("Role", "Engineer")]
NEG_PAIRS = [[("Born", "1801")], [("Born", "1902")]]
FORMATS_USED = ("yaml", "json")
HIDDEN = 8


def record(rec_id: str = "1") -> TrainRecord:
    """Return a record with two hard negatives."""
    return TrainRecord(
        id=rec_id,
        question=f"question {rec_id}",
        pairs=PAIRS,
        truncated_body="positive body",
        body_budget_tokens=400,
        neg_pairs_list=NEG_PAIRS,
        neg_bodies_truncated=["first negative", "second negative"],
    )


class Model(nn.Module):
    """A model stub that scores a batch through a linear head."""

    def __init__(self) -> None:
        """Build the scoring head."""
        super().__init__()
        self.classifier = nn.Linear(HIDDEN, 1)

    def forward(self, input_ids=None, **_kwargs):
        """Return one logit per row of the batch."""
        batch = input_ids.shape[0]
        torch.manual_seed(batch)
        return type("Out", (), {"logits": self.classifier(torch.rand(batch, HIDDEN))})()


class Tokenizer:
    """A tokenizer that returns an encoding sized to the batch."""

    def __call__(self, queries, _passages=None, **_kwargs):
        """Return an encoding whose batch size matches the queries."""
        return Encoding(input_ids=torch.zeros(len(queries), 4, dtype=torch.long))


class Encoding(dict):
    """An encoding that accepts a device move."""

    def to(self, _device):
        """Return the encoding unchanged."""
        return self


class TestRendering:
    def test_renders_the_positive_passage(self):
        out = render_positive(record(), "yaml")
        assert "Born: 1946" in out
        assert out.endswith("positive body")

    def test_renders_a_negative_passage_by_index(self):
        assert "1902" in render_negative(record(), 1, "yaml")
        assert render_negative(record(), 1, "yaml").endswith("second negative")

    def test_uses_the_named_format(self):
        assert render_positive(record(), "json").startswith('{"Born"')


class TestVarAcrossFormats:
    def test_returns_zero_for_a_single_format(self):
        assert _var_across_formats(torch.rand(3, 1)) == 0.0

    def test_returns_zero_when_formats_agree(self):
        t = torch.ones(3, 4)
        assert _var_across_formats(t) == 0.0

    def test_grows_with_disagreement(self):
        low = _var_across_formats(torch.tensor([[1.0, 1.1]]))
        high = _var_across_formats(torch.tensor([[1.0, 5.0]]))
        assert high > low

    def test_uses_population_variance(self):
        t = torch.tensor([[0.0, 2.0]])
        assert _var_across_formats(t) == pytest.approx(1.0)

    def test_averages_the_remaining_dimensions(self):
        t = torch.tensor([[[0.0, 0.0], [2.0, 4.0]]])
        assert _var_across_formats(t) == pytest.approx(2.5)


class TestComputeLoss:
    POS = torch.tensor([[2.0, 2.0]])
    NEG = torch.tensor([[[0.0], [0.0]]])

    def test_returns_the_total_and_both_terms(self):
        total, l_rank, l_inv = compute_loss(self.POS, self.NEG, 1.0)
        assert total == pytest.approx(l_rank + l_inv)

    def test_a_confident_ranking_gives_a_small_ranking_loss(self):
        _, close, _ = compute_loss(torch.tensor([[0.1]]), torch.tensor([[[0.0]]]), 0.0)
        _, far, _ = compute_loss(torch.tensor([[9.0]]), torch.tensor([[[0.0]]]), 0.0)
        assert far < close

    def test_agreeing_formats_give_no_invariance_loss(self):
        _, _, l_inv = compute_loss(self.POS, self.NEG, 1.0)
        assert l_inv == 0.0

    def test_disagreeing_formats_raise_the_invariance_loss(self):
        _, _, l_inv = compute_loss(torch.tensor([[0.0, 4.0]]), self.NEG, 1.0)
        assert l_inv == pytest.approx(4.0)

    def test_lambda_scales_only_the_invariance_term(self):
        pos = torch.tensor([[0.0, 4.0]])
        total_1, rank_1, inv_1 = compute_loss(pos, self.NEG, 1.0)
        total_3, rank_3, inv_3 = compute_loss(pos, self.NEG, 3.0)
        assert rank_1 == rank_3
        assert inv_1 == inv_3
        assert total_3 - total_1 == pytest.approx(2 * inv_1)

    def test_a_zero_lambda_leaves_the_ranking_loss(self):
        total, l_rank, _ = compute_loss(torch.tensor([[0.0, 4.0]]), self.NEG, 0.0)
        assert total == pytest.approx(l_rank)

    def test_the_loss_is_differentiable(self):
        pos = torch.tensor([[0.0, 4.0]], requires_grad=True)
        total, _, _ = compute_loss(pos, self.NEG, 1.0)
        total.backward()
        assert pos.grad is not None


class TestForwardBatch:
    def test_returns_scores_shaped_by_records_formats_and_negatives(self):
        pos, neg = forward_batch(
            Model(), Tokenizer(), [record("1"), record("2")], FORMATS_USED, "cpu"
        )
        assert pos.shape == (2, 2)
        assert neg.shape == (2, 2, 2)

    def test_a_single_format_still_produces_a_format_axis(self):
        pos, neg = forward_batch(Model(), Tokenizer(), [record("1")], ("yaml",), "cpu")
        assert pos.shape == (1, 1)
        assert neg.shape == (1, 1, 2)

    def test_passes_the_token_limit_to_the_tokenizer(self):
        seen = {}

        class Recording(Tokenizer):
            def __call__(self, queries, passages=None, **kwargs):
                seen.update(kwargs)
                return super().__call__(queries, passages, **kwargs)

        forward_batch(
            Model(), Recording(), [record("1")], ("yaml",), "cpu", max_tokens=128
        )
        assert seen["max_length"] == 128
