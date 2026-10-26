"""Tests for fsr.standardisation.store."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr.standardisation.layout import tensor_meta_path, tensor_path
from fsr.standardisation.store import GOLD, ScoreTensor, load_tensor, save_tensor

FORMATS = ["yaml", "json", "toml", "inline_kv", "markdown"]
Q, C, F = 4, 3, len(FORMATS)


def tensor(scores=None, query_ids=None, formats=None):
    """Build a tensor whose values encode their own position."""
    if scores is None:
        scores = np.arange(Q * C * F, dtype=np.float32).reshape(Q, C, F)
    return ScoreTensor(
        scores=scores,
        query_ids=[f"q{i}" for i in range(Q)] if query_ids is None else query_ids,
        formats=list(FORMATS) if formats is None else formats,
        model="jina_v2",
        arm="base",
        split="test",
    )


class TestShape:
    def test_reports_each_axis(self):
        built = tensor()
        assert (built.n_queries, built.n_candidates, built.n_formats) == (Q, C, F)

    def test_the_gold_is_the_first_candidate(self):
        built = tensor()
        assert np.array_equal(built.gold, built.scores[:, GOLD])
        assert built.gold.shape == (Q, F)

    def test_the_negatives_are_the_rest(self):
        built = tensor()
        assert built.negatives.shape == (Q, C - 1, F)
        assert np.array_equal(built.negatives, built.scores[:, 1:])


class TestValidation:
    def test_refuses_scores_that_are_not_three_dimensional(self):
        with pytest.raises(ValueError, match="expected 3"):
            tensor(scores=np.zeros((Q, C), dtype=np.float32))

    def test_refuses_a_query_id_count_that_does_not_match(self):
        with pytest.raises(ValueError, match="query ids"):
            tensor(query_ids=["q0"])

    def test_refuses_a_format_count_that_does_not_match(self):
        with pytest.raises(ValueError, match="format names"):
            tensor(formats=["yaml"])

    def test_refuses_a_query_with_no_negative(self):
        """A candidate list of one cannot be ranked."""
        with pytest.raises(ValueError, match="at least one negative"):
            tensor(scores=np.zeros((Q, 1, F), dtype=np.float32))


class TestRoundTrip:
    def test_what_is_saved_is_what_is_loaded(self, tmp_path):
        built = tensor()
        save_tensor(tmp_path, built)
        read = load_tensor(tmp_path, "test", "jina_v2", "base")
        assert np.array_equal(read.scores, built.scores)
        assert read.query_ids == built.query_ids
        assert read.formats == built.formats
        assert (read.model, read.arm, read.split) == ("jina_v2", "base", "test")

    def test_writes_both_files(self, tmp_path):
        save_tensor(tmp_path, tensor())
        assert tensor_path(tmp_path, "test", "jina_v2", "base").exists()
        assert tensor_meta_path(tmp_path, "test", "jina_v2", "base").exists()

    def test_the_description_records_the_gold_position(self, tmp_path):
        """A reader cannot find the gold row without it."""
        save_tensor(tmp_path, tensor())
        meta = json.loads(
            tensor_meta_path(tmp_path, "test", "jina_v2", "base").read_text()
        )
        assert meta["gold_index"] == GOLD
        assert meta["n_candidates"] == C

    def test_extra_fields_reach_the_description(self, tmp_path):
        save_tensor(tmp_path, tensor(), extra={"adapter": "some/path", "device": "cpu"})
        meta = json.loads(
            tensor_meta_path(tmp_path, "test", "jina_v2", "base").read_text()
        )
        assert meta["adapter"] == "some/path"
        assert meta["device"] == "cpu"


class TestMissing:
    def test_refuses_a_condition_that_was_not_captured(self, tmp_path):
        with pytest.raises(SystemExit, match="no scores for"):
            load_tensor(tmp_path, "test", "jina_v2", "base")

    def test_refuses_scores_with_no_description(self, tmp_path):
        save_tensor(tmp_path, tensor())
        tensor_meta_path(tmp_path, "test", "jina_v2", "base").unlink()
        with pytest.raises(SystemExit, match="no scores for"):
            load_tensor(tmp_path, "test", "jina_v2", "base")
