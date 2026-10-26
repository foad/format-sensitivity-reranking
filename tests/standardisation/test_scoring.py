"""Tests for fsr.standardisation.scoring."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr.formats import FORMAT_NAMES
from fsr.standardisation import scoring as mod
from fsr.standardisation.layout import candidates_path

Q, C = 3, 4


def prepared_queries(n_queries=Q, n_candidates=C):
    """Build candidate lists shaped like the ones the answer axis writes."""
    return [
        {
            "id": f"q{q}",
            "question": f"question {q}",
            "candidates": [
                {
                    "pairs": [["title", f"t{q}{c}"]],
                    "truncated_body": f"body {q} {c}",
                }
                for c in range(n_candidates)
            ],
        }
        for q in range(n_queries)
    ]


def write_candidates(root, split="test", limit=0, prepared=None):
    path = candidates_path(root, split, limit)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"prepared": prepared_queries() if prepared is None else prepared})
    )


class TestLoadCandidates:
    def test_reads_the_lists(self, tmp_path):
        write_candidates(tmp_path)
        assert len(mod.load_candidates(tmp_path, "test")) == Q

    def test_names_the_command_that_builds_them(self, tmp_path):
        """The study cannot build the lists, so it has to say what does."""
        with pytest.raises(SystemExit, match=r"within_query\.py"):
            mod.load_candidates(tmp_path, "test")

    def test_a_capped_build_has_its_own_lists(self, tmp_path):
        write_candidates(tmp_path, limit=2)
        assert len(mod.load_candidates(tmp_path, "test", limit=2)) == Q


class TestQueryIds:
    def test_returns_one_id_per_query_in_order(self):
        assert mod.query_ids(prepared_queries()) == ["q0", "q1", "q2"]


class TestScoreEveryFormat:
    def test_returns_one_score_per_query_candidate_and_format(self, monkeypatch):
        monkeypatch.setattr(mod, "score_batch", lambda *_a, **_k: [0.0] * (Q * C))
        scores = mod.score_every_format(None, None, prepared_queries(), 8, "cpu")
        assert scores.shape == (Q, C, len(FORMAT_NAMES))

    def test_each_format_lands_in_its_own_column(self, monkeypatch):
        """A transposed reshape would silently mix the formats."""
        calls = {"n": 0}

        def fake(_m, _t, pairs, *_a, **_k):
            calls["n"] += 1
            return [float(calls["n"])] * len(pairs)

        monkeypatch.setattr(mod, "score_batch", fake)
        scores = mod.score_every_format(None, None, prepared_queries(), 8, "cpu")
        for index in range(len(FORMAT_NAMES)):
            assert np.all(scores[:, :, index] == index + 1)

    def test_the_rows_follow_the_query_order(self, monkeypatch):
        monkeypatch.setattr(
            mod, "score_batch", lambda _m, _t, pairs, *_a, **_k: list(range(len(pairs)))
        )
        scores = mod.score_every_format(None, None, prepared_queries(), 8, "cpu")
        assert np.array_equal(scores[:, :, 0], np.arange(Q * C).reshape(Q, C))

    def test_renders_the_body_with_the_metadata(self, monkeypatch):
        seen = []

        def fake(_m, _t, pairs, *_a, **_k):
            seen.append(pairs[0][1])
            return [0.0] * len(pairs)

        monkeypatch.setattr(mod, "score_batch", fake)
        mod.score_every_format(None, None, prepared_queries(), 8, "cpu")
        assert all("body 0 0" in text for text in seen)

    def test_reports_each_format_as_it_completes(self, monkeypatch):
        monkeypatch.setattr(mod, "score_batch", lambda *_a, **_k: [0.0] * (Q * C))
        seen = []
        mod.score_every_format(
            None, None, prepared_queries(), 8, "cpu", on_format=seen.append
        )
        assert seen == list(FORMAT_NAMES)

    def test_refuses_an_empty_candidate_set(self, monkeypatch):
        monkeypatch.setattr(mod, "score_batch", lambda *_a, **_k: [])
        with pytest.raises(ValueError, match="no query to score"):
            mod.score_every_format(None, None, [], 8, "cpu")

    def test_refuses_queries_that_carry_no_candidate(self, monkeypatch):
        monkeypatch.setattr(mod, "score_batch", lambda *_a, **_k: [])
        with pytest.raises(ValueError, match="no query to score"):
            mod.score_every_format(
                None, None, prepared_queries(n_candidates=0), 8, "cpu"
            )
