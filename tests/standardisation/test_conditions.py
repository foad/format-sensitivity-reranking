"""Tests for fsr.standardisation.conditions."""

from __future__ import annotations

import numpy as np
import pytest

from fsr.standardisation.conditions import (
    mixed_assignment,
    realise,
    uniform_assignment,
)

Q, C, F = 3, 4, 5


def tensor():
    """Build scores whose value encodes query, candidate and format."""
    return np.arange(Q * C * F, dtype=float).reshape(Q, C, F)


class TestUniformAssignment:
    def test_covers_every_candidate(self):
        assert uniform_assignment(Q, C, 2).shape == (Q, C)

    def test_gives_every_candidate_the_same_format(self):
        assert (uniform_assignment(Q, C, 2) == 2).all()


class TestMixedAssignment:
    def test_covers_every_candidate(self):
        rng = np.random.default_rng(0)
        assert mixed_assignment(Q, C, F, rng).shape == (Q, C)

    def test_stays_inside_the_format_axis(self):
        rng = np.random.default_rng(0)
        drawn = mixed_assignment(Q, C, F, rng)
        assert drawn.min() >= 0
        assert drawn.max() < F

    def test_candidates_of_one_query_can_differ(self):
        """The mixed corpus gives each candidate its own format."""
        rng = np.random.default_rng(0)
        drawn = mixed_assignment(40, C, F, rng)
        assert any(len(set(row)) > 1 for row in drawn)

    def test_the_seed_fixes_the_draw(self):
        first = mixed_assignment(Q, C, F, np.random.default_rng(7))
        second = mixed_assignment(Q, C, F, np.random.default_rng(7))
        assert np.array_equal(first, second)


class TestRealise:
    def test_returns_one_score_per_query_and_candidate(self):
        got = realise(tensor(), uniform_assignment(Q, C, 0))
        assert got.shape == (Q, C)

    def test_takes_the_format_each_candidate_was_given(self):
        scores = tensor()
        got = realise(scores, uniform_assignment(Q, C, 3))
        assert np.array_equal(got, scores[:, :, 3])

    def test_a_mixed_assignment_picks_per_candidate(self):
        scores = tensor()
        assignment = np.zeros((Q, C), dtype=int)
        assignment[0, 1] = 4
        got = realise(scores, assignment)
        assert got[0, 1] == scores[0, 1, 4]
        assert got[0, 0] == scores[0, 0, 0]

    def test_refuses_an_assignment_of_the_wrong_shape(self):
        with pytest.raises(ValueError, match="expected"):
            realise(tensor(), np.zeros((Q, C + 1), dtype=int))
