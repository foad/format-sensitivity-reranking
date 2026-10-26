"""Tests for fsr.standardisation.metrics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from fsr.standardisation import metrics as mod
from fsr.standardisation.store import ScoreTensor

FORMATS = ["yaml", "json", "toml", "inline_kv", "markdown"]
Q, C, F = 6, 4, len(FORMATS)


def tensor(scores):
    return ScoreTensor(
        scores=np.asarray(scores, dtype=np.float32),
        query_ids=[f"q{i}" for i in range(len(scores))],
        formats=list(FORMATS),
        model="jina_v2",
        arm="base",
        split="test",
    )


def leading(gold=1.0, negative=0.0):
    """Build a tensor where the gold beats every negative in every format."""
    scores = np.full((Q, C, F), negative)
    scores[:, 0, :] = gold
    return tensor(scores)


class TestQuality:
    def test_a_gold_that_leads_everywhere_scores_one(self):
        realised = np.array([[2.0, 1.0, 0.0], [2.0, 1.0, 0.0]])
        assert mod.quality(realised) == {"mrr": 1.0, "top1": 1.0}

    def test_a_gold_behind_every_negative_takes_the_last_rank(self):
        realised = np.array([[0.0, 1.0, 1.0]])
        assert mod.quality(realised)["mrr"] == pytest.approx(1 / 3)
        assert mod.quality(realised)["top1"] == 0.0

    def test_a_tie_counts_for_the_gold(self):
        """The ranking metric of the main study resolves ties this way."""
        realised = np.array([[1.0, 1.0, 1.0]])
        assert mod.quality(realised)["top1"] == 1.0


class TestUniformTable:
    def test_holds_one_row_per_format(self):
        table = mod.uniform_table(leading())
        assert list(table.index) == FORMATS

    def test_measures_each_format_on_its_own_column(self):
        scores = np.zeros((Q, C, F))
        scores[:, 0, :] = 1.0
        scores[:, 1, 2] = 5.0
        table = mod.uniform_table(tensor(scores))
        assert table.loc["toml", "top1"] == 0.0
        assert table.loc["yaml", "top1"] == 1.0


class TestMixedLead:
    def test_returns_one_probability_per_query(self):
        assert mod.mixed_lead(leading()).shape == (Q,)

    def test_a_gold_that_leads_in_every_format_is_certain(self):
        assert mod.mixed_lead(leading()) == pytest.approx(1.0)

    def test_a_gold_that_never_leads_is_impossible(self):
        assert mod.mixed_lead(leading(gold=0.0, negative=1.0)) == pytest.approx(0.0)

    def test_matches_an_enumeration_of_every_assignment(self):
        """The closed form must equal brute force on a case small enough."""
        rng = np.random.default_rng(3)
        scores = rng.normal(size=(2, 3, 2)).astype(np.float32)
        small = ScoreTensor(
            scores=scores,
            query_ids=["a", "b"],
            formats=["yaml", "json"],
            model="jina_v2",
            arm="base",
            split="test",
        )
        brute = []
        for q in range(2):
            hits = 0
            for assignment in itertools.product(range(2), repeat=3):
                picked = np.array([scores[q, c, f] for c, f in enumerate(assignment)])
                hits += picked[0] >= picked[1:].max()
            brute.append(hits / 2**3)
        assert mod.mixed_lead(small) == pytest.approx(brute)

    def test_one_negative_that_wins_in_half_the_formats_halves_the_chance(self):
        scores = np.zeros((1, 2, 2))
        scores[0, 0, :] = 1.0
        scores[0, 1, 0] = 5.0
        tensor = ScoreTensor(
            scores=scores.astype(np.float32),
            query_ids=["a"],
            formats=["yaml", "json"],
            model="jina_v2",
            arm="base",
            split="test",
        )
        assert mod.mixed_lead(tensor) == pytest.approx(0.5)


class TestUniformLeads:
    def test_holds_one_column_per_format(self):
        assert mod.uniform_leads(leading()).shape == (Q, F)

    def test_marks_a_gold_that_leads(self):
        assert mod.uniform_leads(leading()).all()

    def test_marks_a_gold_that_does_not_lead(self):
        assert not mod.uniform_leads(leading(gold=0.0, negative=1.0)).any()


class TestLeadsOf:
    def test_a_tie_counts_for_the_gold(self):
        assert mod.leads_of(np.array([[1.0, 1.0, 0.0]])).all()

    def test_a_loss_does_not(self):
        assert not mod.leads_of(np.array([[0.0, 1.0]])).any()


class TestMargin:
    def test_is_the_gold_lead_over_the_best_negative(self):
        assert mod.margin(leading(gold=3.0, negative=1.0)) == pytest.approx(2.0)

    def test_is_negative_when_the_gold_trails(self):
        assert (mod.margin(leading(gold=0.0, negative=1.0)) < 0).all()


class TestInconsistency:
    def test_counts_nothing_when_every_format_agrees(self):
        got = mod.inconsistency(np.ones((Q, F), dtype=bool))
        assert got["n_inconsistent"] == 0
        assert got["inconsistency_pct"] == 0.0

    def test_counts_a_query_that_leads_under_some_formats_only(self):
        leads = np.ones((Q, F), dtype=bool)
        leads[0, 0] = False
        got = mod.inconsistency(leads)
        assert got["n_answerable"] == Q
        assert got["n_inconsistent"] == 1

    def test_a_query_that_never_leads_is_outside_the_pool(self):
        """The pool is the queries any format answers, as H2 defines it."""
        leads = np.ones((Q, F), dtype=bool)
        leads[0, :] = False
        got = mod.inconsistency(leads)
        assert got["n_answerable"] == Q - 1
        assert got["n_inconsistent"] == 0

    def test_an_empty_pool_leaves_the_share_undefined(self):
        got = mod.inconsistency(np.zeros((Q, F), dtype=bool))
        assert got["n_answerable"] == 0
        assert np.isnan(got["inconsistency_pct"])

    def test_reports_plain_numbers(self):
        """The tables and the notebook round and serialise these."""
        got = mod.inconsistency(np.ones((Q, F), dtype=bool))
        assert {type(v) for v in got.values()} == {int, float}


class TestOracleBand:
    def test_a_unanimous_lead_closes_the_band(self):
        got = mod.oracle_band(np.ones((Q, F), dtype=bool))
        assert got["floor"] == 1.0
        assert got["ceiling"] == 1.0
        assert got["headroom"] == 0.0

    def test_the_ceiling_counts_a_query_any_format_answers(self):
        leads = np.zeros((Q, F), dtype=bool)
        leads[0, 0] = True
        got = mod.oracle_band(leads)
        assert got["ceiling"] == pytest.approx(1 / Q)
        assert got["floor"] == 0.0

    def test_the_headroom_is_the_ceiling_over_the_best_single_format(self):
        leads = np.zeros((Q, F), dtype=bool)
        leads[0, 0] = True
        leads[1, 1] = True
        got = mod.oracle_band(leads)
        assert got["best_format"] == pytest.approx(1 / Q)
        assert got["ceiling"] == pytest.approx(2 / Q)
        assert got["headroom"] == pytest.approx(1 / Q)
