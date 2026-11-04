"""Tests for fsr.demo.selection."""

from __future__ import annotations

import numpy as np
import pytest

from fsr.demo import selection as mod
from fsr.formats import FORMAT_NAMES
from fsr.standardisation.results import CONTROL, MIXED, STANDARDISED, UNTRAINED, WINNER
from fsr.standardisation.store import ScoreTensor

K = 1
YAML, MARKDOWN = 0, 4


def pool(index):
    return {
        "id": f"p{index}",
        "kind": "river",
        "target": "source",
        "question": f"question {index}",
        "answer": "a",
        "wrong_answer": "b",
        "mixed": ["yaml", "json", "json"],
        "candidates": [{"role": "gold"}, {"role": "wrong"}, {"role": "near_miss"}],
    }


def tensor(scores, arm):
    scores = np.asarray(scores, dtype=np.float32)
    return ScoreTensor(
        scores=scores,
        query_ids=[f"p{i}" for i in range(len(scores))],
        formats=list(FORMAT_NAMES),
        model="bge_base",
        arm=arm,
        split="demo",
    )


def losing():
    """Pool 0 loses its gold except in markdown; pools 1 and 2 make yaml best.

    The wrong passage outscores the gold in every format but markdown.
    """
    lost = [[0.4, 0.4, 0.4, 0.4, 0.9], [0.6] * 5, [0.1] * 5]
    leads_in_yaml = [[0.9, 0.1, 0.1, 0.1, 0.1], [0.5] * 5, [0.1] * 5]
    return [lost, leads_in_yaml, leads_in_yaml]


def keeping():
    """Every pool keeps its gold in every format."""
    return [[[0.9] * 5, [0.5] * 5, [0.1] * 5]] * 3


def tensors(untrained=None, control=None, winner=None):
    return {
        UNTRAINED: tensor(untrained or losing(), "base"),
        CONTROL: tensor(control or losing(), "5fmt_lam0"),
        WINNER: tensor(winner or keeping(), "5fmt_lam1"),
    }


POOLS = [pool(i) for i in range(3)]


class TestRanks:
    def test_ranks_from_one_best_first(self):
        got = mod.ranks(np.array([[0.2, 0.9, 0.5]]))
        assert got.tolist() == [[3, 1, 2]]

    def test_a_tie_goes_to_the_earlier_candidate(self):
        assert mod.ranks(np.array([[0.5, 0.5, 0.1]])).tolist() == [[1, 2, 3]]

    def test_gold_rank_counts_the_negatives_above(self):
        assert mod.gold_rank(np.array([[0.3, 0.9, 0.5], [0.9, 0.1, 0.2]])).tolist() == [
            3,
            1,
        ]


class TestAssignments:
    def test_mixed_indices_follow_the_tensor_columns(self):
        got = mod.mixed_indices([pool(0)], ["json", "yaml"])
        assert got.tolist() == [[1, 0, 0]]

    def test_best_standard_picks_the_highest_mrr(self):
        assert mod.best_standard(tensors()[UNTRAINED]) == "yaml"

    def test_uniform_ranks_cover_every_format(self):
        got = mod.uniform_ranks(tensors()[UNTRAINED])
        assert got.shape == (3, len(FORMAT_NAMES))
        assert got[0].tolist() == [2, 2, 2, 2, 1]

    def test_swap_ranks_move_only_the_gold(self):
        t = tensors()[UNTRAINED]
        assignment = mod.standard_assignment(t, "yaml")
        got = mod.swap_ranks(t, assignment)
        assert got[0, YAML] == 2
        assert got[0, MARKDOWN] == 1


class TestSelect:
    def test_finds_the_pool_the_format_decides(self):
        got = mod.select(tensors(), POOLS, K)
        assert [m["id"] for m in got["matches"]] == ["p0"]

    def test_reports_the_standard_of_each_condition(self):
        got = mod.select(tensors(), POOLS, K)
        assert got["standards"] == {UNTRAINED: "yaml", CONTROL: "yaml", WINNER: "yaml"}

    def test_records_the_ranks_and_the_rescue(self):
        match = mod.select(tensors(), POOLS, K)["matches"][0]
        assert match["ranks"][UNTRAINED][MIXED] == 2
        assert match["ranks"][UNTRAINED][STANDARDISED] == 2
        assert match["ranks"][WINNER]["uniform"]["markdown"] == 1
        assert match["rescued_by"] == {MIXED: ["markdown"], STANDARDISED: ["markdown"]}

    def test_notes_the_wrong_answer_surviving(self):
        match = mod.select(tensors(), POOLS, K)["matches"][0]
        assert match["wrong_in_topk"] is True

    def test_the_funnel_counts_each_requirement_in_turn(self):
        funnel = mod.select(tensors(), POOLS, K)["funnel"]
        assert [step["n"] for step in funnel] == [3, 1, 1, 1, 1, 1, 1, 1]

    def test_names_the_arms(self):
        assert mod.select(tensors(), POOLS, K)["arms"][WINNER] == "5fmt_lam1"


class TestEachRequirementAlone:
    def test_no_match_when_the_untrained_model_keeps_it(self):
        got = mod.select(tensors(untrained=keeping()), POOLS, K)
        assert got["matches"] == []

    def test_no_match_when_the_control_keeps_it(self):
        got = mod.select(tensors(control=keeping()), POOLS, K)
        assert got["matches"] == []

    def test_no_match_when_the_invariant_arm_loses_one_format(self):
        winner = keeping()
        winner[0] = [[0.9, 0.9, 0.4, 0.9, 0.9], [0.5] * 5, [0.1] * 5]
        got = mod.select(tensors(winner=winner), POOLS, K)
        assert got["matches"] == []
        assert got["funnel"][-3]["n"] == 1
        assert got["funnel"][-2]["n"] == 0

    def test_no_match_when_no_format_rescues_the_gold(self):
        untrained = losing()
        untrained[0] = [[0.4] * 5, [0.6] * 5, [0.1] * 5]
        got = mod.select(tensors(untrained=untrained), POOLS, K)
        assert got["matches"] == []
        assert got["funnel"][-2]["n"] == 1
        assert got["funnel"][-1]["n"] == 0


class TestAlignment:
    def test_refuses_tensors_out_of_order(self):
        with pytest.raises(ValueError, match="do not cover the pools"):
            mod.select(tensors(), list(reversed(POOLS)), K)
