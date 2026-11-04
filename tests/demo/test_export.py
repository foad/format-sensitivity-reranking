"""Tests for fsr.demo.export."""

from __future__ import annotations

import numpy as np

from fsr.demo import export as mod
from fsr.demo.selection import mixed_indices
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.standardisation.results import CONTROL, MIXED, UNTRAINED, WINNER
from fsr.standardisation.store import ScoreTensor

K = 1
STANDARDS = {UNTRAINED: "yaml", CONTROL: "yaml", WINNER: "yaml"}


def candidate(role, marker):
    return {
        "role": role,
        "entity": marker,
        "pairs": [["name", marker]],
        "truncated_body": f"{marker} body",
    }


POOL = {
    "id": "p0",
    "question": "Where does it rise?",
    "answer": "Mount Gold",
    "wrong_answer": "Mount Wrong",
    "mixed": ["json", "markdown", "toml"],
    "candidates": [
        candidate("gold", "GOLDTEXT"),
        candidate("wrong", "WRONGTEXT"),
        candidate("near_miss", "NEARTEXT"),
    ],
}


def tensor(gold, arm):
    scores = np.zeros((1, 3, len(FORMAT_NAMES)), dtype=np.float32)
    scores[0, 0] = gold
    scores[0, 1] = 0.5
    scores[0, 2] = 0.1
    return ScoreTensor(
        scores=scores,
        query_ids=["p0"],
        formats=list(FORMAT_NAMES),
        model="bge_base",
        arm=arm,
        split="demo",
    )


TENSORS = {
    UNTRAINED: tensor(0.2, "base"),
    CONTROL: tensor(0.2, "5fmt_lam0"),
    WINNER: tensor(0.9, "5fmt_lam1"),
}


class TestRankedCandidates:
    def test_orders_best_first(self):
        assignment = mixed_indices([POOL], FORMAT_NAMES)
        got = mod.ranked_candidates(TENSORS[UNTRAINED], POOL, 0, assignment)
        assert [c["role"] for c in got] == ["wrong", "gold", "near_miss"]
        assert [c["rank"] for c in got] == [1, 2, 3]

    def test_renders_each_candidate_in_its_format(self):
        assignment = mixed_indices([POOL], FORMAT_NAMES)
        got = mod.ranked_candidates(TENSORS[UNTRAINED], POOL, 0, assignment)
        wrong = got[0]
        assert wrong["format"] == "markdown"
        assert wrong["text"] == FORMATS["markdown"](
            [["name", "WRONGTEXT"]], "WRONGTEXT body"
        )


class TestPipelineViews:
    def test_one_view_per_condition_and_handling(self):
        views = mod.pipeline_views(TENSORS, [POOL], STANDARDS, "p0")
        assert len(views) == 6
        assert {(v["condition"], v["handling"]) for v in views} >= {(WINNER, MIXED)}

    def test_reports_the_gold_rank(self):
        views = mod.pipeline_views(TENSORS, [POOL], STANDARDS, "p0")
        by = {(v["condition"], v["handling"]): v["gold_rank"] for v in views}
        assert by[UNTRAINED, MIXED] == 2
        assert by[WINNER, MIXED] == 1

    def test_names_the_standard_only_when_standardised(self):
        views = mod.pipeline_views(TENSORS, [POOL], STANDARDS, "p0")
        assert {v["standard"] for v in views if v["handling"] == MIXED} == {None}


class TestMarkdown:
    def test_passes_on_the_gold_only_where_it_is_kept(self):
        views = mod.pipeline_views(TENSORS, [POOL], STANDARDS, "p0")
        text = mod.topk_markdown(POOL, views, K)
        sections = text.split("\n## ")[1:]
        assert len(sections) == 6
        for section in sections:
            kept = section.startswith(WINNER)
            assert ("GOLDTEXT" in section) == kept
            assert ("kept at rank 1" in section) == kept

    def test_states_the_expected_answer(self):
        views = mod.pipeline_views(TENSORS, [POOL], STANDARDS, "p0")
        assert "Expected answer: Mount Gold." in mod.topk_markdown(POOL, views, K)
