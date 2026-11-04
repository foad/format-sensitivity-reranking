"""Tests for fsr.demo.scenarios."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from fsr.demo import scenarios as mod
from fsr.formats import FORMAT_NAMES
from tests.fakes import WordTokenizer

N_COMBOS = sum(len(kind.questions) for kind in mod.KINDS) * len(mod.KEY_STYLES)


@pytest.fixture(scope="module")
def pools():
    return mod.build_scenarios(N_COMBOS * 2, seed=3)


def metadata_text(candidate):
    return " ".join(value for _, value in candidate["pairs"]).lower()


class TestBuild:
    def test_draws_the_number_asked(self, pools):
        assert len(pools) == N_COMBOS * 2

    def test_the_seed_fixes_the_draw(self):
        assert mod.build_scenarios(5, seed=1) == mod.build_scenarios(5, seed=1)

    def test_another_seed_draws_other_pools(self):
        assert mod.build_scenarios(5, seed=1) != mod.build_scenarios(5, seed=2)

    def test_ids_are_unique(self, pools):
        assert len({p["id"] for p in pools}) == len(pools)

    def test_covers_every_kind_target_and_key_style(self, pools):
        seen = {(p["kind"], p["target"], p["key_style"]) for p in pools}
        assert len(seen) == N_COMBOS

    def test_every_pool_holds_the_roles_in_order(self, pools):
        for pool in pools:
            assert tuple(c["role"] for c in pool["candidates"]) == mod.ROLES

    def test_the_mixed_assignment_names_a_format_per_candidate(self, pools):
        for pool in pools:
            assert len(pool["mixed"]) == mod.N_CANDIDATES
            assert set(pool["mixed"]) <= set(FORMAT_NAMES)

    def test_no_pool_leaks(self, pools):
        assert all(not mod.leaks(pool) for pool in pools)


class TestAnswerPlacement:
    """The answer must be unreachable once the gold passage is gone."""

    def test_the_gold_metadata_carries_the_answer(self, pools):
        for pool in pools:
            assert pool["answer"].lower() in metadata_text(pool["candidates"][0])

    def test_the_gold_prose_does_not(self, pools):
        for pool in pools:
            body = pool["candidates"][0]["truncated_body"].lower()
            assert pool["answer"].lower() not in body

    def test_no_other_candidate_carries_the_answer(self, pools):
        for pool in pools:
            for candidate in pool["candidates"][1:]:
                text = mod.passage_text(candidate).lower()
                assert pool["answer"].lower() not in text

    def test_only_the_wrong_passage_carries_the_wrong_answer(self, pools):
        for pool in pools:
            for candidate in pool["candidates"]:
                carries = (
                    pool["wrong_answer"].lower() in mod.passage_text(candidate).lower()
                )
                assert carries == (candidate["role"] == mod.WRONG_ROLE)

    def test_near_misses_describe_the_subject(self, pools):
        for pool in pools:
            subject = pool["candidates"][0]["entity"]
            near = [c for c in pool["candidates"] if c["role"] == mod.NEAR_ROLE]
            assert all(c["entity"] == subject for c in near)

    def test_the_question_names_the_subject(self, pools):
        for pool in pools:
            assert pool["candidates"][0]["entity"] in pool["question"]


class TestLeaks:
    def test_flags_an_answer_in_a_distractor(self, pools):
        pool = copy.deepcopy(pools[0])
        pool["candidates"][3]["truncated_body"] += " " + pool["answer"]
        assert any("carries the answer" in p for p in mod.leaks(pool))

    def test_flags_an_answer_in_the_gold_prose(self, pools):
        pool = copy.deepcopy(pools[0])
        pool["candidates"][0]["truncated_body"] += " " + pool["answer"]
        assert "the gold prose states the answer" in mod.leaks(pool)

    def test_flags_a_wrong_answer_outside_its_passage(self, pools):
        pool = copy.deepcopy(pools[0])
        pool["candidates"][4]["truncated_body"] += " " + pool["wrong_answer"]
        assert any("carries the wrong answer" in p for p in mod.leaks(pool))

    def test_flags_overlapping_answers(self, pools):
        pool = copy.deepcopy(pools[0])
        pool["wrong_answer"] = pool["answer"] + " Junior"
        assert "the answer and the wrong answer overlap" in mod.leaks(pool)


class TestKeyStyles:
    @pytest.mark.parametrize(
        ("style", "expected"),
        [("snake", "launch_date"), ("title", "Launch date"), ("camel", "launchDate")],
    )
    def test_renames_a_key(self, style, expected):
        assert mod.KEY_STYLES[style]("launch_date") == expected

    def test_a_pool_uses_one_style_throughout(self, pools):
        for pool in pools:
            style = mod.KEY_STYLES[pool["key_style"]]
            for candidate in pool["candidates"]:
                assert candidate["pairs"][0][0] == style(mod.NAME_KEY)


class TestOverlong:
    def test_passes_short_pools(self, pools):
        assert mod.overlong(pools[:3], WordTokenizer(), max_tokens=512) == []

    def test_reports_each_long_pair(self, pools):
        found = mod.overlong(pools[:1], WordTokenizer(), max_tokens=1)
        assert len(found) == mod.N_CANDIDATES * len(FORMAT_NAMES)
        assert found[0] == (pools[0]["id"], 0, FORMAT_NAMES[0])


class TestRoleIndex:
    def test_finds_the_wrong_passage(self, pools):
        assert mod.role_index(pools[0], mod.WRONG_ROLE) == 1


class TestDraws:
    def test_pick_stays_inside_the_options(self):
        rng = np.random.default_rng(0)
        assert {mod.pick(rng, "ab") for _ in range(50)} == {"a", "b"}

    def test_a_sibling_name_differs_from_its_source(self):
        rng = np.random.default_rng(0)
        for kind in mod.KINDS:
            name = kind.draw_name(rng)
            assert kind.sibling(name, rng) != name
