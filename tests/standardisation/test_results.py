"""Tests for fsr.standardisation.results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr.h2_layout import selection_path
from fsr.models.registry import by_slug
from fsr.standardisation import results as mod
from fsr.standardisation.layout import tensor_path
from fsr.standardisation.store import ScoreTensor, save_tensor

SLUGS = ["minilm_l6", "bge_base"]
FORMATS = ["yaml", "json", "toml", "inline_kv", "markdown"]
Q, C, F = 20, 4, len(FORMATS)
BOOT, PERM = 200, 400


def selection(winner="1"):
    return {"status": "selected", "winner_lambda": winner, "candidates": []}


def write_tensor(root, slug, arm_name, gold=2.0, bump=None, slope=0.0):
    """Write a tensor whose gold leads, with optional per-format noise."""
    scores = np.zeros((Q, C, F), dtype=np.float32)
    scores[:, 0, :] = gold + slope * np.arange(Q)[:, None]
    if bump is not None:
        index, value = bump
        scores[:, 1, index] = value
    save_tensor(
        root,
        ScoreTensor(
            scores=scores,
            query_ids=[f"q{i}" for i in range(Q)],
            formats=list(FORMATS),
            model=slug,
            arm=arm_name,
            split="test",
        ),
    )


@pytest.fixture
def root(tmp_path, request):
    winner = getattr(request, "param", "1")
    for n, slug in enumerate(SLUGS):
        path = selection_path(tmp_path, slug, "lambda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection(winner)))
        for name in mod.arm_names(tmp_path, slug).values():
            write_tensor(tmp_path, slug, name, slope=0.1 * (n + 1))
    return tmp_path


def small(fn, *a, **k):
    return fn(*a, n_boot=BOOT, **k)


class TestArmNames:
    def test_names_the_three_conditions(self, root):
        assert mod.arm_names(root, "minilm_l6") == {
            mod.UNTRAINED: "base",
            mod.CONTROL: "5fmt_lam0",
            mod.WINNER: "5fmt_lam1",
        }

    @pytest.mark.parametrize("root", ["0"], indirect=True)
    def test_a_winner_of_zero_is_the_control(self, root):
        assert mod.WINNER not in mod.arm_names(root, "minilm_l6")


class TestAvailable:
    def test_finds_a_model_whose_conditions_are_captured(self, root):
        assert mod.available(root) == SLUGS

    def test_reports_nothing_without_a_capture(self, tmp_path):
        assert mod.available(tmp_path) == []

    def test_skips_a_model_missing_one_condition(self, root):
        tensor_path(root, "test", "bge_base", "5fmt_lam1").unlink()
        assert mod.available(root) == ["minilm_l6"]

    def test_skips_a_model_with_no_selection(self, root):
        selection_path(root, "bge_base", "lambda").unlink()
        assert mod.available(root) == ["minilm_l6"]


class TestHardSubset:
    def test_holds_the_named_share_of_queries(self, root):
        picked = mod.hard_subset(root, SLUGS, quantile=0.25)
        assert {len(v) for v in picked.values()} == {Q // 4}

    def test_covers_every_model(self, root):
        assert set(mod.hard_subset(root, SLUGS)) == set(SLUGS)

    def test_gives_every_model_the_same_queries(self, root):
        """One population keeps the rows of a table comparable."""
        write_tensor(root, "minilm_l6", "base", slope=-5.0)
        picked = mod.hard_subset(root, SLUGS)
        first = picked[SLUGS[0]]
        assert all(np.array_equal(first, picked[slug]) for slug in SLUGS)

    def test_every_model_moves_the_partition(self, root):
        """A model on the panel counts towards the queries it selects."""
        before = mod.hard_subset(root, SLUGS)["bge_base"]
        write_tensor(root, "minilm_l6", "base", slope=-5.0)
        assert not np.array_equal(before, mod.hard_subset(root, SLUGS)["bge_base"])

    def test_refuses_a_single_model(self, root):
        with pytest.raises(ValueError, match="at least two models"):
            mod.hard_subset(root, ["minilm_l6"])


class TestHolm:
    def test_leaves_one_value_alone(self):
        assert mod.holm([0.04]) == pytest.approx([0.04])

    def test_multiplies_the_smallest_by_the_count(self):
        assert mod.holm([0.01, 0.5])[0] == pytest.approx(0.02)

    def test_never_decreases(self):
        adjusted = mod.holm([0.01, 0.02, 0.03])
        assert list(adjusted) == sorted(adjusted)

    def test_caps_at_one(self):
        assert (mod.holm([0.6, 0.7, 0.8]) <= 1.0).all()


class TestLevelsTable:
    def test_holds_one_row_per_model_and_condition(self, root):
        table = mod.levels_table(root, SLUGS)
        assert len(table) == len(SLUGS) * 3
        assert table.index.names == ["model", "condition"]

    def test_reports_every_policy(self, root):
        table = mod.levels_table(root, SLUGS)
        assert set(table.columns) == {"standardised_mean", "standardised_best", "mixed"}

    def test_a_gold_that_always_leads_scores_one(self, root):
        assert (mod.levels_table(root, SLUGS) == 1.0).all().all()

    def test_the_subset_changes_what_is_measured(self, root):
        write_tensor(root, "minilm_l6", "base", bump=(0, 9.0))
        full = mod.levels_table(root, ["minilm_l6"])
        part = mod.levels_table(root, ["minilm_l6"], subset={"minilm_l6": np.arange(2)})
        assert full.shape == part.shape


class TestContrastTable:
    def test_holds_one_row_per_model(self, root):
        table = small(
            mod.contrast_table,
            root,
            SLUGS,
            (mod.WINNER, "mixed"),
            (mod.UNTRAINED, "standardised_mean"),
        )
        assert list(table.index) == [by_slug(s).label for s in SLUGS]

    def test_identical_arms_report_no_difference(self, root):
        table = small(
            mod.contrast_table,
            root,
            SLUGS,
            (mod.WINNER, "mixed"),
            (mod.UNTRAINED, "mixed"),
        )
        assert (table["diff"] == 0.0).all()
        assert not table["resolved"].any()

    def test_a_real_difference_resolves(self, root):
        write_tensor(root, "minilm_l6", "base", bump=(0, 9.0))
        table = small(
            mod.contrast_table,
            root,
            ["minilm_l6", "bge_base"],
            (mod.WINNER, "mixed"),
            (mod.UNTRAINED, "mixed"),
        )
        row = table.loc[by_slug("minilm_l6").label]
        assert row["diff"] > 0
        assert row["ci_lo"] > 0

    def test_carries_the_corrected_p_value(self, root):
        table = small(
            mod.contrast_table,
            root,
            SLUGS,
            (mod.WINNER, "mixed"),
            (mod.UNTRAINED, "standardised_mean"),
        )
        assert (table["holm_p"] >= table["p"]).all()

    def test_reports_the_floor_of_the_test(self, root):
        """A permutation test cannot report a p below one over its count."""
        table = small(
            mod.contrast_table,
            root,
            SLUGS,
            (mod.WINNER, "mixed"),
            (mod.UNTRAINED, "standardised_mean"),
        )
        assert (table["p"] >= table["p_floor"]).all()


class TestAgreementTable:
    def test_holds_one_row_per_model(self, root):
        assert len(small(mod.agreement_table, root, SLUGS)) == len(SLUGS)

    def test_agreeing_formats_report_no_disagreement(self, root):
        table = small(mod.agreement_table, root, SLUGS)
        assert (table["disagree_untrained"] == 0.0).all()

    def test_one_disagreeing_format_is_counted(self, root):
        write_tensor(root, "minilm_l6", "base", bump=(0, 9.0))
        table = small(mod.agreement_table, root, SLUGS)
        assert table.loc[by_slug("minilm_l6").label, "disagree_untrained"] > 0

    @pytest.mark.parametrize("root", ["0"], indirect=True)
    def test_refuses_a_model_with_no_winner(self, root):
        with pytest.raises(ValueError, match="no 'invariance' condition"):
            small(mod.agreement_table, root, SLUGS)


class TestFormatTable:
    def test_holds_one_column_per_format_and_the_mixed_corpus(self, root):
        grid = mod.format_table(root, SLUGS)
        assert list(grid.columns) == [*FORMATS, "mixed"]

    def test_one_bad_format_shows_only_in_its_column(self, root):
        write_tensor(root, "minilm_l6", "base", bump=(0, 9.0))
        row = mod.format_table(root, ["minilm_l6"]).loc[by_slug("minilm_l6").label]
        assert row["yaml"] == 0.0
        assert row["json"] == 1.0

    @pytest.mark.parametrize("root", ["0"], indirect=True)
    def test_refuses_a_condition_the_model_does_not_carry(self, root):
        with pytest.raises(ValueError, match="no 'invariance' condition"):
            mod.format_table(root, SLUGS, condition=mod.WINNER)


class TestGuardrailTable:
    def test_holds_one_row_per_model_and_condition(self, root):
        assert len(mod.guardrail_table(root, SLUGS)) == len(SLUGS) * 3

    def test_an_unchanged_ranking_keeps_its_quality(self, root):
        assert mod.guardrail_table(root, SLUGS)["kept"].all()

    def test_carries_the_margin_the_main_study_uses(self, root):
        from fsr.comparison import NI_MARGIN

        assert (mod.guardrail_table(root, SLUGS)["ni_margin"] == NI_MARGIN).all()


class TestInconsistencyAndOracle:
    def test_inconsistency_holds_one_row_per_model_and_condition(self, root):
        assert len(mod.inconsistency_table(root, SLUGS)) == len(SLUGS) * 3

    def test_agreeing_formats_report_no_inconsistency(self, root):
        assert (mod.inconsistency_table(root, SLUGS)["n_inconsistent"] == 0).all()

    def test_oracle_reports_the_band(self, root):
        table = mod.oracle_table(root, SLUGS)
        assert set(table.columns) >= {"floor", "best_format", "ceiling", "headroom"}

    def test_a_unanimous_lead_leaves_no_headroom(self, root):
        assert (mod.oracle_table(root, SLUGS)["headroom"] == 0.0).all()
