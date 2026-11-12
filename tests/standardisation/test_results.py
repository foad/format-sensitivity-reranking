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


@pytest.fixture
def specialist(root):
    """Capture the single-format arms of one model."""
    for name in mod.specialist_arms().values():
        write_tensor(root, "bge_base", name, slope=0.2)
    return root


class TestSpecialistArms:
    def test_names_one_arm_per_format(self):
        assert mod.specialist_arms() == {f: f"only_{f}_lam0" for f in FORMATS}

    def test_finds_a_model_with_every_capture(self, specialist):
        assert mod.specialist_available(specialist, SLUGS) == ["bge_base"]

    def test_skips_a_model_missing_one_capture(self, specialist):
        tensor_path(specialist, "test", "bge_base", "only_toml_lam0").unlink()
        assert mod.specialist_available(specialist, SLUGS) == []

    def test_leaves_the_main_study_alone(self, specialist):
        """Only one model carries the arms, so the others must stay available."""
        assert mod.available(specialist) == SLUGS


class TestStandardisedLeads:
    def test_holds_one_column_per_format(self, specialist):
        leads = mod.standardised_leads(specialist, "bge_base", mod.SPECIALIST)
        assert leads.shape == (Q, F)

    def test_takes_each_column_from_its_own_arm(self, specialist):
        write_tensor(specialist, "bge_base", "only_json_lam0", bump=(1, 9.0))
        leads = mod.standardised_leads(specialist, "bge_base", mod.SPECIALIST)
        assert not leads[:, 1].any()
        assert leads[:, [0, 2, 3, 4]].all()

    def test_ignores_an_arm_outside_its_own_format(self, specialist):
        """The json arm scored under yaml is the off-diagonal, never reported."""
        write_tensor(specialist, "bge_base", "only_json_lam0", bump=(0, 9.0))
        assert mod.standardised_leads(specialist, "bge_base", mod.SPECIALIST).all()

    def test_matches_the_uniform_leads_of_a_main_condition(self, specialist):
        write_tensor(specialist, "bge_base", "base", bump=(3, 9.0))
        leads = mod.standardised_leads(specialist, "bge_base", mod.UNTRAINED)
        assert not leads[:, 3].any()

    def test_refuses_a_capture_over_other_queries(self, specialist):
        scores = np.zeros((Q, C, F), dtype=np.float32)
        save_tensor(
            specialist,
            ScoreTensor(
                scores=scores,
                query_ids=[f"other{i}" for i in range(Q)],
                formats=list(FORMATS),
                model="bge_base",
                arm="only_yaml_lam0",
                split="test",
            ),
        )
        with pytest.raises(ValueError, match="other queries"):
            mod.standardised_leads(specialist, "bge_base", mod.SPECIALIST)


class TestStandardisedTable:
    def test_holds_one_row_per_format_and_one_column_per_condition(self, specialist):
        table = mod.standardised_table(specialist, "bge_base")
        assert list(table.index) == FORMATS
        assert list(table.columns) == [mod.UNTRAINED, mod.SPECIALIST]

    def test_one_bad_arm_shows_only_in_its_cell(self, specialist):
        write_tensor(specialist, "bge_base", "only_toml_lam0", bump=(2, 9.0))
        table = mod.standardised_table(specialist, "bge_base")
        assert table.loc["toml", mod.SPECIALIST] == 0.0
        assert table.loc["toml", mod.UNTRAINED] == 1.0

    def test_measures_the_subset(self, specialist):
        write_tensor(specialist, "bge_base", "base", bump=(0, 9.0))
        subset = {"bge_base": np.arange(5)}
        table = mod.standardised_table(specialist, "bge_base", subset=subset)
        assert table.loc["yaml", mod.UNTRAINED] == 0.0


class TestFormatContrastTable:
    def contrast(self, root, treatment, reference, **k):
        return small(
            mod.format_contrast_table, root, "bge_base", treatment, reference, **k
        )

    def test_holds_one_row_per_format(self, specialist):
        table = self.contrast(
            specialist, (mod.SPECIALIST, mod.STANDARDISED), (mod.CONTROL, mod.MIXED)
        )
        assert list(table.index) == FORMATS

    def test_alike_sides_show_no_change(self, specialist):
        table = self.contrast(
            specialist, (mod.SPECIALIST, mod.STANDARDISED), (mod.CONTROL, mod.MIXED)
        )
        assert (table["diff"] == 0.0).all()
        assert not table["resolved"].any()

    def test_a_worse_arm_shows_only_in_its_row(self, specialist):
        write_tensor(specialist, "bge_base", "only_toml_lam0", bump=(2, 9.0))
        table = self.contrast(
            specialist,
            (mod.SPECIALIST, mod.STANDARDISED),
            (mod.UNTRAINED, mod.STANDARDISED),
        )
        assert table.loc["toml", "diff"] == -1.0
        assert table.loc["toml", "resolved"]
        assert (table.drop("toml")["diff"] == 0.0).all()

    def test_corrects_for_the_formats_tested(self, specialist):
        write_tensor(specialist, "bge_base", "only_toml_lam0", bump=(2, 9.0))
        table = self.contrast(
            specialist,
            (mod.SPECIALIST, mod.STANDARDISED),
            (mod.UNTRAINED, mod.STANDARDISED),
        )
        assert (table["holm_p"] >= table["p"]).all()

    def test_refuses_two_mixed_sides(self, specialist):
        with pytest.raises(ValueError, match="one side must be standardised"):
            self.contrast(specialist, (mod.WINNER, mod.MIXED), (mod.CONTROL, mod.MIXED))

    def test_refuses_a_mixed_single_format_arm(self, specialist):
        """Mixing across the single-format arms would mix models, not formats."""
        with pytest.raises(ValueError, match="no 'mixed' outcome"):
            self.contrast(
                specialist,
                (mod.SPECIALIST, mod.MIXED),
                (mod.UNTRAINED, mod.STANDARDISED),
            )


class TestComparisonPointsTable:
    def test_holds_the_four_comparison_points(self, specialist):
        table = mod.comparison_points_table(specialist, "bge_base")
        assert list(table.columns) == [
            "standardised, untrained",
            "standardised, standardised training",
            "mixed, rank only",
            "mixed, invariance",
        ]
        assert list(table.index) == FORMATS

    def test_a_mixed_column_repeats_down_the_formats(self, specialist):
        write_tensor(specialist, "bge_base", "5fmt_lam0", bump=(0, 9.0))
        column = mod.comparison_points_table(specialist, "bge_base")["mixed, rank only"]
        assert column.nunique() == 1
        assert 0.0 < column.iloc[0] < 1.0

    @pytest.mark.parametrize("root", ["0"], indirect=True)
    def test_omits_the_winner_when_it_is_the_control(self, root):
        for name in mod.specialist_arms().values():
            write_tensor(root, "bge_base", name)
        columns = mod.comparison_points_table(root, "bge_base").columns
        assert "mixed, invariance" not in columns


def write_seed(root, seed, slug="bge_base", **kwargs):
    """Capture every trained arm of one model under one training seed."""
    for name in mod.specialist_arms(seed).values():
        write_tensor(root, slug, name, **kwargs)
    for condition, name in mod.arm_names(root, slug, seed).items():
        if condition != mod.UNTRAINED:
            write_tensor(root, slug, name, **kwargs)


class TestTrainingSeeds:
    def test_names_the_arms_of_a_seed(self, root):
        assert mod.specialist_arms(2)["json"] == "only_json_lam0_s2"
        assert mod.arm_names(root, "bge_base", 2)[mod.CONTROL] == "5fmt_lam0_s2"

    def test_the_untrained_arm_carries_no_seed(self, root):
        assert mod.arm_names(root, "bge_base", 2)[mod.UNTRAINED] == "base"

    def test_finds_the_default_seed(self, specialist):
        assert mod.training_seeds(specialist, "bge_base") == [0]

    def test_finds_every_complete_seed(self, specialist):
        write_seed(specialist, 1)
        write_seed(specialist, 3)
        assert mod.training_seeds(specialist, "bge_base") == [0, 1, 3]

    def test_skips_a_seed_missing_one_arm(self, specialist):
        write_seed(specialist, 1)
        tensor_path(specialist, "test", "bge_base", "5fmt_lam1_s1").unlink()
        assert mod.training_seeds(specialist, "bge_base") == [0]

    def test_skips_the_default_seed_without_its_single_format_arms(self, root):
        write_seed(root, 1)
        assert mod.training_seeds(root, "bge_base") == [1]


class TestSeedAveraging:
    def test_averages_each_query_across_the_seeds(self, specialist):
        write_seed(specialist, 1)
        write_tensor(specialist, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        table = mod.standardised_table(specialist, "bge_base", seeds=(0, 1))
        assert table.loc["toml", mod.SPECIALIST] == 0.5
        assert table.loc["json", mod.SPECIALIST] == 1.0

    def test_leaves_the_untrained_arm_alone(self, specialist):
        write_seed(specialist, 1, bump=(0, 9.0))
        table = mod.standardised_table(specialist, "bge_base", seeds=(0, 1))
        assert table.loc["yaml", mod.UNTRAINED] == 1.0

    def test_the_contrast_tests_the_seed_average(self, specialist):
        write_seed(specialist, 1)
        write_tensor(specialist, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        table = small(
            mod.format_contrast_table,
            specialist,
            "bge_base",
            (mod.SPECIALIST, mod.STANDARDISED),
            (mod.UNTRAINED, mod.STANDARDISED),
            seeds=(0, 1),
        )
        assert table.loc["toml", "diff"] == -0.5

    def test_the_mixed_side_follows_the_seeds(self, specialist):
        write_seed(specialist, 1)
        write_tensor(specialist, "bge_base", "5fmt_lam0_s1", bump=(0, 9.0))
        points = mod.comparison_points_table(specialist, "bge_base", seeds=(0, 1))
        assert 0.5 < points["mixed, rank only"].iloc[0] < 1.0

    def test_refuses_no_seed(self, specialist):
        with pytest.raises(ValueError, match="at least one training seed"):
            mod.standardised_table(specialist, "bge_base", seeds=())


class TestSeedTable:
    SIDES = ((mod.SPECIALIST, mod.STANDARDISED), (mod.UNTRAINED, mod.STANDARDISED))

    def test_holds_one_column_per_seed_and_a_summary(self, specialist):
        write_seed(specialist, 1)
        table = mod.seed_table(specialist, "bge_base", *self.SIDES, seeds=(0, 1))
        assert list(table.columns) == ["seed 0", "seed 1", "mean", "sd", "same_sign"]
        assert list(table.index) == FORMATS

    def test_reports_each_seed_on_its_own(self, specialist):
        write_seed(specialist, 1)
        write_tensor(specialist, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        row = mod.seed_table(specialist, "bge_base", *self.SIDES, seeds=(0, 1)).loc[
            "toml"
        ]
        assert row["seed 0"] == 0.0
        assert row["seed 1"] == -1.0
        assert row["mean"] == -0.5

    def test_counts_the_seeds_that_agree_with_the_mean(self, specialist):
        for seed in (1, 2):
            write_seed(specialist, seed)
        write_tensor(specialist, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        write_tensor(specialist, "bge_base", "only_toml_lam0_s2", bump=(2, 9.0))
        table = mod.seed_table(specialist, "bge_base", *self.SIDES, seeds=(0, 1, 2))
        assert table.loc["toml", "same_sign"] == 2

    def test_one_seed_has_no_spread(self, specialist):
        table = mod.seed_table(specialist, "bge_base", *self.SIDES, seeds=(0,))
        assert table["sd"].isna().all()

    def test_refuses_two_mixed_sides(self, specialist):
        with pytest.raises(ValueError, match="one side must be standardised"):
            mod.seed_table(
                specialist,
                "bge_base",
                (mod.WINNER, mod.MIXED),
                (mod.CONTROL, mod.MIXED),
                seeds=(0,),
            )


@pytest.fixture
def two_seeds(specialist):
    """Give both models every trained arm under seeds 0 and 1."""
    for slug in SLUGS:
        write_seed(specialist, 0, slug)
        write_seed(specialist, 1, slug)
    return specialist


class TestCommonTrainingSeeds:
    def test_returns_the_seeds_every_model_has(self, two_seeds):
        assert mod.common_training_seeds(two_seeds, SLUGS) == [0, 1]

    def test_drops_a_seed_one_model_lacks(self, two_seeds):
        tensor_path(two_seeds, "test", "minilm_l6", "only_json_lam0_s1").unlink()
        assert mod.common_training_seeds(two_seeds, SLUGS) == [0]

    def test_returns_nothing_for_no_models(self, two_seeds):
        assert mod.common_training_seeds(two_seeds, []) == []


class TestSeededMainTables:
    def test_levels_average_a_trained_condition_over_seeds(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "5fmt_lam0_s1", bump=(0, 9.0))
        table = mod.levels_table(two_seeds, ["bge_base"], seeds=(0, 1))
        row = table.loc[(by_slug("bge_base").label, mod.CONTROL)]
        assert row["standardised_mean"] == pytest.approx(0.9)

    def test_levels_leave_the_untrained_condition_alone(self, two_seeds):
        table = mod.levels_table(two_seeds, ["bge_base"], seeds=(0, 1))
        assert table.loc[(by_slug("bge_base").label, mod.UNTRAINED)]["mixed"] == 1.0

    def test_the_default_seed_matches_the_published_tables(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "5fmt_lam0_s1", bump=(0, 9.0))
        assert mod.levels_table(two_seeds, ["bge_base"]).equals(
            mod.levels_table(two_seeds, ["bge_base"], seeds=(0,))
        )

    def test_contrast_tests_the_seed_average(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "5fmt_lam1_s1", bump=(0, 9.0))
        table = small(
            mod.contrast_table,
            two_seeds,
            SLUGS,
            (mod.WINNER, "standardised_mean"),
            (mod.CONTROL, "standardised_mean"),
            seeds=(0, 1),
        )
        assert table.loc[by_slug("bge_base").label, "diff"] == pytest.approx(-0.1)

    def test_agreement_counts_the_share_of_seeds_that_disagree(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "5fmt_lam1_s1", bump=(0, 9.0))
        table = small(mod.agreement_table, two_seeds, SLUGS, seeds=(0, 1))
        assert table.loc[by_slug("bge_base").label, "disagree_invariance"] == 0.5

    def test_refuses_no_seed(self, two_seeds):
        with pytest.raises(ValueError, match="at least one training seed"):
            mod.levels_table(two_seeds, SLUGS, seeds=())


class TestTrainingLevelsTable:
    def test_holds_the_four_conditions_per_model(self, two_seeds):
        table = mod.training_levels_table(two_seeds, SLUGS, seeds=(0, 1))
        assert list(table.columns) == [
            mod.UNTRAINED,
            mod.SPECIALIST,
            mod.CONTROL,
            mod.WINNER,
        ]
        assert list(table.index) == [by_slug(s).label for s in SLUGS]

    def test_averages_over_formats_and_seeds(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        table = mod.training_levels_table(two_seeds, ["bge_base"], seeds=(0, 1))
        assert table.loc[by_slug("bge_base").label, mod.SPECIALIST] == pytest.approx(
            0.9
        )


class TestTrainingGapTable:
    def gaps(self, root, **k):
        return small(mod.training_gap_table, root, SLUGS, seeds=(0, 1), **k)

    def test_holds_one_row_per_model_and_format(self, two_seeds):
        table = self.gaps(two_seeds)
        assert len(table) == len(SLUGS) * F
        assert table.index.names == ["model", "format"]

    def test_a_worse_arm_shows_in_its_cell(self, two_seeds):
        for seed_arm in ("only_toml_lam0", "only_toml_lam0_s1"):
            write_tensor(two_seeds, "bge_base", seed_arm, bump=(2, 9.0))
        row = self.gaps(two_seeds).loc[(by_slug("bge_base").label, "toml")]
        assert row["diff"] == -1.0
        assert row["resolved"]
        assert row["same_sign"] == 2

    def test_reports_the_range_of_the_seeds(self, two_seeds):
        write_tensor(two_seeds, "bge_base", "only_toml_lam0_s1", bump=(2, 9.0))
        row = self.gaps(two_seeds).loc[(by_slug("bge_base").label, "toml")]
        assert row["seed_lo"] == -1.0
        assert row["seed_hi"] == 0.0
        assert row["seed_lo"] <= row["diff"] <= row["seed_hi"]

    def test_compares_to_the_named_condition(self, two_seeds):
        for arm_name in ("5fmt_lam1", "5fmt_lam1_s1"):
            write_tensor(two_seeds, "bge_base", arm_name, bump=(0, 9.0))
        winner = self.gaps(two_seeds, reference=mod.WINNER)
        control = self.gaps(two_seeds, reference=mod.CONTROL)
        label = by_slug("bge_base").label
        assert (winner.loc[label, "diff"] > 0).all()
        assert (control.loc[label, "diff"] == 0).all()


class TestTrainingSummaryTable:
    def test_holds_the_levels_then_each_comparison(self, two_seeds):
        table = small(mod.training_summary_table, two_seeds, SLUGS, seeds=(0, 1))
        assert list(table.columns) == [
            mod.UNTRAINED,
            mod.SPECIALIST,
            mod.CONTROL,
            mod.WINNER,
            "diff vs rank only",
            "formats resolved vs rank only",
            "seeds agreeing vs rank only",
            "diff vs invariance",
            "formats resolved vs invariance",
            "seeds agreeing vs invariance",
        ]

    def test_counts_formats_and_seeds(self, two_seeds):
        for seed_arm in ("only_toml_lam0", "only_toml_lam0_s1"):
            write_tensor(two_seeds, "bge_base", seed_arm, bump=(2, 9.0))
        row = small(mod.training_summary_table, two_seeds, SLUGS, seeds=(0, 1)).loc[
            by_slug("bge_base").label
        ]
        assert row["diff vs rank only"] == pytest.approx(-0.2)
        assert row["formats resolved vs rank only"] == 1
        assert row["seeds agreeing vs rank only"] == 2


class TestTrainingLevelContrastTable:
    def table(self, root):
        return small(mod.training_level_contrast_table, root, SLUGS, seeds=(0, 1))

    def test_holds_both_conditions_per_model(self, two_seeds):
        table = self.table(two_seeds)
        assert len(table) == len(SLUGS) * 2
        assert set(table.index.get_level_values("condition")) == {
            mod.SPECIALIST,
            mod.WINNER,
        }

    def test_averages_the_single_format_arms_over_formats(self, two_seeds):
        for arm_name in ("only_toml_lam0", "only_toml_lam0_s1"):
            write_tensor(two_seeds, "bge_base", arm_name, bump=(2, 9.0))
        row = self.table(two_seeds).loc[(by_slug("bge_base").label, mod.SPECIALIST)]
        assert row["diff"] == pytest.approx(-0.2)

    def test_matches_the_invariance_contrast(self, two_seeds):
        """The winner row is the accuracy contrast the stability figure shows."""
        write_tensor(two_seeds, "bge_base", "5fmt_lam1_s1", bump=(0, 9.0))
        ours = self.table(two_seeds).xs(mod.WINNER, level="condition")
        theirs = small(
            mod.contrast_table,
            two_seeds,
            SLUGS,
            (mod.WINNER, mod.MIXED),
            (mod.CONTROL, mod.MIXED),
            seeds=(0, 1),
        )
        assert ours["diff"].tolist() == pytest.approx(theirs["diff"].tolist())

    def test_corrects_within_each_condition(self, two_seeds):
        for arm_name in ("only_toml_lam0", "only_toml_lam0_s1"):
            write_tensor(two_seeds, "bge_base", arm_name, bump=(2, 9.0))
        table = self.table(two_seeds)
        assert (table["holm_p"] >= table["p"]).all()

    @pytest.mark.parametrize("root", ["0"], indirect=True)
    def test_omits_the_winner_when_it_is_the_control(self, root):
        for slug in SLUGS:
            write_seed(root, 0, slug)
        table = small(mod.training_level_contrast_table, root, SLUGS)
        assert set(table.index.get_level_values("condition")) == {mod.SPECIALIST}
