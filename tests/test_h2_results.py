"""Tests for fsr.h2_results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr import h2_results as mod
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import arm, comparison_path, result_path, selection_path
from fsr.models.registry import by_slug

SLUGS = ["minilm_l6", "bge_base"]
N_QUERIES = 8


def selection(winner="1", baseline=0.9, tie_broken=True, passes=5):
    weights = ["0", "0.01", "0.1", "1", "10"]
    return {
        "status": "selected",
        "winner_lambda": winner,
        "tie_broken": tie_broken,
        "baseline_max_abs_d": baseline,
        "candidates": [
            {
                "lambda": w,
                "dev_max_abs_d": 0.5 - i / 20,
                "dev_mean_mrr": 0.9,
                "mrr_ni_pass": i < passes,
            }
            for i, w in enumerate(weights)
        ],
    }


def comparison(held_out, trained_ood=0.2, trained_mrr=0.95, baseline_all=0.8):
    return {
        "held_out_format": held_out,
        "split": "test",
        "score_axis": {
            "point_estimates_max_d": {
                "all_formats": {"baseline": baseline_all, "trained": trained_ood},
                "in_training": {"baseline": 0.7, "trained": trained_ood / 2},
                "ood": {"baseline": 0.6, "trained": trained_ood},
            },
            "mrr_non_inferiority": {
                "margin": 0.03,
                "baseline_mean_mrr": 0.88,
                "trained_mean_mrr": trained_mrr,
                "passed": True,
            },
        },
        "answer_axis": {"n_queries": N_QUERIES, "n_pool": N_QUERIES},
    }


def leads(pattern):
    """Build gold-lead arrays from one string per format, `1` where it leads."""
    return {name: [c == "1" for c in pattern[i]] for i, name in enumerate(FORMAT_NAMES)}


BASE_LEADS = leads(
    [
        "11111100",
        "11111000",
        "11110100",
        "11101100",
        "11011100",
    ]
)
CONTROL_LEADS = leads(
    [
        "11111100",
        "11110000",
        "11100100",
        "11001100",
        "10011100",
    ]
)
TREATED_LEADS = leads(
    [
        "11111100",
        "11111100",
        "11111100",
        "11111100",
        "11111000",
    ]
)


def write_within(root, slug, arm_name, lead_map):
    path = result_path(root, "test", "within", slug, arm_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_queries": N_QUERIES,
        "results": {by_slug(slug).model_id: {"gold_leads_per_fmt": lead_map}},
    }
    path.write_text(json.dumps(payload))


@pytest.fixture
def root(tmp_path):
    """Write a complete two-model result set and return the corpus directory."""
    for slug in SLUGS:
        path = selection_path(tmp_path, slug, "lambda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection()))
        write_within(tmp_path, slug, "base", BASE_LEADS)
        for weight, trained, lead_map in (
            (0.0, 0.5, CONTROL_LEADS),
            (1.0, 0.2, TREATED_LEADS),
        ):
            for fold in FORMAT_NAMES:
                name = arm(fold, weight)
                out = comparison_path(tmp_path, slug, name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(comparison(fold, trained_ood=trained)))
                write_within(tmp_path, slug, name, lead_map)
    return tmp_path


class TestLoaders:
    def test_load_selection_reads_the_winner(self, root):
        assert mod.winner_weight(mod.load_selection(root, "bge_base")) == 1.0

    def test_load_comparison_reads_the_named_fold(self, root):
        got = mod.load_comparison(root, "bge_base", "toml", 1.0)
        assert got["held_out_format"] == "toml"

    def test_load_within_returns_the_single_entry(self, root):
        entry = mod.load_within(root, "bge_base", "base")
        assert set(entry["gold_leads_per_fmt"]) == set(FORMAT_NAMES)

    def test_load_within_rejects_a_multi_model_file(self, root):
        path = result_path(root, "test", "within", "bge_base", "base")
        path.write_text(json.dumps({"results": {"a": {}, "b": {}}}))
        with pytest.raises(ValueError, match="holds 2 models"):
            mod.load_within(root, "bge_base", "base")


class TestAvailable:
    def test_lists_models_in_roster_order(self, root):
        assert mod.available(root) == SLUGS

    def test_skips_a_model_with_no_selection(self, root):
        selection_path(root, "bge_base", "lambda").unlink()
        assert mod.available(root) == ["minilm_l6"]

    def test_skips_a_model_with_a_missing_fold(self, root):
        comparison_path(root, "bge_base", arm("markdown", 1.0)).unlink()
        assert mod.available(root) == ["minilm_l6"]


class TestSelectionTable:
    def test_carries_the_baseline_then_each_weight(self, root):
        table = mod.selection_table(root, SLUGS)
        assert list(table.columns) == [
            "baseline",
            "0",
            "0.01",
            "0.1",
            "1",
            "10",
            "winner",
        ]

    def test_is_indexed_by_label(self, root):
        table = mod.selection_table(root, SLUGS)
        assert list(table.index) == [by_slug(s).label for s in SLUGS]


class TestSelectionCounts:
    def test_counts_the_passing_candidates(self, root):
        assert mod.selection_counts(root, SLUGS) == {
            "passed": 10,
            "candidates": 10,
            "tie_broken": [by_slug(s).label for s in SLUGS],
        }

    def test_omits_a_model_whose_winner_was_not_tied(self, root):
        path = selection_path(root, "bge_base", "lambda")
        path.write_text(json.dumps(selection(tie_broken=False, passes=3)))
        counts = mod.selection_counts(root, SLUGS)
        assert counts["passed"] == 8
        assert counts["tie_broken"] == [by_slug("minilm_l6").label]


class TestFoldReaders:
    def test_fold_max_d_covers_every_format(self, root):
        values = mod.fold_max_d(root, "bge_base", 1.0)
        assert set(values) == set(FORMAT_NAMES)
        assert set(values.values()) == {0.2}

    def test_fold_max_d_honours_the_subset(self, root):
        values = mod.fold_max_d(root, "bge_base", 1.0, subset="in_training")
        assert set(values.values()) == {0.1}

    def test_fold_mean_mrr_reads_the_trained_arm(self, root):
        assert set(mod.fold_mean_mrr(root, "bge_base", 1.0).values()) == {0.95}

    def test_baseline_max_d_reads_every_pair(self, root):
        assert mod.baseline_max_d(root, "bge_base") == 0.8


class TestScoreTable:
    def test_delta_is_treatment_against_control(self, root):
        table = mod.score_table(root, SLUGS)
        assert table["delta"].unique().tolist() == [pytest.approx(-0.3)]

    def test_counts_the_folds_that_improved(self, root):
        table = mod.score_table(root, SLUGS)
        assert table["folds_improved"].unique().tolist() == [len(FORMAT_NAMES)]

    def test_relative_change_is_a_share_of_the_control(self, root):
        table = mod.score_table(root, SLUGS)
        assert table["delta_pct"].iloc[0] == pytest.approx(-60.0)

    def test_carries_the_untrained_baseline(self, root):
        table = mod.score_table(root, SLUGS)
        assert table["baseline_max_d"].iloc[0] == 0.8


class TestPool:
    def test_marks_every_query_the_untrained_model_answers(self, root):
        pool = mod.baseline_pool(root, "bge_base")
        assert pool.tolist() == [True] * 6 + [False, False]


class TestPooledInconsistency:
    def test_counts_over_the_pool_only(self):
        flags = np.array([[True, True, False, False]])
        pool = np.array([True, False, True, False])
        assert mod.pooled_inconsistency(flags, pool) == pytest.approx(50.0)

    def test_divides_by_the_pool_times_the_folds(self):
        flags = np.array([[True, False], [True, True]])
        pool = np.array([True, True])
        assert mod.pooled_inconsistency(flags, pool) == pytest.approx(75.0)

    def test_is_nan_when_the_pool_is_empty(self):
        flags = np.array([[True, True]])
        pool = np.array([False, False])
        assert np.isnan(mod.pooled_inconsistency(flags, pool))

    def test_applies_a_resample(self):
        flags = np.array([[True, False]])
        pool = np.array([True, True])
        index = np.array([0, 0])
        assert mod.pooled_inconsistency(flags, pool, index) == pytest.approx(100.0)


class TestBootstrapPooledDelta:
    def test_brackets_a_real_reduction(self, root):
        pool = mod.baseline_pool(root, "bge_base")
        control = mod.inconsistent_matrix(root, "bge_base", 0.0)
        treated = mod.inconsistent_matrix(root, "bge_base", 1.0)
        low, high = mod.bootstrap_pooled_delta(control, treated, pool, n_boot=200)
        assert low <= high
        assert high <= 0.0

    def test_is_reproducible_for_one_seed(self, root):
        pool = mod.baseline_pool(root, "bge_base")
        control = mod.inconsistent_matrix(root, "bge_base", 0.0)
        treated = mod.inconsistent_matrix(root, "bge_base", 1.0)
        first = mod.bootstrap_pooled_delta(control, treated, pool, n_boot=50, seed=3)
        second = mod.bootstrap_pooled_delta(control, treated, pool, n_boot=50, seed=3)
        assert first == second


class TestAnswerTable:
    def test_reports_a_reduction(self, root):
        table = mod.answer_table(root, SLUGS, n_boot=200)
        assert (table["delta_pp"] < 0).all()

    def test_pool_is_the_untrained_answerable_count(self, root):
        table = mod.answer_table(root, SLUGS, n_boot=50)
        assert table["n_pool"].unique().tolist() == [6]

    def test_flags_an_interval_that_excludes_zero(self, root):
        table = mod.answer_table(root, SLUGS, subset="all_formats", n_boot=200)
        assert table["excludes_zero"].all()

    def test_leaves_a_straddling_interval_unflagged(self, root):
        table = mod.answer_table(root, SLUGS, n_boot=200)
        assert not table["excludes_zero"].any()


class TestPerFoldMatrix:
    def test_score_axis_has_one_column_per_fold(self, root):
        matrix = mod.per_fold_matrix(root, SLUGS, "score")
        assert list(matrix.columns) == list(FORMAT_NAMES)
        assert np.allclose(matrix.to_numpy(), -0.3)

    def test_answer_axis_reports_percentage_points(self, root):
        matrix = mod.per_fold_matrix(root, SLUGS, "answer")
        assert list(matrix.columns) == list(FORMAT_NAMES)
        assert (matrix.to_numpy() <= 0).all()

    def test_rejects_an_unknown_axis(self, root):
        with pytest.raises(ValueError, match="unknown axis"):
            mod.per_fold_matrix(root, SLUGS, "mrr")


class TestBothAxes:
    def test_joins_on_the_model_label(self, root):
        score = mod.score_table(root, SLUGS)
        answer = mod.answer_table(root, SLUGS, n_boot=50)
        joined = mod.both_axes(score, answer)
        assert list(joined.columns) == ["score_delta", "delta_pp", "ci_lo", "ci_hi"]
        assert list(joined.index) == [by_slug(s).label for s in SLUGS]


class TestCorrelation:
    def test_returns_the_pearson_coefficient(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [2.0, 4.0, 6.0]})
        assert mod.correlation(frame, "a", "b") == pytest.approx(1.0)

    def test_is_nan_for_a_single_row(self):
        import pandas as pd

        frame = pd.DataFrame({"a": [1.0], "b": [2.0]})
        assert np.isnan(mod.correlation(frame, "a", "b"))


class TestRoster:
    def test_leaves_out_a_head_variant(self, root, tmp_path):
        slug = "mxbai_v1_tanh"
        path = selection_path(tmp_path, slug, "lambda")
        path.write_text(json.dumps(selection()))
        write_within(tmp_path, slug, "base", BASE_LEADS)
        for fold in FORMAT_NAMES:
            for weight, lead_map in ((0.0, CONTROL_LEADS), (1.0, TREATED_LEADS)):
                name = arm(fold, weight)
                out = comparison_path(tmp_path, slug, name)
                out.write_text(json.dumps(comparison(fold)))
                write_within(tmp_path, slug, name, lead_map)
        assert slug not in mod.available(root)
