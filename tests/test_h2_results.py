"""Tests for fsr.h2_results."""

from __future__ import annotations

import json

import numpy as np
import pytest

from fsr import h2_results as mod
from fsr.comparison import max_abs_d_over_pairs, pair_subsets
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import arm, comparison_path, result_path, selection_path
from fsr.models.registry import by_slug

SLUGS = ["minilm_l6", "bge_base"]
N_QUERIES = 8
BOOT = 60


def selection(winner="1", baseline=0.9, tie_broken=True, passes=5):
    weights = ["0", "0.01", "0.1", "1", "10"]
    return {
        "status": "selected",
        "winner_lambda": winner,
        "tie_broken": tie_broken,
        "baseline_max_abs_d": baseline,
        "tied_lambdas": ["0.1", "1", "10"],
        "candidates": [
            {
                "lambda": w,
                "dev_max_abs_d": 0.5 - i / 20,
                "dev_max_abs_d_ci_lo": 0.5 - i / 20 - 0.05,
                "dev_max_abs_d_ci_hi": 0.5 - i / 20 + 0.05,
                "dev_mean_mrr": 0.9,
                "mrr_ni_pass": i < passes,
            }
            for i, w in enumerate(weights)
        ],
    }


SUBSETS = ("all_formats", "in_training", "ood")
BASELINE_MAX_D = {"all_formats": 0.8, "in_training": 0.7, "ood": 0.6}


def comparison(held_out, spread, trained_mrr=0.95):
    """Build a comparison whose point estimates come from the same scores."""
    arrays = {name: np.asarray(v) for name, v in scores(spread).items()}
    pairs = pair_subsets(held_out)
    return {
        "held_out_format": held_out,
        "split": "test",
        "score_axis": {
            "point_estimates_max_d": {
                name: {
                    "baseline": BASELINE_MAX_D[name],
                    "trained": max_abs_d_over_pairs(arrays, pairs[name])[0],
                }
                for name in SUBSETS
            },
            "mrr_non_inferiority": {
                "margin": 0.03,
                "baseline_mean_mrr": 0.88,
                "trained_mean_mrr": trained_mrr,
                "delta_mrr_bootstrap": {
                    "delta_mean": trained_mrr - 0.88,
                    "delta_ci": [trained_mrr - 0.90, trained_mrr - 0.86],
                },
                "passed": True,
            },
            "heldout_transfer": {
                "held_out_format": held_out,
                "delta_mrr": 0.07,
                "delta_mrr_ci": [0.06, 0.08],
                "training_format_mean_delta_mrr": 0.0625,
                "transfer_ratio": 0.07 / 0.0625,
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


N_RECORDS = 60


def scores(spread):
    """Separate the formats by `spread` standard deviations, reproducibly."""
    rng = np.random.default_rng(11)
    noise = rng.normal(size=(len(FORMAT_NAMES), N_RECORDS))
    return {
        name: (noise[i] + i * spread).tolist() for i, name in enumerate(FORMAT_NAMES)
    }


def write_cross(root, slug, arm_name, spread):
    path = result_path(root, "test", "cross", slug, arm_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"scores_per_fmt": scores(spread)}))


@pytest.fixture
def root(tmp_path):
    """Write a complete two-model result set and return the corpus directory."""
    for slug in SLUGS:
        path = selection_path(tmp_path, slug, "lambda")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(selection()))
        write_within(tmp_path, slug, "base", BASE_LEADS)
        for weight, lead_map, spread in (
            (0.0, CONTROL_LEADS, 1.0),
            (1.0, TREATED_LEADS, 0.1),
        ):
            for fold in FORMAT_NAMES:
                name = arm(fold, weight)
                out = comparison_path(tmp_path, slug, name)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(comparison(fold, spread)))
                write_within(tmp_path, slug, name, lead_map)
                write_cross(tmp_path, slug, name, spread)
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
        assert all(v > 0 for v in values.values())

    def test_fold_max_d_honours_the_subset(self, root):
        held_out = mod.fold_max_d(root, "bge_base", 1.0)
        trained = mod.fold_max_d(root, "bge_base", 1.0, subset="in_training")
        assert held_out != trained

    def test_fold_max_d_matches_the_scores_it_was_built_from(self, root):
        from_file = np.mean(list(mod.fold_max_d(root, "bge_base", 1.0).values()))
        recomputed = mod.mean_max_d(mod.fold_scores(root, "bge_base", 1.0), "ood")
        assert from_file == pytest.approx(recomputed)

    def test_fold_mean_mrr_reads_the_trained_arm(self, root):
        assert set(mod.fold_mean_mrr(root, "bge_base", 1.0).values()) == {0.95}

    def test_baseline_max_d_reads_every_pair(self, root):
        assert mod.baseline_max_d(root, "bge_base") == 0.8


class TestScoreTable:
    def test_delta_is_treatment_against_control(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
        assert (table["delta"] < 0).all()

    def test_counts_the_folds_that_improved(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
        assert table["folds_improved"].unique().tolist() == [len(FORMAT_NAMES)]

    def test_relative_change_is_a_share_of_the_control(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
        expected = 100.0 * table["delta"] / table["control_max_d"]
        assert table["delta_pct"].tolist() == pytest.approx(expected.tolist())

    def test_carries_the_untrained_baseline(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
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
        assert (matrix.to_numpy() < 0).all()

    def test_answer_axis_reports_percentage_points(self, root):
        matrix = mod.per_fold_matrix(root, SLUGS, "answer")
        assert list(matrix.columns) == list(FORMAT_NAMES)
        assert (matrix.to_numpy() <= 0).all()

    def test_rejects_an_unknown_axis(self, root):
        with pytest.raises(ValueError, match="unknown axis"):
            mod.per_fold_matrix(root, SLUGS, "mrr")


class TestBothAxes:
    def test_joins_on_the_model_label(self, root):
        score = mod.score_table(root, SLUGS, n_boot=BOOT)
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
            for weight, lead_map, spread in (
                (0.0, CONTROL_LEADS, 1.0),
                (1.0, TREATED_LEADS, 0.1),
            ):
                name = arm(fold, weight)
                out = comparison_path(tmp_path, slug, name)
                out.write_text(json.dumps(comparison(fold, spread)))
                write_within(tmp_path, slug, name, lead_map)
                write_cross(tmp_path, slug, name, spread)
        assert slug not in mod.available(root)


class TestLoadScores:
    def test_reads_one_array_per_format(self, root):
        got = mod.load_scores(root, "bge_base", arm("yaml", 1.0))
        assert set(got) == set(FORMAT_NAMES)
        assert all(len(v) == N_RECORDS for v in got.values())

    def test_fold_scores_covers_every_fold(self, root):
        folds = mod.fold_scores(root, "bge_base", 1.0)
        assert len(folds) == len(FORMAT_NAMES)
        held_out = [set(pairs["ood"][0]) for _, pairs in folds]
        assert all(FORMAT_NAMES[i] in h for i, h in enumerate(held_out))


class TestMeanMaxD:
    def test_a_wider_spread_gives_a_larger_effect(self, root):
        control = mod.fold_scores(root, "bge_base", 0.0)
        treated = mod.fold_scores(root, "bge_base", 1.0)
        assert mod.mean_max_d(control, "ood") > mod.mean_max_d(treated, "ood")

    def test_a_resample_changes_the_value(self, root):
        folds = mod.fold_scores(root, "bge_base", 0.0)
        index = np.zeros(N_RECORDS, dtype=int)
        assert mod.mean_max_d(folds, "ood", index) != mod.mean_max_d(folds, "ood")


class TestBootstrapDeltaMaxD:
    def test_brackets_a_real_reduction(self, root):
        control = mod.fold_scores(root, "bge_base", 0.0)
        treated = mod.fold_scores(root, "bge_base", 1.0)
        low, high = mod.bootstrap_delta_max_d(control, treated, n_boot=BOOT)
        assert low <= high
        assert high < 0.0

    def test_is_reproducible_for_one_seed(self, root):
        control = mod.fold_scores(root, "bge_base", 0.0)
        treated = mod.fold_scores(root, "bge_base", 1.0)
        first = mod.bootstrap_delta_max_d(control, treated, n_boot=BOOT, seed=5)
        second = mod.bootstrap_delta_max_d(control, treated, n_boot=BOOT, seed=5)
        assert first == second

    def test_honours_the_subset(self, root):
        control = mod.fold_scores(root, "bge_base", 0.0)
        treated = mod.fold_scores(root, "bge_base", 1.0)
        held_out = mod.bootstrap_delta_max_d(control, treated, n_boot=BOOT)
        trained = mod.bootstrap_delta_max_d(
            control, treated, subset="in_training", n_boot=BOOT
        )
        assert held_out != trained


class TestScoreTableIntervals:
    def test_carries_an_interval_around_the_change(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
        assert (table["ci_lo"] <= table["delta"]).all()
        assert (table["delta"] <= table["ci_hi"]).all()

    def test_flags_an_interval_that_excludes_zero(self, root):
        table = mod.score_table(root, SLUGS, n_boot=BOOT)
        assert table["excludes_zero"].all()


class TestWeightFrame:
    def test_is_ordered_by_weight(self, root):
        frame = mod.weight_frame(root, "bge_base")
        assert list(frame.index) == ["0", "0.01", "0.1", "1", "10"]

    def test_marks_the_winner_once(self, root):
        frame = mod.weight_frame(root, "bge_base")
        assert frame["winner"].sum() == 1
        assert frame[frame["winner"]].index[0] == "1"

    def test_marks_the_tied_group(self, root):
        frame = mod.weight_frame(root, "bge_base")
        assert frame[frame["tied"]].index.tolist() == ["0.1", "1", "10"]

    def test_carries_the_interval(self, root):
        frame = mod.weight_frame(root, "bge_base")
        assert (frame["ci_lo"] <= frame["max_abs_d"]).all()
        assert (frame["max_abs_d"] <= frame["ci_hi"]).all()

    def test_a_missing_interval_is_nan(self, root):
        payload = selection()
        for candidate in payload["candidates"]:
            del candidate["dev_max_abs_d_ci_lo"]
            del candidate["dev_max_abs_d_ci_hi"]
        selection_path(root, "bge_base", "lambda").write_text(json.dumps(payload))
        frame = mod.weight_frame(root, "bge_base")
        assert frame["ci_lo"].isna().all()

    def test_no_tied_group_marks_nothing(self, root):
        payload = selection()
        del payload["tied_lambdas"]
        selection_path(root, "bge_base", "lambda").write_text(json.dumps(payload))
        assert not mod.weight_frame(root, "bge_base")["tied"].any()


class TestGuardrailFrame:
    def test_has_a_row_for_every_model_and_fold(self, root):
        frame = mod.guardrail_frame(root, SLUGS)
        assert len(frame) == len(SLUGS) * len(FORMAT_NAMES)
        assert list(frame.index.names) == ["model", "held_out"]

    def test_carries_the_interval_and_the_margin(self, root):
        frame = mod.guardrail_frame(root, SLUGS)
        assert (frame["ci_lo"] <= frame["delta_mrr"]).all()
        assert (frame["delta_mrr"] <= frame["ci_hi"]).all()
        assert frame["margin"].unique().tolist() == [0.03]

    def test_reports_the_outcome_without_recomputing_it(self, root):
        assert mod.guardrail_frame(root, SLUGS)["passed"].all()


class TestTransferFrame:
    def test_has_a_row_for_every_model_and_fold(self, root):
        frame = mod.transfer_frame(root, SLUGS)
        assert len(frame) == len(SLUGS) * len(FORMAT_NAMES)

    def test_ratio_relates_the_held_out_change_to_the_training_mean(self, root):
        frame = mod.transfer_frame(root, SLUGS)
        expected = frame["delta_mrr"] / frame["training_mean"]
        assert frame["transfer_ratio"].tolist() == pytest.approx(expected.tolist())

    def test_carries_no_threshold(self, root):
        frame = mod.transfer_frame(root, SLUGS)
        assert "passed" not in frame.columns


N_PROSE = 12
PROSE_IDS = [f"p{i}" for i in range(N_PROSE)]
# The winner beats the control on every record of every fold, by this much.
PROSE_GAIN = 0.02
BASE_PROSE_MRR = 0.90


def prose_result(slug, arm_name, mrr):
    return {
        "model": slug,
        "arm": arm_name,
        "split": "prose",
        "mrr": mrr,
        "record_ids": list(PROSE_IDS),
        "reciprocal_ranks": [mrr] * N_PROSE,
    }


def prose_comparison(slug, arm_name, delta, ni_pass=True, ids=None):
    return {
        "model": slug,
        "arm": arm_name,
        "split": "prose",
        "baseline_mrr": BASE_PROSE_MRR,
        "trained_mrr": BASE_PROSE_MRR + delta,
        "delta_mrr_mean": delta,
        "delta_mrr_ci": [delta - 0.005, delta + 0.005],
        "ni_pass": ni_pass,
        "record_ids": list(PROSE_IDS if ids is None else ids),
        "per_record_delta_rr": [delta] * N_PROSE,
    }


def write_prose(root, slug, control_delta=0.01, ni_pass=True, ids=None):
    """Write one model's prose baseline and both arms of every fold."""
    path = result_path(root, "prose", "mrr", slug, "base")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(prose_result(slug, "base", BASE_PROSE_MRR)))
    for weight, delta in ((0.0, control_delta), (1.0, control_delta + PROSE_GAIN)):
        for fold in FORMAT_NAMES:
            name = arm(fold, weight)
            out = comparison_path(root, slug, f"prose_{name}")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(
                json.dumps(
                    prose_comparison(
                        slug,
                        name,
                        delta,
                        ni_pass,
                        ids if weight == 1.0 else None,
                    )
                )
            )


@pytest.fixture
def prose_root(root):
    """Add a complete prose result set to the standard corpus."""
    for slug in SLUGS:
        write_prose(root, slug)
    return root


class TestProseAvailable:
    def test_finds_a_model_whose_prose_run_is_complete(self, prose_root):
        assert mod.prose_available(prose_root) == SLUGS

    def test_reports_nothing_when_no_prose_run_is_present(self, root):
        assert mod.prose_available(root) == []

    def test_skips_a_model_missing_one_arm(self, prose_root):
        target = comparison_path(prose_root, "bge_base", "prose_toml_lam1")
        target.unlink()
        assert mod.prose_available(prose_root) == ["minilm_l6"]

    def test_skips_a_model_with_no_selection(self, prose_root):
        selection_path(prose_root, "bge_base", "lambda").unlink()
        assert mod.prose_available(prose_root) == ["minilm_l6"]


class TestProseArms:
    def test_names_the_control_and_the_winner_of_every_fold(self, prose_root):
        names = mod.prose_arms(prose_root, "minilm_l6")
        assert set(names) == set(FORMAT_NAMES)
        assert names["yaml"] == {"control": "yaml_lam0", "winner": "yaml_lam1"}


class TestProseBaseline:
    def test_reads_the_untrained_ranking_quality(self, prose_root):
        assert mod.prose_baseline_mrr(prose_root, "minilm_l6") == BASE_PROSE_MRR


class TestProsePairedGains:
    def test_holds_one_row_per_fold(self, prose_root):
        gains = mod.prose_paired_gains(prose_root, "minilm_l6")
        assert gains.shape == (len(FORMAT_NAMES), N_PROSE)

    def test_subtracts_the_control_from_the_winner(self, prose_root):
        gains = mod.prose_paired_gains(prose_root, "minilm_l6")
        assert gains == pytest.approx(PROSE_GAIN)

    def test_refuses_arms_that_cover_different_records(self, root):
        """An unpaired subtraction would silently compare different queries."""
        write_prose(root, "minilm_l6", ids=[f"q{i}" for i in range(N_PROSE)])
        with pytest.raises(ValueError, match="different records"):
            mod.prose_paired_gains(root, "minilm_l6")


class TestProsePairedDelta:
    def test_reports_the_mean_gain(self, prose_root):
        mean, _, _ = mod.prose_paired_delta(prose_root, "minilm_l6", n_boot=BOOT)
        assert mean == pytest.approx(PROSE_GAIN)

    def test_a_constant_gain_gives_a_degenerate_interval(self, prose_root):
        """Every record moves by the same amount, so no resample can differ."""
        _, low, high = mod.prose_paired_delta(prose_root, "minilm_l6", n_boot=BOOT)
        assert low == pytest.approx(PROSE_GAIN)
        assert high == pytest.approx(PROSE_GAIN)


class TestProseFoldFrame:
    def test_holds_one_row_per_held_out_format(self, prose_root):
        frame = mod.prose_fold_frame(prose_root, "minilm_l6")
        assert list(frame.index) == list(FORMAT_NAMES)

    def test_carries_both_arms(self, prose_root):
        frame = mod.prose_fold_frame(prose_root, "minilm_l6")
        assert frame.loc["yaml", "control_delta"] == pytest.approx(0.01)
        assert frame.loc["yaml", "winner_delta"] == pytest.approx(0.01 + PROSE_GAIN)

    def test_carries_the_gain_of_each_fold(self, prose_root):
        frame = mod.prose_fold_frame(prose_root, "minilm_l6")
        assert frame["winner_gain"].to_numpy() == pytest.approx(PROSE_GAIN)

    def test_marks_the_arms_that_kept_their_quality(self, prose_root):
        frame = mod.prose_fold_frame(prose_root, "minilm_l6")
        assert frame["control_kept"].all()
        assert frame["winner_kept"].all()

    def test_marks_an_arm_that_lost_quality(self, root):
        write_prose(root, "minilm_l6", ni_pass=False)
        frame = mod.prose_fold_frame(root, "minilm_l6")
        assert not frame["winner_kept"].any()


class TestProseTable:
    def test_holds_one_row_per_model(self, prose_root):
        table = mod.prose_table(prose_root, SLUGS, n_boot=BOOT)
        assert list(table.index) == [by_slug(s).label for s in SLUGS]

    def test_reports_the_untrained_quality_and_both_arms(self, prose_root):
        table = mod.prose_table(prose_root, SLUGS, n_boot=BOOT)
        row = table.loc[by_slug("minilm_l6").label]
        assert row["baseline_mrr"] == BASE_PROSE_MRR
        assert row["control_mrr"] == pytest.approx(BASE_PROSE_MRR + 0.01)
        assert row["winner_mrr"] == pytest.approx(BASE_PROSE_MRR + 0.01 + PROSE_GAIN)

    def test_counts_the_arms_that_kept_their_quality(self, prose_root):
        table = mod.prose_table(prose_root, SLUGS, n_boot=BOOT)
        assert (table["arms_kept"] == 2 * len(FORMAT_NAMES)).all()

    def test_a_gain_clear_of_zero_is_marked(self, prose_root):
        table = mod.prose_table(prose_root, SLUGS, n_boot=BOOT)
        assert table["gain_excludes_zero"].all()

    def test_no_gain_is_not_marked(self, root):
        """A winner that matches its control must not read as an improvement."""
        for slug in SLUGS:
            write_prose(root, slug)
            for fold in FORMAT_NAMES:
                out = comparison_path(root, slug, f"prose_{arm(fold, 1.0)}")
                out.write_text(json.dumps(prose_comparison(slug, fold, 0.01)))
        table = mod.prose_table(root, SLUGS, n_boot=BOOT)
        assert not table["gain_excludes_zero"].any()


class TestProseMatrix:
    def test_holds_one_column_per_held_out_format(self, prose_root):
        grid = mod.prose_matrix(prose_root, SLUGS)
        assert list(grid.columns) == list(FORMAT_NAMES)

    def test_reports_the_winner_by_default(self, prose_root):
        grid = mod.prose_matrix(prose_root, SLUGS)
        assert grid.to_numpy() == pytest.approx(0.01 + PROSE_GAIN)

    def test_reports_the_control_when_asked(self, prose_root):
        grid = mod.prose_matrix(prose_root, SLUGS, condition="control")
        assert grid.to_numpy() == pytest.approx(0.01)

    def test_refuses_an_unknown_arm(self, prose_root):
        with pytest.raises(ValueError, match="unknown arm"):
            mod.prose_matrix(prose_root, SLUGS, condition="winner_lam3")
