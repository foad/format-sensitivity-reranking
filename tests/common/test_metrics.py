"""Tests for fsr.common.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

import fsr.common.metrics as metrics_module
from fsr.common.metrics import (
    MIN_RANK_STABILITY_RECORDS,
    _extreme,
    _percentile_ci,
    bootstrap_ci_of_max_abs_d,
    bootstrap_ci_of_mean,
    format_sensitivity_summary,
    pairwise_cohen_d,
    per_format_summary,
    rank_stability,
    reciprocal_ranks,
)

A = [1.0, 2.0, 3.0, 4.0]
B = [2.0, 4.0, 4.0, 4.0]
TWO = {"a": A, "b": B}
PAIR = ("a", "b")

MONOTONE = {
    "a": [1.0, 2.0, 3.0, 4.0, 5.0],
    "b": [2.0, 4.0, 6.0, 8.0, 10.0],
    "c": [-1.0, -2.0, -3.0, -4.0, -5.0],
}
TRIPLE = ("a", "b", "c")

CONSTANT = {"a": [1.0] * 40, "b": [1.0] * 40}


class TestPercentileCi:
    def test_returns_the_requested_percentiles(self):
        lo, hi = _percentile_ci(np.arange(101.0), 0.95)
        assert lo == pytest.approx(2.5)
        assert hi == pytest.approx(97.5)

    def test_a_wider_interval_gives_wider_bounds(self):
        narrow = _percentile_ci(np.arange(101.0), 0.5)
        wide = _percentile_ci(np.arange(101.0), 0.99)
        assert wide[0] < narrow[0]
        assert wide[1] > narrow[1]


class TestExtreme:
    def test_selects_the_minimum_and_its_pair(self):
        entries = [{"pair": "a", "v": 3.0}, {"pair": "b", "v": 1.0}]
        assert _extreme(entries, "v", min) == (1.0, "b")

    def test_selects_the_maximum_and_its_pair(self):
        entries = [{"pair": "a", "v": 3.0}, {"pair": "b", "v": 1.0}]
        assert _extreme(entries, "v", max) == (3.0, "a")

    def test_ignores_undefined_values(self):
        entries = [{"pair": "a", "v": math.nan}, {"pair": "b", "v": 2.0}]
        assert _extreme(entries, "v", min) == (2.0, "b")

    def test_reports_undefined_when_nothing_is_defined(self):
        value, pair = _extreme([{"pair": "a", "v": math.nan}], "v", min)
        assert math.isnan(value) and pair is None

    def test_reports_undefined_for_no_entries(self):
        value, pair = _extreme([], "v", min)
        assert math.isnan(value) and pair is None

    def test_a_tie_keeps_the_first_pair(self):
        entries = [{"pair": "a", "v": 1.0}, {"pair": "b", "v": 1.0}]
        assert _extreme(entries, "v", min) == (1.0, "a")


class TestPerFormatSummary:
    def test_reports_mean_std_and_median(self):
        assert per_format_summary(TWO) == {
            "a": {"mean": 2.5, "std": pytest.approx(1.118033988749895), "median": 2.5},
            "b": {"mean": 3.5, "std": pytest.approx(0.8660254037844386), "median": 4.0},
        }

    def test_covers_every_supplied_format(self):
        assert set(per_format_summary(MONOTONE)) == {"a", "b", "c"}


class TestPairwiseCohenD:
    def test_entry_values(self):
        out, _, _ = pairwise_cohen_d(TWO, formats=PAIR)
        assert out == [
            {
                "pair": "a - b",
                "mean_diff": -1.0,
                "cohen_d": pytest.approx(-1.414213562371095),
                "wilcoxon_p": pytest.approx(0.25),
            }
        ]

    def test_reports_the_largest_absolute_effect_and_its_pair(self):
        _, max_abs_d, max_d_pair = pairwise_cohen_d(TWO, formats=PAIR)
        assert max_abs_d == pytest.approx(1.414213562371095)
        assert max_d_pair == "a - b"

    def test_pairs_every_combination_of_the_given_formats(self):
        out, _, _ = pairwise_cohen_d(MONOTONE, formats=TRIPLE)
        assert [e["pair"] for e in out] == ["a - b", "a - c", "b - c"]

    def test_a_single_format_yields_no_pairs(self):
        out, max_abs_d, max_d_pair = pairwise_cohen_d(TWO, formats=("a",))
        assert (out, max_abs_d, max_d_pair) == ([], 0.0, None)

    def test_a_scipy_value_error_becomes_a_p_value_of_one(self, monkeypatch):
        # SciPy 1.18 does not raise for any input this module passes, so the
        # fallback is only reachable by forcing the error.
        def raise_value_error(*args, **kwargs):
            raise ValueError("forced")

        monkeypatch.setattr(metrics_module.stats, "wilcoxon", raise_value_error)
        out, _, _ = pairwise_cohen_d(TWO, formats=PAIR)
        assert out[0]["wilcoxon_p"] == 1.0

    def test_identical_scores_give_zero_effect_and_a_p_value_of_one(self):
        out, _, _ = pairwise_cohen_d(CONSTANT, formats=PAIR)
        assert out[0]["cohen_d"] == 0.0
        assert out[0]["mean_diff"] == 0.0
        assert out[0]["wilcoxon_p"] == 1.0

    @pytest.mark.parametrize("n", [10, 25, 26, 56])
    def test_the_p_value_for_identical_scores_does_not_depend_on_size(self, n):
        scores = {"a": [1.0] * n, "b": [1.0] * n}
        out, _, _ = pairwise_cohen_d(scores, formats=PAIR)
        assert out[0]["wilcoxon_p"] == 1.0


class TestRankStability:
    def test_a_monotone_transform_agrees_perfectly(self):
        out, _, _, _, _ = rank_stability(MONOTONE, sample_size=200, formats=PAIR)
        assert out[0]["spearman_rho"] == pytest.approx(1.0)
        assert out[0]["flip_rate_pct"] == 0.0

    def test_an_inverted_transform_disagrees_completely(self):
        out, _, _, _, _ = rank_stability(MONOTONE, sample_size=200, formats=("a", "c"))
        assert out[0]["spearman_rho"] == pytest.approx(-1.0)
        assert out[0]["flip_rate_pct"] == 100.0

    def test_reports_the_extreme_pairs(self):
        _, min_rho, min_rho_pair, max_flip, max_flip_pair = rank_stability(
            MONOTONE, sample_size=200, formats=TRIPLE
        )
        assert min_rho == pytest.approx(-1.0)
        assert min_rho_pair == "a vs c"
        assert max_flip == 100.0
        assert max_flip_pair == "a vs c"

    def test_rejects_a_single_record(self):
        with pytest.raises(ValueError, match="at least 2 records"):
            rank_stability({"a": [1.0], "b": [2.0]}, formats=PAIR)

    def test_accepts_two_records(self):
        out, _, _, _, _ = rank_stability(
            {"a": [1.0, 2.0], "b": [2.0, 4.0]}, sample_size=50, formats=PAIR
        )
        assert len(out) == 1

    def test_is_deterministic_for_a_given_seed(self):
        first = rank_stability(MONOTONE, sample_size=200, seed=7, formats=TRIPLE)
        second = rank_stability(MONOTONE, sample_size=200, seed=7, formats=TRIPLE)
        assert first == second

    @pytest.mark.filterwarnings("ignore")
    def test_an_undefined_rho_is_reported_as_undefined(self):
        _, min_rho, min_rho_pair, _, _ = rank_stability(
            CONSTANT, sample_size=50, formats=PAIR
        )
        assert math.isnan(min_rho)
        assert min_rho_pair is None

    @pytest.mark.filterwarnings("ignore")
    def test_an_undefined_rho_does_not_hide_a_defined_one(self):
        scores = {"a": [1.0] * 40, "b": [1.0] * 40, "c": [float(i) for i in range(40)]}
        _, min_rho, min_rho_pair, _, _ = rank_stability(
            scores, sample_size=200, formats=("a", "b", "c")
        )
        assert math.isnan(min_rho)
        assert min_rho_pair is None

    def test_no_pairs_gives_an_undefined_extreme(self):
        _, min_rho, min_rho_pair, max_flip, max_flip_pair = rank_stability(
            MONOTONE, sample_size=50, formats=("a",)
        )
        assert math.isnan(min_rho) and min_rho_pair is None
        assert math.isnan(max_flip) and max_flip_pair is None


class TestFormatSensitivitySummary:
    def test_rejects_a_single_record(self):
        single = {f: [1.0] for f in TRIPLE}
        with pytest.raises(ValueError, match="at least 2 records"):
            format_sensitivity_summary(single, formats=TRIPLE)

    def test_the_minimum_record_count_is_two(self):
        assert MIN_RANK_STABILITY_RECORDS == 2

    def test_summary_holds_every_reported_field(self):
        summary = format_sensitivity_summary(MONOTONE, formats=TRIPLE)["summary"]
        assert sorted(summary) == [
            "format_ranking_by_mean",
            "max_abs_cohen_d",
            "max_d_pair",
            "max_flip_pair",
            "max_flip_rate_pct",
            "mean_abs_cohen_d",
            "min_rho_pair",
            "min_spearman_rho",
        ]

    def test_orders_formats_by_descending_mean(self):
        result = format_sensitivity_summary(MONOTONE, formats=TRIPLE)
        assert result["summary"]["format_ranking_by_mean"] == ["b", "a", "c"]

    def test_sections_match_the_component_functions(self):
        result = format_sensitivity_summary(MONOTONE, formats=TRIPLE)
        pairwise, max_d, max_d_pair = pairwise_cohen_d(MONOTONE, formats=TRIPLE)
        assert result["per_format"] == per_format_summary(MONOTONE)
        assert result["pairwise"] == pairwise
        assert result["summary"]["max_abs_cohen_d"] == max_d
        assert result["summary"]["max_d_pair"] == max_d_pair

    def test_mean_absolute_effect_averages_the_pairwise_effects(self):
        result = format_sensitivity_summary(MONOTONE, formats=TRIPLE)
        expected = np.mean([abs(p["cohen_d"]) for p in result["pairwise"]])
        assert result["summary"]["mean_abs_cohen_d"] == pytest.approx(expected)


class TestReciprocalRanks:
    def test_a_top_scoring_gold_ranks_first(self):
        assert reciprocal_ranks([5.0], [[1.0, 2.0, 3.0]]).tolist() == [1.0]

    def test_a_bottom_scoring_gold_ranks_last(self):
        assert reciprocal_ranks([1.0], [[2.0, 3.0, 4.0]]).tolist() == [0.25]

    def test_a_tie_counts_in_the_gold_passages_favour(self):
        assert reciprocal_ranks([2.0], [[2.0, 2.0]]).tolist() == [1.0]

    def test_returns_one_value_per_query(self):
        result = reciprocal_ranks([1.0, 5.0], [[2.0, 3.0], [1.0, 2.0]])
        assert result.tolist() == [1 / 3, 1.0]


class TestBootstrapCiOfMean:
    def test_interval_values(self):
        assert bootstrap_ci_of_mean([*A, 5.0], n_boot=500) == (2.0, 4.2)

    def test_is_deterministic_for_a_given_seed(self):
        values = [*A, 5.0]
        assert bootstrap_ci_of_mean(values, n_boot=200, seed=3) == bootstrap_ci_of_mean(
            values, n_boot=200, seed=3
        )

    def test_a_different_seed_gives_a_different_interval(self):
        values = [float(i) for i in range(50)]
        assert bootstrap_ci_of_mean(values, n_boot=200, seed=1) != bootstrap_ci_of_mean(
            values, n_boot=200, seed=2
        )

    def test_the_lower_bound_does_not_exceed_the_upper_bound(self):
        lo, hi = bootstrap_ci_of_mean(A, n_boot=200)
        assert lo <= hi


class TestBootstrapCiOfMaxAbsD:
    def test_is_deterministic_for_a_given_seed(self):
        first = bootstrap_ci_of_max_abs_d(TWO, n_boot=200, seed=5, formats=PAIR)
        second = bootstrap_ci_of_max_abs_d(TWO, n_boot=200, seed=5, formats=PAIR)
        assert first == second

    def test_the_lower_bound_does_not_exceed_the_upper_bound(self):
        lo, hi = bootstrap_ci_of_max_abs_d(TWO, n_boot=200, formats=PAIR)
        assert lo <= hi

    def test_a_degenerate_resample_saturates_the_upper_bound(self):
        # A resample that draws one index repeatedly gives a zero standard
        # deviation, so the effect size reaches the 1 / COHEN_D_EPSILON ceiling.
        _, hi = bootstrap_ci_of_max_abs_d(TWO, n_boot=500, formats=PAIR)
        assert hi == pytest.approx(1e12)
