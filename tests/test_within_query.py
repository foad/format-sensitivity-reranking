"""Tests for fsr.within_query."""

from __future__ import annotations

import math

import numpy as np
import pytest

from fsr.within_query import (
    bootstrap_ci_of_inconsistency,
    bootstrap_tables,
    conditional_inconsistency,
    conditional_inconsistency_from_matrices,
    gold_leads,
    gold_top1_from_matrices,
    gold_top1_stability,
    score_scale_diagnostic,
    within_query_mrr,
    within_query_rank_stability,
)

PAIR = ("yaml", "json")
TRIPLE = ("yaml", "json", "toml")


def mats(**kwargs) -> dict[str, np.ndarray]:
    """Return candidate matrices, one per named format."""
    return {k: np.array(v, dtype=float) for k, v in kwargs.items()}


class TestGoldLeads:
    def test_marks_a_leading_gold(self):
        out = gold_leads(mats(yaml=[[9.0, 1.0, 2.0]]))
        assert out["yaml"].tolist() == [True]

    def test_marks_an_outscored_gold(self):
        out = gold_leads(mats(yaml=[[1.0, 9.0]]))
        assert out["yaml"].tolist() == [False]

    def test_a_tie_counts_in_the_golds_favour(self):
        assert gold_leads(mats(yaml=[[5.0, 5.0]]))["yaml"].tolist() == [True]

    def test_honours_a_different_gold_column(self):
        out = gold_leads(mats(yaml=[[1.0, 9.0]]), gold_col=1)
        assert out["yaml"].tolist() == [True]


class TestConditionalInconsistency:
    def test_a_query_leading_everywhere_is_consistent(self):
        out = conditional_inconsistency({f: np.array([True]) for f in PAIR}, PAIR)
        assert out["inconsistent"] == 0
        assert out["inconsistency_pct"] == 0.0

    def test_a_query_leading_under_some_formats_is_inconsistent(self):
        out = conditional_inconsistency(
            {"yaml": np.array([True]), "json": np.array([False])}, PAIR
        )
        assert out["n_answerable"] == 1
        assert out["inconsistency_pct"] == 100.0

    def test_a_query_leading_nowhere_is_excluded(self):
        out = conditional_inconsistency(
            {"yaml": np.array([True, False]), "json": np.array([False, False])}, PAIR
        )
        assert out["n_answerable"] == 1
        assert out["answerable_pct"] == 50.0
        assert out["inconsistency_pct"] == 100.0

    def test_reports_undefined_when_nothing_is_answerable(self):
        out = conditional_inconsistency(
            {f: np.array([False, False]) for f in PAIR}, PAIR
        )
        assert out["n_answerable"] == 0
        assert math.isnan(out["inconsistency_pct"])

    def test_ignores_a_format_that_is_absent(self):
        out = conditional_inconsistency({"yaml": np.array([True])}, TRIPLE)
        assert out["inconsistency_pct"] == 0.0

    def test_matches_the_matrix_form(self):
        m = mats(yaml=[[9.0, 1.0], [1.0, 9.0]], json=[[1.0, 9.0], [1.0, 9.0]])
        assert conditional_inconsistency_from_matrices(m, formats=PAIR) == (
            conditional_inconsistency(gold_leads(m), PAIR)
        )


class TestGoldTop1Stability:
    def test_splits_queries_three_ways(self):
        top1 = {
            "yaml": np.array([True, True, False]),
            "json": np.array([True, False, False]),
        }
        out = gold_top1_stability(top1, PAIR)
        assert out["gold_top1_all_formats"] == 1
        assert out["gold_top1_format_dependent"] == 1
        assert out["gold_top1_never"] == 1

    def test_the_three_shares_sum_to_one_hundred(self):
        top1 = {"yaml": np.array([True, False]), "json": np.array([True, True])}
        out = gold_top1_stability(top1, PAIR)
        total = (
            out["gold_top1_all_formats_pct"]
            + out["gold_top1_format_dependent_pct"]
            + out["gold_top1_never_pct"]
        )
        assert total == pytest.approx(100.0)

    def test_reports_the_worst_disagreeing_pair(self):
        top1 = {
            "yaml": np.array([True, True]),
            "json": np.array([True, True]),
            "toml": np.array([False, False]),
        }
        out = gold_top1_stability(top1, TRIPLE)
        assert out["worst_pair_disagreement_pct"] == 100.0
        assert out["worst_pair"] in {"yaml vs toml", "json vs toml"}

    def test_reports_the_per_format_share(self):
        top1 = {"yaml": np.array([True, False]), "json": np.array([True, True])}
        out = gold_top1_stability(top1, PAIR)
        assert out["per_format_top1_pct"] == {"yaml": 50.0, "json": 100.0}

    def test_matches_the_matrix_form(self):
        m = mats(yaml=[[9.0, 1.0]], json=[[1.0, 9.0]])
        assert gold_top1_from_matrices(m, formats=PAIR) == gold_top1_stability(
            gold_leads(m), PAIR
        )


class TestWithinQueryRankStability:
    IDENTICAL = mats(yaml=[[3.0, 2.0, 1.0]], json=[[3.0, 2.0, 1.0]])
    REVERSED = mats(yaml=[[3.0, 2.0, 1.0]], json=[[1.0, 2.0, 3.0]])

    def test_identical_orderings_never_flip(self):
        per_pair, summary, _ = within_query_rank_stability(self.IDENTICAL, formats=PAIR)
        assert per_pair[0]["mean_flip_rate_pct"] == 0.0
        assert summary["min_kendall_tau"] == pytest.approx(1.0)

    def test_a_reversed_ordering_flips_every_pair(self):
        per_pair, summary, _ = within_query_rank_stability(self.REVERSED, formats=PAIR)
        assert per_pair[0]["mean_flip_rate_pct"] == 100.0
        assert summary["min_kendall_tau"] == pytest.approx(-1.0)

    def test_reports_the_shape_of_the_candidate_set(self):
        _, summary, _ = within_query_rank_stability(self.IDENTICAL, formats=PAIR)
        assert (summary["n_queries"], summary["n_candidates"]) == (1, 3)

    def test_tracks_the_gold_rank_change(self):
        per_pair, _, _ = within_query_rank_stability(self.REVERSED, formats=PAIR)
        assert per_pair[0]["mean_gold_rank_delta"] == 2.0
        assert per_pair[0]["gold_rank_changed_pct"] == 100.0

    def test_tracks_the_top_one_change(self):
        per_pair, _, _ = within_query_rank_stability(self.REVERSED, formats=PAIR)
        assert per_pair[0]["top1_changed_pct"] == 100.0

    @pytest.mark.filterwarnings("ignore:Mean of empty slice")
    def test_an_all_tied_query_has_no_comparable_pair(self):
        tied = mats(yaml=[[1.0, 1.0, 1.0]], json=[[1.0, 1.0, 1.0]])
        per_pair, _, per_query = within_query_rank_stability(tied, formats=PAIR)
        assert math.isnan(per_query["yaml vs json"]["flip_rate_pct"][0])
        assert math.isnan(per_pair[0]["mean_flip_rate_pct"])

    def test_returns_one_entry_per_format_pair(self):
        m = mats(yaml=[[3.0, 1.0]], json=[[3.0, 1.0]], toml=[[1.0, 3.0]])
        per_pair, _, _ = within_query_rank_stability(m, formats=TRIPLE)
        assert [e["pair"] for e in per_pair] == [
            "yaml vs json",
            "yaml vs toml",
            "json vs toml",
        ]

    def test_records_a_per_query_series(self):
        m = mats(yaml=[[3.0, 1.0], [1.0, 3.0]], json=[[3.0, 1.0], [3.0, 1.0]])
        _, _, per_query = within_query_rank_stability(m, formats=PAIR)
        assert len(per_query["yaml vs json"]["flip_rate_pct"]) == 2


class TestWithinQueryMrr:
    def test_a_leading_gold_scores_one(self):
        out = within_query_mrr(mats(yaml=[[9.0, 1.0, 2.0]]))
        assert out["yaml"]["mrr"] == 1.0

    def test_a_trailing_gold_scores_its_reciprocal_rank(self):
        out = within_query_mrr(mats(yaml=[[1.0, 9.0, 8.0]]))
        assert out["yaml"]["mrr"] == pytest.approx(1 / 3)

    def test_returns_the_per_query_series(self):
        out = within_query_mrr(mats(yaml=[[9.0, 1.0], [1.0, 9.0]]))
        assert out["yaml"]["reciprocal_ranks"] == [1.0, 0.5]


class TestScoreScaleDiagnostic:
    def test_identical_formats_give_no_cross_format_spread(self):
        m = mats(yaml=[[1.0, 2.0, 3.0]], json=[[1.0, 2.0, 3.0]])
        out = score_scale_diagnostic(m, formats=PAIR)
        assert out["delta"] == 0.0
        assert out["ratio"] == 0.0

    def test_reports_the_within_query_spread(self):
        m = mats(yaml=[[0.0, 2.0]], json=[[0.0, 2.0]])
        assert score_scale_diagnostic(m, formats=PAIR)["s_within"] == pytest.approx(1.0)

    def test_the_ratio_rises_with_cross_format_spread(self):
        near = mats(yaml=[[0.0, 2.0]], json=[[0.1, 2.1]])
        far = mats(yaml=[[0.0, 2.0]], json=[[5.0, 7.0]])
        assert (
            score_scale_diagnostic(far, formats=PAIR)["ratio"]
            > score_scale_diagnostic(near, formats=PAIR)["ratio"]
        )


BOOTSTRAP_FORMATS = ["yaml", "json", "toml"]
BOOTSTRAP_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L6-v2",
    "BAAI/bge-reranker-base",
]


def bootstrap_entry(seed=0, n_queries=8):
    rng = np.random.default_rng(seed)
    pair = "yaml vs json"
    return {
        "reciprocal_ranks": {
            f: rng.choice([1.0, 0.5, 0.25], size=n_queries).tolist()
            for f in BOOTSTRAP_FORMATS
        },
        "within_query": {"summary": {"max_flip_pair": pair}},
        "per_query": {
            pair: {
                "flip_rate_pct": rng.uniform(0, 30, size=n_queries).tolist(),
                "kendall_tau": rng.uniform(0.5, 1.0, size=n_queries).tolist(),
            }
        },
    }


class TestBootstrapTables:
    def _tables(self, n_boot=200):
        results = {m: bootstrap_entry(i) for i, m in enumerate(BOOTSTRAP_MODELS)}
        return bootstrap_tables(results, BOOTSTRAP_FORMATS, n_boot=n_boot, seed=1)

    def test_reports_every_model(self):
        per_model, _, models, _ = self._tables()
        assert set(per_model) == set(BOOTSTRAP_MODELS)
        assert models == BOOTSTRAP_MODELS

    def test_reports_an_interval_for_each_statistic(self):
        per_model, _, _, _ = self._tables()
        value = per_model[BOOTSTRAP_MODELS[0]]
        for key in ("format_dependent_ci", "flip_rate_ci", "kendall_tau_ci"):
            low, high = value[key]
            assert low <= high

    def test_orders_by_format_dependent_share(self):
        per_model, _, models, order = self._tables()
        shares = [per_model[models[i]]["format_dependent_pct"] for i in order]
        assert shares == sorted(shares, reverse=True)

    def test_reports_adjacent_differences(self):
        _, diffs, _, _ = self._tables()
        assert len(diffs) == len(BOOTSTRAP_MODELS) - 1
        assert "separated" in diffs[0]

    def test_labels_the_compared_models(self):
        _, diffs, _, _ = self._tables()
        assert diffs[0]["pair"] in {"MiniLM-L6 - bge-base", "bge-base - MiniLM-L6"}

    def test_separated_follows_the_interval(self):
        _, diffs, _, _ = self._tables()
        low, high = diffs[0]["ci"]
        assert diffs[0]["separated"] == bool(low > 0 or high < 0)

    def test_names_the_worst_pair(self):
        per_model, _, _, _ = self._tables()
        assert per_model[BOOTSTRAP_MODELS[0]]["worst_pair"] == "yaml vs json"

    def test_is_deterministic_for_a_seed(self):
        first, diffs, models, order = self._tables()
        again, diffs_again, models_again, order_again = self._tables()
        assert first == again
        assert diffs == diffs_again
        assert models == models_again
        assert order.tolist() == order_again.tolist()

    def test_a_wider_interval_is_not_narrower(self):
        results = {m: bootstrap_entry(i) for i, m in enumerate(BOOTSTRAP_MODELS)}
        narrow, _, _, _ = bootstrap_tables(
            results, BOOTSTRAP_FORMATS, n_boot=200, seed=1, ci=0.50
        )
        wide, _, _, _ = bootstrap_tables(
            results, BOOTSTRAP_FORMATS, n_boot=200, seed=1, ci=0.99
        )
        model = BOOTSTRAP_MODELS[0]
        narrow_low, narrow_high = narrow[model]["format_dependent_ci"]
        wide_low, wide_high = wide[model]["format_dependent_ci"]
        assert wide_high - wide_low >= narrow_high - narrow_low


class TestBootstrapCiOfInconsistency:
    def _top1(self, n=200, seed=0):
        rng = np.random.default_rng(seed)
        base = rng.random(n) < 0.8
        return {f: base & (rng.random(n) < 0.9) for f in ("yaml", "json", "toml")}

    def test_brackets_the_point_estimate(self):
        top1 = self._top1()
        point = conditional_inconsistency(top1, formats=list(top1))["inconsistency_pct"]
        low, high = bootstrap_ci_of_inconsistency(top1, n_boot=500, formats=list(top1))
        assert low <= point <= high

    def test_a_wider_interval_is_not_narrower(self):
        top1 = self._top1()
        narrow = bootstrap_ci_of_inconsistency(
            top1, n_boot=500, ci=0.50, formats=list(top1)
        )
        wide = bootstrap_ci_of_inconsistency(
            top1, n_boot=500, ci=0.99, formats=list(top1)
        )
        assert wide[1] - wide[0] >= narrow[1] - narrow[0]

    def test_is_deterministic_for_a_seed(self):
        top1 = self._top1()
        first = bootstrap_ci_of_inconsistency(top1, n_boot=200, formats=list(top1))
        again = bootstrap_ci_of_inconsistency(top1, n_boot=200, formats=list(top1))
        assert first == again

    def test_perfect_consistency_gives_a_zero_interval(self):
        top1 = {f: np.ones(50, dtype=bool) for f in ("yaml", "json")}
        assert bootstrap_ci_of_inconsistency(top1, n_boot=200, formats=list(top1)) == (
            0.0,
            0.0,
        )

    def test_nothing_answerable_gives_nan(self):
        top1 = {f: np.zeros(40, dtype=bool) for f in ("yaml", "json")}
        low, high = bootstrap_ci_of_inconsistency(top1, n_boot=200, formats=list(top1))
        assert math.isnan(low) and math.isnan(high)

    def test_ignores_a_format_not_asked_for(self):
        top1 = self._top1()
        top1["markdown"] = np.zeros(200, dtype=bool)
        paired = bootstrap_ci_of_inconsistency(
            top1, n_boot=200, formats=["yaml", "json", "toml"]
        )
        assert paired == bootstrap_ci_of_inconsistency(
            self._top1(), n_boot=200, formats=["yaml", "json", "toml"]
        )
