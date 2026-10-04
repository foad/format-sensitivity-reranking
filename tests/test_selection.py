from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fsr.formats import FORMAT_NAMES
from fsr.metrics import bootstrap_ci_of_mean
from fsr.selection import (
    LAMBDA,
    NI_MARGIN,
    RANK,
    SWEEPS,
    Sweep,
    candidate_row,
    cis_overlap,
    delta_mrr_ci,
    extract_value,
    mean_mrr_per_query,
    select,
    select_winner,
)


def guardrail(per_query):
    """Return a guardrail section whose formats all carry the same ranks."""
    return {"per_format_reciprocal_ranks": {f: list(per_query) for f in FORMAT_NAMES}}


def evaluation(path, ranks, max_abs_d, interval=None):
    payload = {
        "mrr_guardrail": guardrail(ranks),
        "format_sensitivity": {"summary": {"max_abs_cohen_d": max_abs_d}},
    }
    if interval is not None:
        payload["max_abs_d_ci95"] = list(interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


class TestExtractValue:
    def test_reads_a_weight(self):
        assert extract_value(Path("dev_cross_m_5fmt_lam0.01.json"), LAMBDA) == "0.01"

    def test_reads_a_whole_weight(self):
        assert extract_value(Path("dev_cross_m_5fmt_lam10.json"), LAMBDA) == "10"

    def test_reads_a_weight_before_a_suffix(self):
        assert extract_value(Path("dev_cross_m_5fmt_lam0.1_r8.json"), LAMBDA) == "0.1"

    def test_reads_a_rank(self):
        assert extract_value(Path("dev_cross_m_5fmt_lam1_r16.json"), RANK) == "16"

    def test_refuses_a_name_with_no_weight(self):
        with pytest.raises(ValueError, match="no lambda in"):
            extract_value(Path("dev_cross_m_base.json"), LAMBDA)

    def test_refuses_a_name_with_no_rank(self):
        with pytest.raises(ValueError, match="no rank in"):
            extract_value(Path("dev_cross_m_5fmt_lam1.json"), RANK)


class TestSweeps:
    def test_names_both_quantities(self):
        assert set(SWEEPS) == {"lambda", "rank"}

    def test_a_weight_sorts_as_a_number(self):
        assert LAMBDA.parse("10") > LAMBDA.parse("0.1")

    def test_a_rank_sorts_as_a_whole_number(self):
        assert RANK.parse("8") == 8

    def test_a_weight_tie_takes_the_middle(self):
        ordered = [{"lambda": "0"}, {"lambda": "0.1"}, {"lambda": "1"}]
        assert LAMBDA.pick(ordered)["lambda"] == "0.1"

    def test_an_even_weight_tie_takes_the_lower_middle(self):
        ordered = [{"lambda": "0"}, {"lambda": "0.1"}]
        assert LAMBDA.pick(ordered)["lambda"] == "0"

    def test_a_rank_tie_takes_the_smallest(self):
        ordered = [{"rank": "4"}, {"rank": "8"}, {"rank": "16"}]
        assert RANK.pick(ordered)["rank"] == "4"


class TestMeanMrrPerQuery:
    def test_averages_over_the_formats(self):
        ranks = {f: [1.0, 0.5] for f in FORMAT_NAMES}
        ranks[FORMAT_NAMES[0]] = [0.0, 0.5]
        out = mean_mrr_per_query({"per_format_reciprocal_ranks": ranks})
        assert out[0] == pytest.approx(4 / len(FORMAT_NAMES))
        assert out[1] == pytest.approx(0.5)

    def test_returns_one_value_per_query(self):
        assert len(mean_mrr_per_query(guardrail([1.0, 0.5, 0.25]))) == 3


class TestDeltaMrrCi:
    def test_reports_the_mean_change(self):
        base = np.array([0.5, 0.5, 0.5])
        trained = np.array([0.6, 0.7, 0.8])
        mean, _, _ = delta_mrr_ci(base, trained)
        assert mean == pytest.approx(0.2)

    def test_the_interval_brackets_the_mean(self):
        base = np.array([0.5] * 20)
        trained = np.array([0.6] * 10 + [0.4] * 10)
        mean, low, high = delta_mrr_ci(base, trained)
        assert low <= mean <= high

    def test_matches_the_shared_bootstrap(self):
        base = np.array([0.5, 0.4, 0.3, 0.9])
        trained = np.array([0.6, 0.2, 0.8, 0.1])
        _, low, high = delta_mrr_ci(base, trained, seed=7)
        assert (low, high) == bootstrap_ci_of_mean(trained - base, seed=7)

    def test_the_seed_fixes_the_interval(self):
        base = np.array([0.5, 0.4, 0.3, 0.9])
        trained = np.array([0.6, 0.2, 0.8, 0.1])
        assert delta_mrr_ci(base, trained, seed=1) == delta_mrr_ci(
            base, trained, seed=1
        )


class TestCisOverlap:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            ((0.0, 1.0), (0.5, 1.5), True),
            ((0.0, 1.0), (1.0, 2.0), True),
            ((0.0, 1.0), (1.1, 2.0), False),
            ((1.1, 2.0), (0.0, 1.0), False),
            ((0.0, 2.0), (0.5, 1.5), True),
            ((0.5, 0.5), (0.5, 0.5), True),
        ],
    )
    def test_reports_whether_they_meet(self, a, b, expected):
        assert cis_overlap(*a, *b) is expected


class TestCandidateRow:
    def test_summarises_a_candidate(self, tmp_path):
        rr_base = np.array([0.5, 0.5])
        path = evaluation(
            tmp_path / "dev_cross_m_5fmt_lam0.1.json", [0.6, 0.6], 0.3, (0.2, 0.4)
        )
        row = candidate_row(path, rr_base, LAMBDA)
        assert row["lambda"] == "0.1"
        assert row["dev_max_abs_d"] == 0.3
        assert row["dev_max_abs_d_ci_lo"] == 0.2
        assert row["delta_mrr_mean"] == pytest.approx(0.1)

    def test_passes_a_candidate_that_keeps_ranking_quality(self, tmp_path):
        rr_base = np.array([0.5] * 30)
        path = evaluation(
            tmp_path / "dev_cross_m_5fmt_lam0.json", [0.5] * 30, 0.3, (0.2, 0.4)
        )
        assert candidate_row(path, rr_base, LAMBDA)["mrr_ni_pass"]

    def test_fails_a_candidate_that_loses_ranking_quality(self, tmp_path):
        rr_base = np.array([0.9] * 30)
        path = evaluation(
            tmp_path / "dev_cross_m_5fmt_lam10.json", [0.1] * 30, 0.1, (0.0, 0.2)
        )
        assert not candidate_row(path, rr_base, LAMBDA)["mrr_ni_pass"]

    def test_uses_the_point_when_an_interval_is_missing(self, tmp_path):
        rr_base = np.array([0.5, 0.5])
        path = evaluation(tmp_path / "dev_cross_m_5fmt_lam1.json", [0.5, 0.5], 0.42)
        row = candidate_row(path, rr_base, LAMBDA)
        assert row["ci_missing"]
        assert row["dev_max_abs_d_ci_lo"] == row["dev_max_abs_d_ci_hi"] == 0.42

    def test_marks_an_interval_that_is_present(self, tmp_path):
        rr_base = np.array([0.5, 0.5])
        path = evaluation(
            tmp_path / "dev_cross_m_5fmt_lam1.json", [0.5, 0.5], 0.42, (0.3, 0.5)
        )
        assert not candidate_row(path, rr_base, LAMBDA)["ci_missing"]

    def test_refuses_a_candidate_on_different_queries(self, tmp_path):
        rr_base = np.array([0.5, 0.5, 0.5])
        path = evaluation(
            tmp_path / "dev_cross_m_5fmt_lam1.json", [0.5, 0.5], 0.4, (0.3, 0.5)
        )
        with pytest.raises(ValueError, match="covers 2 queries"):
            candidate_row(path, rr_base, LAMBDA)

    def test_the_margin_is_the_published_one(self):
        assert NI_MARGIN == 0.03


def row(value, max_abs_d, ci, delta_ci, field="lambda", passes=True):
    return {
        field: value,
        "dev_max_abs_d": max_abs_d,
        "dev_max_abs_d_ci_lo": ci[0],
        "dev_max_abs_d_ci_hi": ci[1],
        "delta_mrr_ci_lo": delta_ci[0],
        "delta_mrr_ci_hi": delta_ci[1],
        "mrr_ni_pass": passes,
    }


class TestSelectWinner:
    def test_takes_the_leader_when_nothing_ties_it(self):
        survivors = [
            row("0", 0.10, (0.05, 0.15), (-0.01, 0.01)),
            row("1", 0.90, (0.80, 1.00), (-0.01, 0.01)),
        ]
        winner, tied = select_winner(survivors, LAMBDA)
        assert winner["lambda"] == "0"
        assert len(tied) == 1

    def test_a_tie_needs_both_intervals_to_meet(self):
        survivors = [
            row("0", 0.10, (0.05, 0.15), (-0.01, 0.01)),
            row("1", 0.12, (0.06, 0.16), (0.50, 0.60)),
        ]
        _, tied = select_winner(survivors, LAMBDA)
        assert len(tied) == 1

    def test_a_three_way_weight_tie_takes_the_middle(self):
        survivors = [
            row("0", 0.10, (0.00, 1.00), (-0.01, 0.01)),
            row("0.1", 0.12, (0.00, 1.00), (-0.01, 0.01)),
            row("1", 0.14, (0.00, 1.00), (-0.01, 0.01)),
        ]
        winner, tied = select_winner(survivors, LAMBDA)
        assert len(tied) == 3
        assert winner["lambda"] == "0.1"

    def test_a_two_way_weight_tie_takes_the_smaller(self):
        survivors = [
            row("0.1", 0.12, (0.00, 1.00), (-0.01, 0.01)),
            row("10", 0.10, (0.00, 1.00), (-0.01, 0.01)),
        ]
        winner, _ = select_winner(survivors, LAMBDA)
        assert winner["lambda"] == "0.1"

    def test_weights_tie_in_numeric_order_not_text_order(self):
        survivors = [
            row("0.1", 0.10, (0.00, 1.00), (-0.01, 0.01)),
            row("2", 0.12, (0.00, 1.00), (-0.01, 0.01)),
            row("10", 0.14, (0.00, 1.00), (-0.01, 0.01)),
        ]
        winner, _ = select_winner(survivors, LAMBDA)
        assert winner["lambda"] == "2"

    def test_a_rank_tie_takes_the_smallest(self):
        survivors = [
            row("4", 0.14, (0.00, 1.00), (-0.01, 0.01), field="rank"),
            row("8", 0.10, (0.00, 1.00), (-0.01, 0.01), field="rank"),
            row("16", 0.12, (0.00, 1.00), (-0.01, 0.01), field="rank"),
        ]
        winner, tied = select_winner(survivors, RANK)
        assert len(tied) == 3
        assert winner["rank"] == "4"

    def test_the_leader_is_the_lowest_sensitivity(self):
        survivors = [
            row("0", 0.50, (0.49, 0.51), (-0.01, 0.01)),
            row("1", 0.20, (0.19, 0.21), (-0.01, 0.01)),
        ]
        winner, _ = select_winner(survivors, LAMBDA)
        assert winner["lambda"] == "1"


class TestSelect:
    def write_case(self, tmp_path, candidates, baseline_ranks=None):
        base = evaluation(
            tmp_path / "dev_cross_m_base.json",
            baseline_ranks or [0.5] * 30,
            0.80,
            (0.70, 0.90),
        )
        paths = []
        for value, ranks, max_abs_d, interval in candidates:
            paths.append(
                evaluation(
                    tmp_path / f"dev_cross_m_5fmt_lam{value}.json",
                    ranks,
                    max_abs_d,
                    interval,
                )
            )
        return base, paths

    def test_selects_a_winner(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [
                ("0", [0.5] * 30, 0.60, (0.55, 0.65)),
                ("0.1", [0.5] * 30, 0.20, (0.15, 0.25)),
            ],
        )
        out = select(base, paths, LAMBDA)
        assert out["status"] == "selected"
        assert out["winner_lambda"] == "0.1"

    def test_names_the_swept_quantity(self, tmp_path):
        base, paths = self.write_case(
            tmp_path, [("0.1", [0.5] * 30, 0.20, (0.15, 0.25))]
        )
        assert select(base, paths, LAMBDA)["swept"] == "lambda"

    def test_records_every_candidate(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [
                ("0", [0.5] * 30, 0.60, (0.55, 0.65)),
                ("0.1", [0.5] * 30, 0.20, (0.15, 0.25)),
            ],
        )
        assert len(select(base, paths, LAMBDA)["candidates"]) == 2

    def test_records_the_baseline(self, tmp_path):
        base, paths = self.write_case(
            tmp_path, [("0.1", [0.5] * 30, 0.20, (0.15, 0.25))]
        )
        out = select(base, paths, LAMBDA)
        assert out["baseline_max_abs_d"] == 0.80
        assert out["baseline_mean_mrr"] == pytest.approx(0.5)

    def test_reports_a_tie(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [
                ("0", [0.5] * 30, 0.60, (0.00, 1.00)),
                ("0.1", [0.5] * 30, 0.20, (0.00, 1.00)),
                ("1", [0.5] * 30, 0.40, (0.00, 1.00)),
            ],
        )
        out = select(base, paths, LAMBDA)
        assert out["tie_broken"]
        assert out["tied_lambdas"] == ["0", "0.1", "1"]
        assert out["winner_lambda"] == "0.1"

    def test_records_no_tied_group_when_the_leader_wins(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [
                ("0", [0.5] * 30, 0.60, (0.55, 0.65)),
                ("0.1", [0.5] * 30, 0.20, (0.15, 0.25)),
            ],
        )
        out = select(base, paths, LAMBDA)
        assert not out["tie_broken"]
        assert out["tied_lambdas"] == []

    def test_halts_when_every_candidate_degrades_ranking(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [("0", [0.1] * 30, 0.60, (0.55, 0.65))],
            baseline_ranks=[0.9] * 30,
        )
        out = select(base, paths, LAMBDA)
        assert out["status"] == "halt"
        assert out["reason"] == "no_lambda_passed_mrr_non_inferiority"

    def test_a_halt_still_records_the_candidates(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [("0", [0.1] * 30, 0.60, (0.55, 0.65))],
            baseline_ranks=[0.9] * 30,
        )
        assert select(base, paths, LAMBDA)["candidates"]

    def test_a_halt_names_the_swept_quantity(self, tmp_path):
        base, paths = self.write_case(
            tmp_path,
            [("0", [0.1] * 30, 0.60, (0.55, 0.65))],
            baseline_ranks=[0.9] * 30,
        )
        as_rank = Sweep(
            name="rank", pattern=LAMBDA.pattern, parse=LAMBDA.parse, pick=RANK.pick
        )
        out = select(base, paths, as_rank)
        assert out["reason"] == "no_rank_passed_mrr_non_inferiority"

    def test_carries_the_margin(self, tmp_path):
        base, paths = self.write_case(
            tmp_path, [("0.1", [0.5] * 30, 0.20, (0.15, 0.25))]
        )
        assert select(base, paths, LAMBDA)["ni_margin"] == NI_MARGIN
