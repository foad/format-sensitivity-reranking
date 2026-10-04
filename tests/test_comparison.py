from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from fsr.comparison import (
    NI_MARGIN,
    PRIMARY_THRESHOLD,
    STRETCH_THRESHOLD,
    all_pair_deltas,
    bootstrap_delta_max_d,
    bootstrap_delta_mrr,
    cohen_band,
    delta_mrr_ci,
    max_abs_d_over_pairs,
    mean_mrr_per_query,
)
from fsr.formats import FORMAT_NAMES
from fsr.metrics import bootstrap_ci_of_mean, cohen_d

PAIRS = list(combinations(FORMAT_NAMES, 2))


def scores(offsets, n=40, seed=0):
    """Return per-format scores whose means differ by the given offsets."""
    rng = np.random.default_rng(seed)
    return {
        name: (rng.normal(0, 1.0, size=n) + offsets.get(name, 0.0)).tolist()
        for name in FORMAT_NAMES
    }


class TestCohenBand:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0.0, "trivial"),
            (0.19, "trivial"),
            (0.2, "small"),
            (0.49, "small"),
            (0.5, "medium"),
            (0.79, "medium"),
            (0.8, "large"),
            (2.0, "large"),
        ],
    )
    def test_names_the_band(self, value, expected):
        assert cohen_band(value) == expected

    def test_ignores_the_sign(self):
        assert cohen_band(-0.9) == cohen_band(0.9) == "large"

    def test_the_thresholds_are_the_published_ones(self):
        assert PRIMARY_THRESHOLD == 0.5
        assert STRETCH_THRESHOLD == 0.2
        assert NI_MARGIN == 0.03


class TestMaxAbsDOverPairs:
    def test_finds_the_largest_pair(self):
        per_format = scores({FORMAT_NAMES[0]: 3.0})
        largest, pair = max_abs_d_over_pairs(per_format, PAIRS)
        assert FORMAT_NAMES[0] in pair
        assert largest > 1.0

    def test_reports_no_pair_when_every_format_agrees(self):
        per_format = {name: [1.0, 2.0, 3.0] for name in FORMAT_NAMES}
        assert max_abs_d_over_pairs(per_format, PAIRS) == (0.0, None)

    def test_restricts_to_the_given_records(self):
        per_format = scores({FORMAT_NAMES[0]: 3.0}, n=40)
        whole, _ = max_abs_d_over_pairs(per_format, PAIRS)
        part, _ = max_abs_d_over_pairs(per_format, PAIRS, np.arange(10))
        assert whole != part

    def test_an_index_may_repeat_a_record(self):
        per_format = scores({FORMAT_NAMES[0]: 2.0})
        repeated = np.zeros(20, dtype=int)
        largest, _ = max_abs_d_over_pairs(per_format, PAIRS, repeated)
        assert largest >= 0.0

    def test_covers_only_the_pairs_it_is_given(self):
        per_format = scores({FORMAT_NAMES[0]: 3.0})
        without = [p for p in PAIRS if FORMAT_NAMES[0] not in p]
        largest, pair = max_abs_d_over_pairs(per_format, without)
        assert FORMAT_NAMES[0] not in (pair or "")
        assert largest < 1.0


class TestAllPairDeltas:
    def test_covers_every_pair(self):
        rows = all_pair_deltas(scores({}), scores({}, seed=1), PAIRS)
        assert [r["pair"] for r in rows] == [f"{a} - {b}" for a, b in PAIRS]

    def test_reports_both_effect_sizes(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.1}, seed=1)
        row = all_pair_deltas(base, trained, PAIRS)[0]
        assert row["d_baseline"] == cohen_d(base[PAIRS[0][0]], base[PAIRS[0][1]])
        assert row["d_trained"] == cohen_d(trained[PAIRS[0][0]], trained[PAIRS[0][1]])

    def test_the_change_is_in_magnitude(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: -2.0}, seed=0)
        row = all_pair_deltas(base, trained, PAIRS)[0]
        assert row["delta_abs_d"] == pytest.approx(
            row["abs_d_trained"] - row["abs_d_baseline"]
        )

    def test_a_sign_flip_of_equal_size_is_no_change(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = {name: [-v for v in values] for name, values in base.items()}
        row = all_pair_deltas(base, trained, PAIRS)[0]
        assert row["delta_abs_d"] == pytest.approx(0.0)

    def test_training_that_removes_the_effect_reports_a_fall(self):
        base = scores({FORMAT_NAMES[0]: 3.0})
        trained = scores({})
        row = all_pair_deltas(base, trained, PAIRS)[0]
        assert row["delta_abs_d"] < 0


class TestBootstrapDeltaMaxD:
    def test_reports_the_change_and_both_runs(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.2}, seed=1)
        out = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=200)
        assert set(out) == {
            "delta_ci",
            "delta_mean",
            "baseline_max_d_ci",
            "trained_max_d_ci",
        }

    def test_the_interval_brackets_the_mean(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.2}, seed=1)
        out = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=200)
        low, high = out["delta_ci"]
        assert low <= out["delta_mean"] <= high

    def test_reports_a_fall_when_training_removes_the_effect(self):
        base = scores({FORMAT_NAMES[0]: 3.0})
        trained = scores({})
        out = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=200)
        assert out["delta_mean"] < 0
        assert out["delta_ci"][1] < 0

    def test_reports_no_change_when_the_runs_match(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        out = bootstrap_delta_max_d(base, base, PAIRS, n_boot=200)
        assert out["delta_mean"] == pytest.approx(0.0)
        assert out["delta_ci"] == [pytest.approx(0.0), pytest.approx(0.0)]

    def test_pairs_the_two_runs_on_one_resample(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        out = bootstrap_delta_max_d(base, base, PAIRS, n_boot=200)
        assert out["baseline_max_d_ci"] == out["trained_max_d_ci"]

    def test_the_seed_fixes_the_intervals(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.2}, seed=1)
        first = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=100, seed=3)
        second = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=100, seed=3)
        assert first == second

    def test_a_different_seed_gives_a_different_interval(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.2}, seed=1)
        first = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=100, seed=3)
        second = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=100, seed=4)
        assert first["delta_ci"] != second["delta_ci"]

    def test_a_wider_interval_reaches_further(self):
        base = scores({FORMAT_NAMES[0]: 2.0})
        trained = scores({FORMAT_NAMES[0]: 0.2}, seed=1)
        narrow = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=300, ci=0.50)
        wide = bootstrap_delta_max_d(base, trained, PAIRS, n_boot=300, ci=0.99)
        assert wide["delta_ci"][0] <= narrow["delta_ci"][0]
        assert wide["delta_ci"][1] >= narrow["delta_ci"][1]


def guardrail(per_query):
    return {"per_format_reciprocal_ranks": {f: list(per_query) for f in FORMAT_NAMES}}


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
        mean, _, _ = delta_mrr_ci(np.array([0.5] * 3), np.array([0.6, 0.7, 0.8]))
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


class TestBootstrapDeltaMrr:
    def test_reports_the_change_as_a_section(self):
        base = np.array([0.5] * 10)
        trained = np.array([0.6] * 10)
        out = bootstrap_delta_mrr(base, trained)
        assert set(out) == {"delta_mean", "delta_ci"}
        assert out["delta_mean"] == pytest.approx(0.1)

    def test_carries_the_same_interval_as_the_pair_form(self):
        base = np.array([0.5, 0.4, 0.3, 0.9])
        trained = np.array([0.6, 0.2, 0.8, 0.1])
        mean, low, high = delta_mrr_ci(base, trained, seed=2)
        out = bootstrap_delta_mrr(base, trained, seed=2)
        assert out == {"delta_mean": mean, "delta_ci": [low, high]}
