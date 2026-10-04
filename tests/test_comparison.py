from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from fsr.comparison import (
    ALL_FORMATS,
    IN_TRAINING,
    MIN_SUBSET_RECORDS,
    NI_MARGIN,
    OOD,
    PRIMARY_THRESHOLD,
    STRETCH_THRESHOLD,
    SUBSET_NAMES,
    all_pair_deltas,
    bands_crossed,
    bootstrap_delta_max_d,
    bootstrap_delta_mrr,
    cohen_band,
    compare,
    delta_mrr_ci,
    held_out_transfer,
    max_abs_d_over_pairs,
    max_d_section,
    mean_mrr_per_query,
    pair_subsets,
    paired_reciprocal_ranks,
    relative_change,
    slice_scores,
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


class TestPairSubsets:
    def test_covers_every_pair(self):
        subsets = pair_subsets("yaml")
        assert len(subsets[ALL_FORMATS]) == len(PAIRS)

    def test_splits_the_pairs_in_two(self):
        subsets = pair_subsets("yaml")
        assert len(subsets[IN_TRAINING]) + len(subsets[OOD]) == len(PAIRS)

    def test_the_held_out_pairs_all_involve_it(self):
        assert all("yaml" in p for p in pair_subsets("yaml")[OOD])

    def test_the_training_pairs_never_involve_it(self):
        assert all("yaml" not in p for p in pair_subsets("yaml")[IN_TRAINING])

    def test_four_pairs_involve_the_held_out_format(self):
        assert len(pair_subsets("toml")[OOD]) == len(FORMAT_NAMES) - 1

    def test_the_names_are_the_published_ones(self):
        assert SUBSET_NAMES == (ALL_FORMATS, IN_TRAINING, OOD)


class TestBandsCrossed:
    def test_a_fall_of_one_band_counts_minus_one(self):
        assert bands_crossed(0.9, 0.6) == -1

    def test_a_fall_of_three_bands_counts_minus_three(self):
        assert bands_crossed(0.9, 0.1) == -3

    def test_no_change_counts_zero(self):
        assert bands_crossed(0.9, 0.85) == 0

    def test_a_rise_counts_positive(self):
        assert bands_crossed(0.1, 0.9) == 3


class TestRelativeChange:
    def test_reports_the_share_of_the_baseline(self):
        assert relative_change(0.8, 0.4) == pytest.approx(-0.5)

    def test_returns_none_at_a_zero_baseline(self):
        assert relative_change(0.0, 0.4) is None


class TestSliceScores:
    def test_keeps_the_given_positions(self):
        per_format = {name: [0.0, 1.0, 2.0, 3.0] for name in FORMAT_NAMES}
        sliced = slice_scores(per_format, [1, 3])
        assert sliced[FORMAT_NAMES[0]] == [1.0, 3.0]

    def test_covers_every_format(self):
        per_format = {name: [0.0, 1.0] for name in FORMAT_NAMES}
        assert set(slice_scores(per_format, [0])) == set(FORMAT_NAMES)


class TestMaxDSection:
    def test_reports_both_runs_with_their_bands(self):
        base = scores({FORMAT_NAMES[0]: 3.0})
        trained = scores({}, seed=1)
        section = max_d_section(base, trained, PAIRS)
        assert section["baseline"] > section["trained"]
        assert section["baseline_band"] == "large"

    def test_names_the_pair_behind_each(self):
        base = scores({FORMAT_NAMES[0]: 3.0})
        section = max_d_section(base, base, PAIRS)
        assert section["baseline_pair"] == section["trained_pair"]


def evaluation(offsets, n=40, seed=0, ranks=None, ids=None):
    per_format = scores(offsets, n=n, seed=seed)
    record_ids = ids or [f"r{i}" for i in range(n)]
    rr = ranks if ranks is not None else [0.5] * n
    return {
        "n_records_kept": n,
        "scores_per_fmt": per_format,
        "record_ids": record_ids,
        "mrr_guardrail": {
            "per_format_reciprocal_ranks": {f: list(rr) for f in FORMAT_NAMES},
            "record_ids": record_ids,
        },
    }


class TestHeldOutTransfer:
    def test_reports_the_change_on_the_held_out_format(self):
        base = evaluation({}, ranks=[0.4] * 30)
        trained = evaluation({}, ranks=[0.6] * 30, seed=1)
        out = held_out_transfer(base["mrr_guardrail"], trained["mrr_guardrail"], "yaml")
        assert out["delta_mrr"] == pytest.approx(0.2)
        assert out["held_out_format"] == "yaml"

    def test_covers_every_training_format(self):
        base = evaluation({}, ranks=[0.4] * 30)
        trained = evaluation({}, ranks=[0.6] * 30, seed=1)
        out = held_out_transfer(base["mrr_guardrail"], trained["mrr_guardrail"], "yaml")
        assert set(out["training_format_delta_mrr_per_fmt"]) == set(FORMAT_NAMES) - {
            "yaml"
        }

    def test_a_matched_change_gives_a_ratio_of_one(self):
        base = evaluation({}, ranks=[0.4] * 30)
        trained = evaluation({}, ranks=[0.6] * 30, seed=1)
        out = held_out_transfer(base["mrr_guardrail"], trained["mrr_guardrail"], "yaml")
        assert out["transfer_ratio"] == pytest.approx(1.0)

    def test_an_unmoved_training_set_gives_no_ratio(self):
        base = evaluation({}, ranks=[0.5] * 30)
        out = held_out_transfer(base["mrr_guardrail"], base["mrr_guardrail"], "yaml")
        assert np.isnan(out["transfer_ratio"])

    def test_restricts_to_the_given_queries(self):
        base = evaluation({}, ranks=[0.4] * 10 + [0.9] * 20)
        trained = evaluation({}, ranks=[0.6] * 10 + [0.9] * 20, seed=1)
        whole = held_out_transfer(
            base["mrr_guardrail"], trained["mrr_guardrail"], "yaml"
        )
        part = held_out_transfer(
            base["mrr_guardrail"], trained["mrr_guardrail"], "yaml", 0, range(10)
        )
        assert part["delta_mrr"] > whole["delta_mrr"]


class TestCompare:
    def result(self, **kwargs):
        base = evaluation({FORMAT_NAMES[0]: 3.0}, ranks=[0.5] * 40)
        trained = evaluation({}, seed=1, ranks=[0.5] * 40)
        return compare(base, trained, FORMAT_NAMES[0], n_boot=100, **kwargs)

    def test_covers_every_subset(self):
        out = self.result()
        assert set(out["point_estimates_max_d"]) == set(SUBSET_NAMES)
        assert set(out["bootstrap_delta_max_d"]) == set(SUBSET_NAMES)

    def test_reports_one_row_per_pair(self):
        assert len(self.result()["per_pair_delta"]) == len(PAIRS)

    def test_carries_the_held_out_format(self):
        assert self.result()["held_out_format"] == FORMAT_NAMES[0]

    def test_applies_the_thresholds_to_the_training_pairs(self):
        out = self.result()
        verdict = out["in_training_verdict"]
        trained = out["point_estimates_max_d"][IN_TRAINING]["trained"]
        assert verdict["primary_pass"] == (trained < PRIMARY_THRESHOLD)
        assert verdict["stretch_pass"] == (trained < STRETCH_THRESHOLD)

    def test_describes_the_held_out_pairs_without_a_threshold(self):
        descriptive = self.result()["ood_descriptive"]
        assert "cohen_bands_crossed" in descriptive
        assert "primary_pass" not in descriptive

    def test_reports_the_ranking_guardrail(self):
        guardrail = self.result()["mrr_non_inferiority"]
        assert guardrail["margin"] == NI_MARGIN
        assert guardrail["passed"] is True

    def test_reports_the_transfer(self):
        assert self.result()["heldout_transfer"]["held_out_format"] == FORMAT_NAMES[0]

    def test_skips_the_subset_when_none_is_given(self):
        assert self.result()["metadata_only_subset"] is None

    def test_refuses_evaluations_of_different_sizes(self):
        base = evaluation({}, n=40)
        trained = evaluation({}, n=30, seed=1)
        with pytest.raises(ValueError, match="kept 40 records"):
            compare(base, trained, "yaml", n_boot=10)

    def test_reports_no_guardrail_when_one_is_absent(self):
        base = evaluation({}, ranks=[0.5] * 40)
        trained = evaluation({}, seed=1, ranks=[0.5] * 40)
        trained["mrr_guardrail"] = None
        out = compare(base, trained, "yaml", n_boot=10)
        assert out["mrr_non_inferiority"]["passed"] is None
        assert out["heldout_transfer"] is None

    def test_refuses_guardrails_of_different_sizes(self):
        base = evaluation({}, n=40, ranks=[0.5] * 40)
        trained = evaluation({}, n=40, seed=1, ranks=[0.5] * 30)
        with pytest.raises(ValueError, match="40 and 30 queries"):
            compare(base, trained, "yaml", n_boot=10)


class TestMetadataOnlySubset:
    def pair(self, n=40, ranks=None):
        base = evaluation({FORMAT_NAMES[0]: 3.0}, n=n, ranks=ranks or [0.4] * n)
        trained = evaluation({}, n=n, seed=1, ranks=ranks or [0.6] * n)
        return base, trained

    def test_measures_the_named_records(self):
        base, trained = self.pair()
        ids = set(base["record_ids"][:25])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)[
            "metadata_only_subset"
        ]
        assert out["n_records"] == 25

    def test_reports_both_pair_subsets(self):
        base, trained = self.pair()
        ids = set(base["record_ids"][:25])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)[
            "metadata_only_subset"
        ]
        assert set(out["point_estimates_max_d"]) == {IN_TRAINING, OOD}

    def test_skips_a_subset_that_is_too_small(self):
        base, trained = self.pair()
        ids = set(base["record_ids"][: MIN_SUBSET_RECORDS - 1])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)
        assert out["metadata_only_subset"] is None

    def test_measures_ranking_on_the_subset(self):
        base, trained = self.pair()
        ids = set(base["record_ids"][:25])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)[
            "metadata_only_subset"
        ]
        assert out["mrr"]["n_records"] == 25
        assert out["mrr"]["overall_delta_mrr_bootstrap"]["delta_mean"] > 0

    def test_skips_ranking_when_the_guardrail_covers_too_few(self):
        base, trained = self.pair()
        for payload in (base, trained):
            payload["mrr_guardrail"]["record_ids"] = ["x"] * 40
        ids = set(base["record_ids"][:25])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)[
            "metadata_only_subset"
        ]
        assert out["mrr"] is None

    def test_skips_ranking_when_a_guardrail_is_absent(self):
        base, trained = self.pair()
        trained["mrr_guardrail"] = None
        ids = set(base["record_ids"][:25])
        out = compare(base, trained, FORMAT_NAMES[0], n_boot=50, metadata_only=ids)[
            "metadata_only_subset"
        ]
        assert out["mrr"] is None


class TestPairedReciprocalRanks:
    def ranking(self, ids, ranks):
        return {"record_ids": list(ids), "reciprocal_ranks": list(ranks)}

    def test_lines_up_matching_rankings(self):
        base = self.ranking(["a", "b"], [0.5, 0.25])
        trained = self.ranking(["a", "b"], [1.0, 0.5])
        shared, rr_base, rr_trained = paired_reciprocal_ranks(base, trained)
        assert shared == ["a", "b"]
        assert rr_base.tolist() == [0.5, 0.25]
        assert rr_trained.tolist() == [1.0, 0.5]

    def test_keeps_the_baseline_order(self):
        base = self.ranking(["b", "a"], [0.25, 0.5])
        trained = self.ranking(["a", "b"], [1.0, 0.5])
        shared, _, rr_trained = paired_reciprocal_ranks(base, trained)
        assert shared == ["b", "a"]
        assert rr_trained.tolist() == [0.5, 1.0]

    def test_keeps_only_the_shared_records(self):
        base = self.ranking(["a", "b", "c"], [0.5, 0.25, 1.0])
        trained = self.ranking(["b", "c"], [0.5, 1.0])
        shared, rr_base, _ = paired_reciprocal_ranks(base, trained)
        assert shared == ["b", "c"]
        assert rr_base.tolist() == [0.25, 1.0]

    def test_refuses_rankings_with_nothing_in_common(self):
        base = self.ranking(["a"], [0.5])
        trained = self.ranking(["b"], [0.5])
        with pytest.raises(ValueError, match="share no record"):
            paired_reciprocal_ranks(base, trained)


class TestResampleCount:
    def test_a_smaller_count_still_brackets_the_change(self):
        base = np.array([0.5] * 40)
        trained = np.array([0.6] * 20 + [0.4] * 20)
        mean, low, high = delta_mrr_ci(base, trained, n_boot=50)
        assert low <= mean <= high

    def test_the_count_changes_the_interval(self):
        rng = np.random.default_rng(0)
        base = rng.random(40)
        trained = rng.random(40)
        assert delta_mrr_ci(base, trained, n_boot=50) != delta_mrr_ci(
            base, trained, n_boot=500
        )

    def test_the_section_form_takes_the_count_too(self):
        base = np.array([0.5] * 20)
        trained = np.array([0.6] * 20)
        mean, low, high = delta_mrr_ci(base, trained, seed=3, n_boot=80)
        out = bootstrap_delta_mrr(base, trained, seed=3, n_boot=80)
        assert out == {"delta_mean": mean, "delta_ci": [low, high]}
