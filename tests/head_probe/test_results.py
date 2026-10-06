"""Tests for fsr.head_probe.results."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from fsr.h2_layout import frontier_path
from fsr.head_probe.heads import GELU, LINEAR, TANH, WIDE_LINEAR
from fsr.head_probe.results import (
    REPORTED_HEADS,
    activation_contrast,
    arm_table,
    available,
    capacity_contrast,
    contrast_verdict,
    frontier_frame,
    load_frontier,
    mrr_table,
    points_frame,
    weights_of,
)

SLUG = "mxbai_v1"
OTHER = "jina_v2"
WEIGHTS = (0.0, 1.0)
PARAMETERS = {LINEAR: 769, WIDE_LINEAR: 591361, GELU: 591361, TANH: 591361}

# max |d| by head and weight, then the per-seed offsets that build the spread.
SCORES = {
    (LINEAR, 0.0): 0.50,
    (LINEAR, 1.0): 0.25,
    (WIDE_LINEAR, 0.0): 0.40,
    (WIDE_LINEAR, 1.0): 0.12,
    (GELU, 0.0): 0.30,
    (GELU, 1.0): 0.10,
    (TANH, 0.0): 0.40,
    (TANH, 1.0): 0.11,
}
OFFSETS = (-0.001, 0.0, 0.001)


def write_frontier(root, slug, scores=None, heads=REPORTED_HEADS, weights=WEIGHTS):
    """Write a frontier whose points are fixed, so every table is predictable."""
    scores = SCORES if scores is None else scores
    points = [
        {
            "head": head,
            "lambda_inv": weight,
            "seed": seed,
            "parameters": PARAMETERS[head],
            "affine": head in (LINEAR, WIDE_LINEAR),
            "bounded": head == TANH,
            "max_abs_cohen_d": scores[(head, weight)] + offset,
            "mean_mrr": 0.95,
            "min_mrr": 0.90,
            "mrr_per_format": {},
            "final_loss": 0.1,
            "final_rank_loss": 0.1,
            "final_inv_loss": 0.1,
            "history": [],
        }
        for head in heads
        for weight in weights
        for seed, offset in enumerate(OFFSETS)
    ]
    path = frontier_path(root, slug)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "model": slug,
                "lambdas": list(weights),
                "heads": list(heads),
                "seeds": [0, 1, 2],
                "n_fit_records": 100,
                "n_eval_records": 20,
                "eval_split": "dev",
                "points": points,
                "failures": [],
            }
        )
    )


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "nq"
    write_frontier(root, SLUG)
    write_frontier(root, OTHER)
    return root


class TestAvailable:
    def test_finds_a_model_whose_frontier_is_present(self, data_root):
        assert set(available(data_root)) == {SLUG, OTHER}

    def test_reports_nothing_when_no_frontier_is_present(self, tmp_path):
        assert available(tmp_path / "nq") == []

    def test_keeps_the_roster_order(self, data_root):
        """The roster order is what every table and figure is indexed by."""
        assert available(data_root) == [SLUG, OTHER]


class TestPointsFrame:
    def test_holds_one_row_per_fit(self, data_root):
        frame = points_frame(data_root, SLUG)
        assert len(frame) == len(REPORTED_HEADS) * len(WEIGHTS) * len(OFFSETS)

    def test_drops_an_arm_the_caller_did_not_name(self, data_root):
        frame = points_frame(data_root, SLUG, (GELU, TANH))
        assert set(frame["head"]) == {GELU, TANH}

    def test_orders_the_arms_as_the_caller_named_them(self, data_root):
        """The figure legend and the table rows follow this order."""
        frame = points_frame(data_root, SLUG, (TANH, GELU))
        assert list(frame["head"].unique()) == [TANH, GELU]

    def test_refuses_a_frontier_holding_none_of_the_arms(self, data_root):
        with pytest.raises(ValueError, match="holds no fit"):
            points_frame(data_root, SLUG, ("deep",))


class TestArmTable:
    def test_reports_the_parameter_count_of_each_arm(self, data_root):
        table = arm_table(data_root, SLUG)
        assert table.loc[LINEAR, "parameters"] == 769
        assert table.loc[GELU, "parameters"] == table.loc[TANH, "parameters"]

    def test_marks_the_arms_that_compose_to_an_affine_function(self, data_root):
        table = arm_table(data_root, SLUG)
        assert table.loc[WIDE_LINEAR, "affine"]
        assert not table.loc[GELU, "affine"]

    def test_marks_the_one_arm_whose_activation_saturates(self, data_root):
        table = arm_table(data_root, SLUG)
        assert table.loc[TANH, "bounded"]
        assert not table.loc[GELU, "bounded"]


class TestFrontierFrame:
    def test_averages_the_seeds(self, data_root):
        frame = frontier_frame(data_root, SLUG)
        assert frame.loc[(GELU, 0.0), "max_abs_d"] == pytest.approx(0.30)

    def test_reports_the_spread_over_the_seeds(self, data_root):
        frame = frontier_frame(data_root, SLUG)
        assert frame.loc[(GELU, 0.0), "max_abs_d_sd"] == pytest.approx(0.001)

    def test_counts_the_seeds_behind_each_row(self, data_root):
        frame = frontier_frame(data_root, SLUG)
        assert frame.loc[(GELU, 0.0), "seeds"] == len(OFFSETS)

    def test_a_single_seed_leaves_the_spread_undefined(self, tmp_path):
        """One seed cannot bound itself, so the band has to be absent."""
        root = tmp_path / "nq"
        write_frontier(root, SLUG)
        payload = json.loads(frontier_path(root, SLUG).read_text())
        payload["points"] = [p for p in payload["points"] if p["seed"] == 0]
        frontier_path(root, SLUG).write_text(json.dumps(payload))
        frame = frontier_frame(root, SLUG)
        assert pd.isna(frame.loc[(GELU, 0.0), "max_abs_d_sd"])


class TestWeightsOf:
    def test_returns_the_swept_weights_in_order(self, data_root):
        assert weights_of(data_root, SLUG) == [0.0, 1.0]


class TestActivationContrast:
    def test_holds_one_row_per_model_and_weight(self, data_root):
        contrast = activation_contrast(data_root, [SLUG, OTHER])
        assert len(contrast) == 2 * len(WEIGHTS)

    def test_a_negative_difference_favours_tanh(self, data_root):
        """The sign is the whole reading of the figure."""
        contrast = activation_contrast(data_root, [SLUG])
        row = contrast.iloc[0]
        assert row["tanh_minus_gelu"] == pytest.approx(0.10)
        assert row["tanh"] > row["gelu"]

    def test_a_difference_inside_the_band_does_not_resolve(self, tmp_path):
        scores = dict(SCORES)
        scores[(TANH, 0.0)] = scores[(GELU, 0.0)]
        root = tmp_path / "nq"
        write_frontier(root, SLUG, scores=scores)
        contrast = activation_contrast(root, [SLUG])
        assert not contrast.loc[(contrast.index[0])]["resolved"]

    def test_a_difference_outside_the_band_resolves(self, data_root):
        contrast = activation_contrast(data_root, [SLUG])
        assert contrast.iloc[0]["resolved"]

    def test_a_wider_band_resolves_less(self, data_root):
        wide = activation_contrast(data_root, [SLUG], noise=1e6)
        assert not wide["resolved"].any()

    def test_is_indexed_by_label_and_weight(self, data_root):
        contrast = activation_contrast(data_root, [SLUG])
        assert contrast.index.names == ["model", "weight"]


class TestContrastVerdict:
    def test_counts_the_cells_that_clear_the_band(self, data_root):
        verdict = contrast_verdict(activation_contrast(data_root, [SLUG, OTHER]))
        assert verdict["cells"] == 2 * len(WEIGHTS)
        assert verdict["resolved"] == verdict["cells"]

    def test_splits_the_resolved_cells_by_which_arm_leads(self, data_root):
        verdict = contrast_verdict(activation_contrast(data_root, [SLUG]))
        assert verdict["gelu_lower"] == len(WEIGHTS)
        assert verdict["tanh_lower"] == 0

    def test_counts_a_cell_that_favours_tanh(self, tmp_path):
        scores = dict(SCORES)
        scores[(TANH, 0.0)] = 0.10
        root = tmp_path / "nq"
        write_frontier(root, SLUG, scores=scores)
        verdict = contrast_verdict(activation_contrast(root, [SLUG]))
        assert verdict["tanh_lower"] == 1


class TestCapacityContrast:
    def test_holds_one_row_per_model_and_weight(self, data_root):
        assert len(capacity_contrast(data_root, [SLUG])) == len(WEIGHTS)

    def test_a_negative_difference_favours_the_two_layer_head(self, data_root):
        row = capacity_contrast(data_root, [SLUG]).iloc[0]
        assert row["wide_minus_linear"] == pytest.approx(-0.10)
        assert row["resolved"]

    def test_a_wider_band_resolves_less(self, data_root):
        wide = capacity_contrast(data_root, [SLUG], noise=1e6)
        assert not wide["resolved"].any()


class TestMrrTable:
    def test_holds_one_row_per_model_and_arm(self, data_root):
        table = mrr_table(data_root, [SLUG, OTHER])
        assert len(table) == 2 * len(REPORTED_HEADS)

    def test_holds_one_column_per_weight(self, data_root):
        table = mrr_table(data_root, [SLUG])
        assert list(table.columns) == list(WEIGHTS)

    def test_is_indexed_by_label_and_arm(self, data_root):
        table = mrr_table(data_root, [SLUG])
        assert table.index.names == ["model", "head"]


class TestLoadFrontier:
    def test_returns_the_written_payload(self, data_root):
        assert load_frontier(data_root, SLUG)["model"] == SLUG
