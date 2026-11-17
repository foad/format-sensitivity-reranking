"""Tests for fsr.h2_pooler."""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import save_file

from fsr import h2_pooler as mod
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import adapter_dir, arm, comparison_path, selection_path
from tests.test_h2_results import (
    BASE_LEADS,
    CONTROL_LEADS,
    TREATED_LEADS,
    comparison,
    selection,
    write_cross,
    write_within,
)

BASE, VARIANT = "minilm_l6", "minilm_l6_pooler"
SEEDS = (0, 1)
TREATED_SPREAD = {BASE: 0.1, VARIANT: 0.3}


@pytest.fixture
def root(tmp_path):
    """Write a base model and its pooler variant under two seeds."""
    path = selection_path(tmp_path, BASE, "lambda")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(selection()))
    for slug in (BASE, VARIANT):
        write_within(tmp_path, slug, "base", BASE_LEADS)
        for seed in SEEDS:
            for weight, lead_map, spread in (
                (0.0, CONTROL_LEADS, 1.0),
                (1.0, TREATED_LEADS, TREATED_SPREAD[slug]),
            ):
                for fold in FORMAT_NAMES:
                    name = arm(fold, weight, seed=seed)
                    out = comparison_path(tmp_path, slug, name)
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_text(json.dumps(comparison(fold, spread)))
                    write_within(tmp_path, slug, name, lead_map)
                    write_cross(tmp_path, slug, name, spread)
    return tmp_path


class TestPairs:
    def test_pairs_every_pooler_variant_with_its_base(self):
        assert mod.PAIRS == (
            ("minilm_l6", "minilm_l6_pooler"),
            ("minilm_l12", "minilm_l12_pooler"),
            ("mxbai_v1", "mxbai_v1_pooler"),
        )


class TestSeedLevels:
    def test_has_one_row_per_seed(self, root):
        frame = mod.seed_levels(root, BASE, "ood", SEEDS)
        assert list(frame.index) == list(SEEDS)

    def test_the_change_is_a_share_of_the_control(self, root):
        row = mod.seed_levels(root, BASE, "ood", SEEDS).iloc[0]
        expected = (
            100 * (row["score_chosen"] - row["score_control"]) / row["score_control"]
        )
        assert row["score_pct"] == pytest.approx(expected)

    def test_the_variant_reads_the_base_weight(self, root):
        frame = mod.seed_levels(root, VARIANT, "ood", SEEDS)
        assert frame["score_chosen"].notna().all()

    def test_the_delta_is_the_chosen_less_the_control(self, root):
        row = mod.seed_levels(root, BASE, "ood", SEEDS).iloc[0]
        assert row["score_delta"] == pytest.approx(
            row["score_chosen"] - row["score_control"]
        )


class TestSignTest:
    def test_counts_the_seeds_where_the_variant_is_lower(self):
        out = mod.sign_test([1, 2, 3, 4, 5], [0, 1, 2, 3, 4])
        assert out["variant_lower"] == 5
        assert out["sign_p"] == pytest.approx(0.0625)

    def test_leaves_ties_out(self):
        out = mod.sign_test([1, 2, 3], [1, 1, 3])
        assert out["untied"] == 1
        assert out["variant_lower"] == 1

    def test_all_ties_give_no_p_value(self):
        assert np.isnan(mod.sign_test([1, 2], [1, 2])["sign_p"])


class TestPairedTable:
    def test_reports_each_pair_under_the_base_label(self, root):
        table = mod.paired_table(root, "score_pct", "ood", [(BASE, VARIANT)], SEEDS)
        assert list(table.index) == ["MiniLM-L6"]

    def test_a_weaker_treatment_reduces_less(self, root):
        row = mod.paired_table(root, "score_pct", "ood", [(BASE, VARIANT)], SEEDS)
        assert row["variant_mean"].iloc[0] > row["base_mean"].iloc[0]
        assert row["variant_lower"].iloc[0] == 0


class TestSeedPairs:
    def test_sets_base_and_variant_side_by_side(self, root):
        frame = mod.seed_pairs(root, "score_pct", "ood", [(BASE, VARIANT)], SEEDS)
        assert list(frame.columns) == ["base", "variant", "difference"]
        assert list(frame.index) == [("MiniLM-L6", s) for s in SEEDS]


def write_adapter(root, slug, name, tensors):
    path = adapter_dir(root, slug, name) / mod.ADAPTER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({mod.PEFT_PREFIX + k: v for k, v in tensors.items()}, str(path))


PRETRAINED = {
    "pooler.dense.weight": np.eye(2, dtype=np.float32),
    "pooler.dense.bias": np.zeros(2, dtype=np.float32),
    "classifier.weight": np.ones((1, 2), dtype=np.float32),
}
LORA = {"layer.0.query.lora_A.weight": np.zeros((4, 2), dtype=np.float32)}


class TestHeadRole:
    @pytest.mark.parametrize(
        ("module", "role"),
        [
            ("bert.pooler.dense", "dense"),
            ("pooler.dense", "dense"),
            ("classifier.dense", "dense"),
            ("classifier", "output"),
            ("classifier.out_proj", "output"),
        ],
    )
    def test_names_the_role_of_each_head_module(self, module, role):
        assert mod.head_role(module) == role


class TestSeedDifference:
    def test_is_the_variant_less_the_base(self, root):
        frame = mod.seed_pairs(root, "score_delta", "ood", [(BASE, VARIANT)], SEEDS)
        assert np.allclose(frame["difference"], frame["variant"] - frame["base"])


class TestTrainableParameters:
    def test_counts_lora_and_each_head_module(self, root):
        write_adapter(root, BASE, arm(FORMAT_NAMES[0], 0.0), {**LORA, **PRETRAINED})
        row = mod.trainable_parameters(root, [BASE]).loc["MiniLM-L6"]
        assert row["lora"] == 8
        assert row["dense"] == 6
        assert row["output"] == 2
        assert row["total"] == 16


class TestHeadWeightChange:
    @pytest.fixture
    def weights(self, tmp_path):
        path = tmp_path / "pretrained.safetensors"
        save_file(PRETRAINED, str(path))
        return path

    def test_measures_the_change_against_the_pretrained_norm(self, root, weights):
        for seed in SEEDS:
            for fold in FORMAT_NAMES:
                for weight, scale in ((0.0, 1.5), (1.0, 2.0)):
                    trained = {**PRETRAINED}
                    trained["pooler.dense.weight"] = scale * np.eye(2, dtype=np.float32)
                    write_adapter(
                        root, BASE, arm(fold, weight, seed=seed), {**LORA, **trained}
                    )
        frame = mod.head_weight_change(root, [BASE], lambda _: weights, SEEDS)
        row = frame.loc[("MiniLM-L6", "pooler.dense")]
        assert row[mod.CONTROL] == pytest.approx(0.5)
        assert row[mod.CHOSEN] == pytest.approx(1.0)
        assert frame.loc[("MiniLM-L6", "classifier"), mod.CHOSEN] == 0.0
        assert frame.loc[("MiniLM-L6", "pooler.dense"), "role"] == "dense"

    def test_skips_lora_and_bias_tensors(self, root, weights):
        for seed in SEEDS:
            for fold in FORMAT_NAMES:
                for weight in (0.0, 1.0):
                    write_adapter(
                        root, BASE, arm(fold, weight, seed=seed), {**LORA, **PRETRAINED}
                    )
        frame = mod.head_weight_change(root, [BASE], lambda _: weights, SEEDS)
        modules = set(frame.index.get_level_values("module"))
        assert modules == {"pooler.dense", "classifier"}
