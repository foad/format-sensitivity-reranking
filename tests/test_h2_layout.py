from __future__ import annotations

from pathlib import Path

import pytest

from fsr.h2_layout import (
    ADAPTER_CONFIG_NAME,
    ADAPTER_NAME,
    AXES,
    BASE_ARM,
    PHASE1_FOLD,
    SPLITS,
    SWEEPS,
    adapter_config_path,
    adapter_dir,
    arm,
    comparison_path,
    feature_meta_path,
    feature_path,
    feature_score_path,
    format_lambda,
    h2_dir,
    result_path,
    selection_path,
    train_dir,
)

ROOT = Path("data/nq")
LAMBDAS = (0, 0.01, 0.1, 1.0, 10.0)


class TestH2Dir:
    def test_sits_under_the_corpus_directory(self):
        assert h2_dir(ROOT) == ROOT / "h2"

    def test_keeps_the_two_phases_apart(self):
        assert h2_dir(ROOT) != ROOT / "h1"


class TestFormatLambda:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [(0, "0"), (0.0, "0"), (0.01, "0.01"), (0.1, "0.1"), (1.0, "1"), (10.0, "10")],
    )
    def test_formats_each_published_weight(self, value, expected):
        assert format_lambda(value) == expected

    def test_every_published_weight_reads_back(self):
        assert [float(format_lambda(x)) for x in LAMBDAS] == [float(x) for x in LAMBDAS]

    def test_the_published_weights_stay_distinct(self):
        assert len({format_lambda(x) for x in LAMBDAS}) == len(LAMBDAS)


class TestArm:
    def test_names_the_five_format_phase(self):
        assert arm(None, 0.1) == "5fmt_lam0.1"
        assert PHASE1_FOLD == "5fmt"

    def test_names_a_held_out_fold(self):
        assert arm("yaml", 0) == "yaml_lam0"

    def test_adds_the_rank_when_one_is_swept(self):
        assert arm(None, 0.1, rank=8) == "5fmt_lam0.1_r8"

    def test_omits_the_rank_by_default(self):
        assert "_r" not in arm("markdown", 1.0)

    def test_every_fold_and_weight_gives_a_distinct_name(self):
        names = {
            arm(fold, lam)
            for fold in (None, "yaml", "json", "toml", "inline_kv", "markdown")
            for lam in LAMBDAS
        }
        assert len(names) == 6 * len(LAMBDAS)

    def test_differs_from_the_untrained_arm(self):
        assert arm(None, 0) != BASE_ARM


class TestTrainDir:
    def test_names_the_model_and_the_arm(self):
        assert train_dir(ROOT, "bge_base", "5fmt_lam0.1") == (
            ROOT / "h2" / "train" / "bge_base_5fmt_lam0.1"
        )

    def test_the_adapter_sits_inside_it(self):
        assert adapter_dir(ROOT, "bge_base", "yaml_lam0") == (
            train_dir(ROOT, "bge_base", "yaml_lam0") / ADAPTER_NAME
        )

    def test_the_adapter_carries_no_protocol_qualifier(self):
        assert ADAPTER_NAME == "adapter"

    def test_two_arms_of_one_model_stay_apart(self):
        assert train_dir(ROOT, "m", "yaml_lam0") != train_dir(ROOT, "m", "json_lam0")


class TestResultPath:
    def test_orders_the_fields_from_general_to_specific(self):
        assert result_path(ROOT, "test", "cross", "mxbai_v1", "5fmt_lam1") == (
            ROOT / "h2" / "test_cross_mxbai_v1_5fmt_lam1.json"
        )

    def test_names_the_untrained_arm(self):
        assert result_path(ROOT, "dev", "cross", "jina_v2", BASE_ARM).name == (
            "dev_cross_jina_v2_base.json"
        )

    def test_covers_the_answer_axis(self):
        assert result_path(ROOT, "test", "within", "m", "a").name.startswith(
            "test_within_"
        )

    def test_covers_the_prose_check(self):
        assert result_path(ROOT, "prose", "mrr", "m", "a").name == (
            "prose_mrr_m_a.json"
        )

    def test_rejects_an_unknown_split(self):
        with pytest.raises(ValueError, match="unknown split 'train'"):
            result_path(ROOT, "train", "cross", "m", "a")

    def test_rejects_an_unknown_axis(self):
        with pytest.raises(ValueError, match="unknown axis 'rank'"):
            result_path(ROOT, "test", "rank", "m", "a")

    def test_names_every_known_split_and_axis(self):
        paths = {
            result_path(ROOT, split, axis, "m", "a")
            for split in SPLITS
            for axis in AXES
        }
        assert len(paths) == len(SPLITS) * len(AXES)

    def test_sits_beside_the_other_results(self):
        assert result_path(ROOT, "test", "cross", "m", "a").parent == h2_dir(ROOT)


class TestSelectionPath:
    def test_names_the_model_and_the_sweep(self):
        assert selection_path(ROOT, "minilm_l6", "lambda") == (
            ROOT / "h2" / "selection" / "minilm_l6_lambda.json"
        )

    def test_covers_the_rank_sweep(self):
        assert selection_path(ROOT, "m", "rank").name == "m_rank.json"

    def test_rejects_an_unknown_sweep(self):
        with pytest.raises(ValueError, match="unknown sweep 'dropout'"):
            selection_path(ROOT, "m", "dropout")

    def test_names_every_known_sweep(self):
        assert len({selection_path(ROOT, "m", s) for s in SWEEPS}) == len(SWEEPS)


class TestComparisonPath:
    def test_names_the_model_and_the_contrast(self):
        assert comparison_path(ROOT, "bge_v2_m3", "phase2") == (
            ROOT / "h2" / "comparison" / "bge_v2_m3_phase2.json"
        )

    def test_keeps_two_contrasts_apart(self):
        assert comparison_path(ROOT, "m", "phase1") != comparison_path(
            ROOT, "m", "prose"
        )


class TestNoProtocolQualifier:
    def test_no_path_carries_the_private_suffix(self):
        paths = [
            train_dir(ROOT, "m", arm(None, 0.1)),
            adapter_dir(ROOT, "m", arm("yaml", 0)),
            result_path(ROOT, "test", "cross", "m", BASE_ARM),
            selection_path(ROOT, "m", "lambda"),
            comparison_path(ROOT, "m", "phase1"),
        ]
        assert all("fixed" not in str(p) for p in paths)

    def test_no_path_names_an_environment(self):
        assert "hex" not in str(train_dir(ROOT, "minilm_l6", arm(None, 0)))


class TestAdapterConfigPath:
    def test_sits_inside_the_adapter_directory(self):
        assert adapter_config_path(ROOT, "m", "yaml_lam0").parent == adapter_dir(
            ROOT, "m", "yaml_lam0"
        )

    def test_names_the_configuration_file(self):
        assert adapter_config_path(ROOT, "m", "yaml_lam0").name == ADAPTER_CONFIG_NAME
        assert ADAPTER_CONFIG_NAME == "adapter_config.json"

    def test_two_arms_stay_apart(self):
        assert adapter_config_path(ROOT, "m", "yaml_lam0") != adapter_config_path(
            ROOT, "m", "json_lam0"
        )


class TestProbePaths:
    def test_names_the_feature_cache(self, tmp_path):
        path = feature_path(tmp_path, "train", "minilm_l6")
        assert path.name == "train_features_minilm_l6.npy"
        assert path.parent.name == "probe"

    def test_puts_the_scores_beside_the_features(self, tmp_path):
        path = feature_score_path(tmp_path, "dev", "minilm_l6")
        assert path.name == "dev_scores_minilm_l6.npy"
        assert path.parent == feature_path(tmp_path, "dev", "minilm_l6").parent

    def test_puts_the_description_beside_the_features(self, tmp_path):
        path = feature_meta_path(tmp_path, "dev", "minilm_l6")
        assert path.name == "dev_features_minilm_l6.json"

    def test_refuses_a_split_the_probe_does_not_cover(self, tmp_path):
        with pytest.raises(ValueError, match="unknown probe split"):
            feature_path(tmp_path, "test", "minilm_l6")
