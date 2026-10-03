"""Tests for fsr.models.registry."""

from __future__ import annotations

import shlex

import pytest

from fsr.models.registry import (
    BASE_MODEL_IDS,
    BASE_MODELS,
    MODELS,
    by_id,
    by_slug,
    config_lines,
    label_of,
    main,
)


class TestRoster:
    def test_holds_every_variant(self):
        assert len(MODELS) == 7

    def test_slugs_are_unique(self):
        assert len({m.slug for m in MODELS}) == len(MODELS)

    def test_labels_are_unique(self):
        assert len({m.label for m in MODELS}) == len(MODELS)

    def test_no_slug_names_an_environment(self):
        for model in MODELS:
            assert "hex" not in model.slug

    def test_base_models_exclude_the_patched_head(self):
        assert len(BASE_MODELS) == 6
        assert all(not m.tanh_head for m in BASE_MODELS)

    def test_base_model_ids_keep_the_published_order(self):
        assert BASE_MODEL_IDS[0] == "cross-encoder/ms-marco-MiniLM-L6-v2"
        assert BASE_MODEL_IDS[-1] == "jinaai/jina-reranker-v2-base-multilingual"

    def test_base_model_ids_are_distinct(self):
        assert len(set(BASE_MODEL_IDS)) == len(BASE_MODEL_IDS)

    def test_the_patched_variant_shares_its_base_identifier(self):
        assert by_slug("mxbai_v1_tanh").model_id == by_slug("mxbai_v1").model_id

    def test_every_model_names_its_lora_targets(self):
        assert all(m.lora_targets for m in MODELS)

    def test_the_fused_attention_model_has_one_target(self):
        assert by_slug("jina_v2").lora_targets == ("Wqkv",)


class TestBySlug:
    def test_returns_the_model(self):
        assert by_slug("bge_v2_m3").model_id == "BAAI/bge-reranker-v2-m3"

    def test_rejects_an_unknown_slug(self):
        with pytest.raises(ValueError, match="unknown model 'bge'"):
            by_slug("bge")

    def test_names_the_known_slugs(self):
        with pytest.raises(ValueError, match="minilm_l6"):
            by_slug("bge")


class TestById:
    def test_returns_the_model(self):
        assert by_id("BAAI/bge-reranker-base").slug == "bge_base"

    def test_returns_the_base_model_for_a_shared_identifier(self):
        assert by_id("mixedbread-ai/mxbai-rerank-base-v1").slug == "mxbai_v1"

    def test_returns_none_for_an_unknown_identifier(self):
        assert by_id("some/other-model") is None


class TestLabelOf:
    def test_returns_the_label(self):
        assert label_of("cross-encoder/ms-marco-MiniLM-L12-v2") == "MiniLM-L12"

    def test_falls_back_to_the_identifier(self):
        assert label_of("some/other-model") == "some/other-model"


class TestBatchGeometry:
    def test_every_model_carries_a_batch_geometry(self):
        for model in MODELS:
            assert model.physical_batch > 0
            assert model.grad_accum > 0
            assert model.eval_batch > 0

    def test_effective_batch_is_the_product(self):
        model = by_slug("bge_base")
        assert model.effective_batch == model.physical_batch * model.grad_accum

    def test_effective_batch_is_uniform_across_the_roster(self):
        assert {m.effective_batch for m in MODELS} == {16}

    def test_the_patched_variant_matches_its_base_geometry(self):
        tanh = by_slug("mxbai_v1_tanh")
        plain = by_slug("mxbai_v1")
        assert tanh.physical_batch == plain.physical_batch
        assert tanh.grad_accum == plain.grad_accum
        assert tanh.eval_batch == plain.eval_batch

    def test_the_minilm_pair_share_a_geometry(self):
        l6 = by_slug("minilm_l6")
        l12 = by_slug("minilm_l12")
        assert (l6.physical_batch, l6.grad_accum, l6.eval_batch) == (4, 4, 32)
        assert (l12.physical_batch, l12.grad_accum, l12.eval_batch) == (4, 4, 32)


class TestConfigLines:
    def test_reports_every_setting(self):
        names = [line.split("=", 1)[0] for line in config_lines(by_slug("bge_base"))]
        assert names == [
            "MODEL_ID",
            "LORA_TARGETS",
            "PHYSICAL_BATCH",
            "GRAD_ACCUM",
            "EVAL_BATCH",
            "TANH_HEAD",
        ]

    def test_a_shell_recovers_every_scalar(self):
        for model in MODELS:
            expected = {
                "MODEL_ID": model.model_id,
                "PHYSICAL_BATCH": str(model.physical_batch),
                "GRAD_ACCUM": str(model.grad_accum),
                "EVAL_BATCH": str(model.eval_batch),
                "TANH_HEAD": "1" if model.tanh_head else "0",
            }
            for line in config_lines(model):
                name, quoted = line.split("=", 1)
                if name == "LORA_TARGETS":
                    continue
                assert shlex.split(quoted) == [expected[name]]

    def test_the_lora_targets_are_a_shell_array(self):
        lines = dict(line.split("=", 1) for line in config_lines(by_slug("bge_base")))
        assert lines["LORA_TARGETS"] == "(query key value)"

    def test_a_single_target_stays_an_array(self):
        lines = dict(line.split("=", 1) for line in config_lines(by_slug("jina_v2")))
        assert lines["LORA_TARGETS"] == "(Wqkv)"

    def test_reports_the_patched_head_as_one(self):
        lines = dict(ln.split("=", 1) for ln in config_lines(by_slug("mxbai_v1_tanh")))
        assert shlex.split(lines["TANH_HEAD"]) == ["1"]

    def test_reports_an_unpatched_head_as_zero(self):
        lines = dict(ln.split("=", 1) for ln in config_lines(by_slug("mxbai_v1")))
        assert shlex.split(lines["TANH_HEAD"]) == ["0"]


class TestMain:
    def test_list_prints_every_slug(self, capsys):
        assert main(["--list"]) == 0
        printed = capsys.readouterr().out.split()
        assert printed == [m.slug for m in MODELS]

    def test_config_prints_the_settings(self, capsys):
        assert main(["--config", "jina_v2"]) == 0
        assert capsys.readouterr().out.splitlines() == config_lines(by_slug("jina_v2"))

    def test_config_rejects_an_unknown_slug(self, capsys):
        assert main(["--config", "bge"]) == 2
        assert "unknown model 'bge'" in capsys.readouterr().err

    def test_requires_one_of_the_two(self):
        with pytest.raises(SystemExit):
            main([])

    def test_refuses_both(self):
        with pytest.raises(SystemExit):
            main(["--list", "--config", "jina_v2"])
