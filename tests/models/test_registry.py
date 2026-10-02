"""Tests for fsr.models.registry."""

from __future__ import annotations

import pytest

from fsr.models.registry import (
    BASE_MODEL_IDS,
    BASE_MODELS,
    MODELS,
    by_id,
    by_slug,
    label_of,
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
