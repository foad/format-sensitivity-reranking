from __future__ import annotations

import pytest
import torch
from torch import nn

from fsr.models import adapters
from fsr.models.adapters import (
    DEFAULT_LORA_ALPHA,
    DEFAULT_LORA_DROPOUT,
    DEFAULT_LORA_RANK,
    DEFAULT_LORA_TARGETS,
    AdaptedModel,
    adapt,
    build_adapted_model,
    lora_config,
    parameter_counts,
    require_fp32,
)


class Base(nn.Module):
    """A sequence-classification model with the attention names LoRA targets."""

    def __init__(self, width: int = 8):
        """Build the projections and the classifier head."""
        super().__init__()
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.classifier = nn.Linear(width, 1)
        self.embeddings_required = False
        self.checkpointing = False

    def forward(self, x):
        """Score the input."""
        return self.classifier(self.query(x))

    def enable_input_require_grads(self):
        """Record the call the adapter wrapper makes."""
        self.embeddings_required = True

    def gradient_checkpointing_enable(self, **_kwargs):
        """Record the call the adapter wrapper makes."""
        self.checkpointing = True


class TestLoraConfig:
    def test_uses_the_published_defaults(self):
        config = lora_config()
        assert config.r == DEFAULT_LORA_RANK == 16
        assert config.lora_alpha == DEFAULT_LORA_ALPHA == 32
        assert config.lora_dropout == DEFAULT_LORA_DROPOUT == 0.05

    def test_targets_the_attention_projections_by_default(self):
        assert set(lora_config().target_modules) == set(DEFAULT_LORA_TARGETS)

    def test_takes_a_fused_projection_target(self):
        assert set(lora_config(("Wqkv",)).target_modules) == {"Wqkv"}

    def test_trains_no_bias(self):
        assert lora_config().bias == "none"

    def test_names_the_classification_task(self):
        assert lora_config().task_type == "SEQ_CLS"

    def test_takes_an_overridden_rank(self):
        assert lora_config(rank=8, alpha=16).r == 8


class TestRequireFp32:
    def test_accepts_a_float32_model(self):
        require_fp32(Base())

    def test_rejects_a_float16_model(self):
        with pytest.raises(ValueError, match="float16"):
            require_fp32(Base().half())

    def test_accepts_bfloat16(self):
        require_fp32(Base().bfloat16())


class TestParameterCounts:
    def test_counts_every_parameter(self):
        _, total = parameter_counts(Base(width=2))
        assert total == sum(p.numel() for p in Base(width=2).parameters())

    def test_counts_only_the_trainable_ones(self):
        model = Base(width=2)
        for p in model.query.parameters():
            p.requires_grad_(False)
        trainable, total = parameter_counts(model)
        assert trainable < total

    def test_counts_nothing_for_a_frozen_model(self):
        model = Base()
        for p in model.parameters():
            p.requires_grad_(False)
        assert parameter_counts(model)[0] == 0


class TestAdapt:
    def test_attaches_the_adapter(self):
        result = adapt(Base(), lora_config(("query", "key", "value")))
        assert any("lora_A" in name for name, _ in result.model.named_parameters())

    def test_leaves_the_model_in_training_mode(self):
        assert adapt(Base(), lora_config()).model.training

    def test_trains_a_small_share_of_the_parameters(self):
        result = adapt(Base(width=64), lora_config())
        assert 0 < result.trainable_fraction < 0.5

    def test_enables_gradient_checkpointing_by_default(self):
        base = Base()
        adapt(base, lora_config())
        assert base.embeddings_required
        assert base.checkpointing

    def test_skips_gradient_checkpointing_on_request(self):
        base = Base()
        result = adapt(base, lora_config(), gradient_checkpointing=False)
        assert not base.checkpointing
        assert not result.gradient_checkpointing

    def test_reports_no_jina_patch_for_plain_projections(self):
        assert adapt(Base(), lora_config()).jina_patched == 0

    def test_rejects_a_half_precision_model(self):
        with pytest.raises(ValueError, match="float32"):
            adapt(Base().half(), lora_config(), gradient_checkpointing=False)

    def test_keeps_the_classifier_trainable(self):
        result = adapt(Base(), lora_config())
        names = [n for n, p in result.model.named_parameters() if p.requires_grad]
        assert any("classifier" in n for n in names)

    def test_the_task_type_marks_the_classifier_for_saving(self):
        config = lora_config()
        assert config.modules_to_save is None
        adapt(Base(), config)
        assert config.modules_to_save == ["classifier", "score"]


class TestAdaptedModel:
    def test_reports_the_trainable_fraction(self):
        assert AdaptedModel(None, 0, 25, 100, True).trainable_fraction == 0.25

    def test_reports_zero_for_a_model_with_no_parameters(self):
        assert AdaptedModel(None, 0, 0, 0, True).trainable_fraction == 0.0


class TestBuildAdaptedModel:
    @pytest.fixture
    def loaded(self, monkeypatch):
        seen = {}

        def fake_load_model(model_id, device, **kwargs):
            seen["model_id"] = model_id
            seen["device"] = device
            seen.update(kwargs)
            return Base()

        monkeypatch.setattr(adapters, "load_model", fake_load_model)
        return seen

    def test_loads_the_named_model_onto_the_device(self, loaded):
        build_adapted_model("BAAI/bge-reranker-base", "cpu")
        assert loaded["model_id"] == "BAAI/bge-reranker-base"
        assert loaded["device"] == "cpu"

    def test_asks_for_no_head_replacement_by_default(self, loaded):
        build_adapted_model("m", "cpu")
        assert loaded["tanh_head"] is False

    def test_passes_the_head_replacement_through(self, loaded):
        build_adapted_model("m", "cpu", tanh_head=True)
        assert loaded["tanh_head"] is True

    def test_passes_the_attention_kernel_choice_through(self, loaded):
        build_adapted_model("m", "cpu", eager_attn=True)
        assert loaded["eager_attn"] is True

    @pytest.mark.usefixtures("loaded")
    def test_applies_the_given_rank(self):
        result = build_adapted_model("m", "cpu", rank=4, alpha=8)
        assert result.model.peft_config["default"].r == 4

    @pytest.mark.usefixtures("loaded")
    def test_applies_the_given_targets(self):
        result = build_adapted_model("m", "cpu", lora_targets=("query",))
        assert set(result.model.peft_config["default"].target_modules) == {"query"}

    @pytest.mark.usefixtures("loaded")
    def test_returns_an_adapted_model(self):
        result = build_adapted_model("m", "cpu")
        assert isinstance(result, AdaptedModel)
        assert result.trainable > 0

    @pytest.mark.usefixtures("loaded")
    def test_skips_gradient_checkpointing_on_request(self):
        result = build_adapted_model("m", "cpu", gradient_checkpointing=False)
        assert not result.gradient_checkpointing


class TestTanhHeadPath:
    def test_the_head_is_replaced_before_the_adapter_attaches(self, monkeypatch):
        order = []

        def fake_load_model(_model_id, _device, **kwargs):
            base = Base()
            if kwargs.get("tanh_head"):
                order.append("patch")
                base.classifier = nn.Sequential(
                    nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1)
                )
            return base

        monkeypatch.setattr(adapters, "load_model", fake_load_model)
        monkeypatch.setattr(
            adapters,
            "get_peft_model",
            lambda base, config: (order.append("adapt"), get_real(base, config))[1],
        )
        from peft import get_peft_model as get_real

        result = build_adapted_model("m", "cpu", tanh_head=True)
        assert order == ["patch", "adapt"]
        names = [n for n, p in result.model.named_parameters() if p.requires_grad]
        assert any("classifier" in n for n in names)

    def test_the_replaced_head_stays_trainable(self, monkeypatch):
        def fake_load_model(_model_id, _device, **_kwargs):
            base = Base()
            base.classifier = nn.Sequential(nn.Linear(8, 8), nn.Tanh(), nn.Linear(8, 1))
            return base

        monkeypatch.setattr(adapters, "load_model", fake_load_model)
        result = build_adapted_model("m", "cpu", tanh_head=True)
        trainable = {n for n, p in result.model.named_parameters() if p.requires_grad}
        assert any(
            n.endswith("classifier.modules_to_save.default.0.weight") for n in trainable
        )
        assert any(
            n.endswith("classifier.modules_to_save.default.2.weight") for n in trainable
        )


class TestFp32Guard:
    def test_a_half_precision_load_is_refused(self, monkeypatch):
        monkeypatch.setattr(adapters, "load_model", lambda *_a, **_k: Base().half())
        with pytest.raises(ValueError, match="float32"):
            build_adapted_model("m", "cpu", gradient_checkpointing=False)


class TestTrainableParameters:
    def test_an_adapted_model_trains_only_the_adapter_and_the_head(self):
        result = adapt(Base(width=16), lora_config())
        for name, param in result.model.named_parameters():
            if param.requires_grad:
                assert "lora_" in name or "classifier" in name

    def test_the_base_projections_stay_frozen(self):
        result = adapt(Base(width=16), lora_config())
        frozen = [
            n
            for n, p in result.model.named_parameters()
            if not p.requires_grad and "query" in n
        ]
        assert frozen


class TestGradientFlow:
    def test_a_backward_pass_reaches_the_adapter(self):
        result = adapt(Base(width=8), lora_config(), gradient_checkpointing=False)
        wrapped = result.model.base_model.model
        wrapped(torch.randn(2, 8)).sum().backward()
        grads = [
            p.grad
            for n, p in result.model.named_parameters()
            if "lora_A" in n and p.grad is not None
        ]
        assert grads

    def test_the_frozen_base_weight_takes_no_gradient(self):
        result = adapt(Base(width=8), lora_config(), gradient_checkpointing=False)
        wrapped = result.model.base_model.model
        wrapped(torch.randn(2, 8)).sum().backward()
        assert wrapped.query.base_layer.weight.grad is None
