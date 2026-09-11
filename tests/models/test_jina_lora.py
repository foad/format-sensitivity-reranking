"""Tests for fsr.models.jina_lora."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora.layer import Linear as PeftLoraLinear

from fsr.models.jina_lora import JinaWqkvLora, _is_linear_residual, patch_jina_lora

IN_DIM = 64
OUT_DIM = 192
RANK = 4
ALPHA = 8
ADAPTER = "default"


class LinearResidual(nn.Linear):
    """A stand-in for jina's LinearResidual, which returns its input as well."""

    def forward(self, x):
        """Return the projection and the unchanged input."""
        return super().forward(x), x


def lora_config(rank: int = RANK, alpha: int = ALPHA) -> LoraConfig:
    """Return a LoRA config for a sequence-classification task."""
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=["_test"],
        task_type="SEQ_CLS",
        bias="none",
    )


def wrap(base: nn.Linear, rank: int = RANK, alpha: int = ALPHA) -> PeftLoraLinear:
    """Wrap a layer in PEFT's default Linear LoRA."""
    return PeftLoraLinear(
        base,
        adapter_name=ADAPTER,
        config=lora_config(rank, alpha),
        r=rank,
        lora_alpha=alpha,
    )


def wrapped_residual(seed: int = 0) -> tuple[PeftLoraLinear, LinearResidual]:
    """Return a wrapped LinearResidual with a non-zero LoRA delta.

    PEFT initialises lora_B to zero, which makes the delta zero. Both factors
    are re-initialised so the delta is measurable.
    """
    torch.manual_seed(seed)
    base = LinearResidual(IN_DIM, OUT_DIM)
    wrapper = wrap(base)
    with torch.no_grad():
        for factor in (wrapper.lora_A[ADAPTER], wrapper.lora_B[ADAPTER]):
            factor.weight.copy_(torch.randn_like(factor.weight) * 0.1)
    return wrapper, base


def patched() -> tuple[JinaWqkvLora, LinearResidual]:
    """Return a patched wrapper and its base layer."""
    wrapper, base = wrapped_residual()
    patch_jina_lora(nn.ModuleList([wrapper]))
    return wrapper, base


def inputs(batch: int = 2, seq: int = 10) -> torch.Tensor:
    """Return a deterministic input tensor."""
    torch.manual_seed(99)
    return torch.randn(batch, seq, IN_DIM)


class TestIsLinearResidual:
    def test_accepts_a_linear_residual(self):
        assert _is_linear_residual(LinearResidual(IN_DIM, OUT_DIM))

    def test_rejects_a_plain_linear(self):
        assert not _is_linear_residual(nn.Linear(IN_DIM, OUT_DIM))

    def test_rejects_a_non_linear_module(self):
        assert not _is_linear_residual(nn.LayerNorm(IN_DIM))


class TestVanillaPeftWrapper:
    def test_fails_on_a_tuple_returning_base_layer(self):
        wrapper, _ = wrapped_residual()
        with pytest.raises(AttributeError, match="dtype"):
            wrapper(inputs())


class TestPatchJinaLora:
    def test_swaps_the_wrapper_class(self):
        wrapper, _ = wrapped_residual()
        container = nn.ModuleList([wrapper])
        assert patch_jina_lora(container) == 1
        assert type(container[0]) is JinaWqkvLora

    def test_a_second_call_changes_nothing(self):
        container = nn.ModuleList([wrapped_residual()[0]])
        assert patch_jina_lora(container) == 1
        assert patch_jina_lora(container) == 0

    def test_leaves_a_plain_linear_wrapper_alone(self):
        torch.manual_seed(0)
        container = nn.ModuleList([wrap(nn.Linear(IN_DIM, OUT_DIM))])
        assert patch_jina_lora(container) == 0
        assert type(container[0]) is PeftLoraLinear

    def test_counts_every_swapped_wrapper(self):
        container = nn.ModuleList([wrapped_residual(seed=i)[0] for i in range(3)])
        assert patch_jina_lora(container) == 3

    def test_a_model_with_nothing_to_swap_is_unchanged(self):
        assert patch_jina_lora(nn.Module()) == 0


class TestForward:
    def test_returns_the_projection_and_the_residual(self):
        wrapper, _ = patched()
        x = inputs()
        y, residual = wrapper(x)
        assert y.shape == (2, 10, OUT_DIM)
        assert residual.shape == (2, 10, IN_DIM)

    def test_the_residual_is_the_input_tensor_itself(self):
        wrapper, _ = patched()
        x = inputs()
        assert wrapper(x)[1] is x

    def test_adds_the_scaled_lora_delta_to_the_projection(self):
        wrapper, base = patched()
        x = inputs()
        y_actual, _ = wrapper(x)
        with torch.no_grad():
            base_y, _ = base(x)
            delta = wrapper.lora_B[ADAPTER](wrapper.lora_A[ADAPTER](x))
            y_expected = base_y + delta * wrapper.scaling[ADAPTER]
        assert torch.allclose(y_actual, y_expected, atol=1e-5)

    def test_disabled_adapters_give_the_base_projection(self):
        wrapper, base = patched()
        x = inputs()
        wrapper.enable_adapters(False)
        y_disabled, _ = wrapper(x)
        with torch.no_grad():
            base_y, _ = base(x)
        assert torch.allclose(y_disabled, base_y, atol=1e-6)

    def test_disabled_adapters_still_return_the_residual(self):
        wrapper, _ = patched()
        x = inputs()
        wrapper.enable_adapters(False)
        assert wrapper(x)[1] is x

    def test_skips_an_active_adapter_with_no_weights(self):
        wrapper, base = patched()
        wrapper._active_adapter = [ADAPTER, "absent"]
        x = inputs()
        y_actual, _ = wrapper(x)
        with torch.no_grad():
            base_y, _ = base(x)
            delta = wrapper.lora_B[ADAPTER](wrapper.lora_A[ADAPTER](x))
        assert torch.allclose(
            y_actual, base_y + delta * wrapper.scaling[ADAPTER], atol=1e-5
        )

    def test_delegates_to_a_registered_lora_variant(self):
        wrapper, _ = patched()
        seen = {}

        class Variant:
            """A stand-in for a PEFT LoRA variant."""

            def forward(self, _layer, *, active_adapter, x, result, **_kwargs):
                """Record the call and return a marker tensor."""
                seen["adapter"] = active_adapter
                seen["x_shape"] = tuple(x.shape)
                return torch.full_like(result, 7.0)

        wrapper.lora_variant[ADAPTER] = Variant()
        y, _ = wrapper(inputs())
        assert seen["adapter"] == ADAPTER
        assert seen["x_shape"] == (2, 10, IN_DIM)
        assert torch.equal(y, torch.full_like(y, 7.0))

    def test_rejects_a_base_layer_that_returns_a_tensor(self):
        wrapper, _ = wrapped_residual()
        wrapper.base_layer = nn.Linear(IN_DIM, OUT_DIM)
        wrapper.__class__ = JinaWqkvLora
        with pytest.raises(TypeError, match="did not match"):
            wrapper(inputs())

    def test_rejects_mixed_batch_adapter_names(self):
        wrapper, _ = patched()
        with pytest.raises(NotImplementedError, match="adapter_names"):
            wrapper(inputs(), adapter_names=[ADAPTER, ADAPTER])


class TestMergeAndUnmerge:
    def test_merging_preserves_the_output(self):
        wrapper, _ = patched()
        x = inputs()
        y_before, _ = wrapper(x)
        wrapper.merge()
        assert wrapper.merged
        y_merged, residual = wrapper(x)
        assert torch.allclose(y_before, y_merged, atol=1e-5)
        assert residual is x

    def test_unmerging_restores_the_output(self):
        wrapper, _ = patched()
        x = inputs()
        y_before, _ = wrapper(x)
        wrapper.merge()
        wrapper.unmerge()
        assert not wrapper.merged
        y_after, _ = wrapper(x)
        assert torch.allclose(y_before, y_after, atol=1e-5)

    def test_disabling_adapters_while_merged_unmerges_them(self):
        wrapper, base = patched()
        x = inputs()
        wrapper.merge()
        wrapper.enable_adapters(False)
        y_disabled, _ = wrapper(x)
        assert not wrapper.merged
        with torch.no_grad():
            base_y, _ = base(x)
        assert torch.allclose(y_disabled, base_y, atol=1e-6)
