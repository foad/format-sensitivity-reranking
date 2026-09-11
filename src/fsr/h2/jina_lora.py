"""LoRA support for jina-reranker-v2's fused Wqkv projection.

jina's Wqkv is a `LinearResidual`, an `nn.Linear` subclass whose forward returns
the pair `(projection, input)`. PEFT's default LoRA wrapper expects a tensor.
This module provides a wrapper that accepts the pair, and a helper that installs
it on a wrapped model.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from peft.tuners.lora.layer import VARIANT_KWARG_KEYS
from peft.tuners.lora.layer import Linear as PeftLoraLinear


class JinaWqkvLora(PeftLoraLinear):
    """PEFT LoRA wrapper for a LinearResidual base layer.

    The wrapper overrides `forward` only. Adapter state, merge behaviour, and
    the saved adapter format come from PEFT's Linear LoRA. A merge touches
    `base_layer.weight`. A saved adapter stays in standard PEFT format.
    """

    def forward(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the LoRA delta to the projection and repack the pair.

        Args:
            x: The input tensor.
            *args: Further positional arguments for the base layer.
            **kwargs: Further keyword arguments for the base layer.

        Returns:
            The projection with the LoRA delta applied, and the residual.

        Raises:
            TypeError: If the base layer does not return a pair.
            NotImplementedError: If `adapter_names` requests mixed-batch
                inference.
        """
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        variant_kwargs = {k: kwargs.pop(k, None) for k in VARIANT_KWARG_KEYS}

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self._base_pair(x, *args, **kwargs)

        if adapter_names is not None:
            raise NotImplementedError(
                "adapter_names mixed-batch inference is not supported for JinaWqkvLora."
            )

        y, residual = self._base_pair(x, *args, **kwargs)
        if self.merged:
            return y, residual

        result = y
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_A = self.lora_A[active_adapter]  # noqa: N806
            lora_B = self.lora_B[active_adapter]  # noqa: N806
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x_cast = self._cast_input_dtype(x, lora_A.weight.dtype)
            if active_adapter not in self.lora_variant:
                result = result + lora_B(lora_A(dropout(x_cast))) * scaling
            else:
                result = self.lora_variant[active_adapter].forward(
                    self,
                    active_adapter=active_adapter,
                    x=x_cast,
                    result=result,
                    **variant_kwargs,
                    **kwargs,
                )
        return result.to(torch_result_dtype), residual

    def _base_pair(
        self, x: torch.Tensor, *args: Any, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Call the base layer and unpack its projection and residual.

        Args:
            x: The input tensor.
            *args: Further positional arguments for the base layer.
            **kwargs: Further keyword arguments for the base layer.

        Returns:
            The projection and the residual.

        Raises:
            TypeError: If the base layer does not return a pair.
        """
        base_out = self.base_layer(x, *args, **kwargs)
        if not (isinstance(base_out, tuple) and len(base_out) == 2):
            raise TypeError(
                f"base_layer did not match (y, residual), got {type(base_out).__name__}"
            )
        return base_out


def _is_linear_residual(module: nn.Module) -> bool:
    """Report whether a module is one of jina's LinearResidual layers.

    Args:
        module: The module to test.

    Returns:
        True if the module is a LinearResidual.
    """
    return isinstance(module, nn.Linear) and type(module).__name__ == "LinearResidual"


def patch_jina_lora(model: nn.Module) -> int:
    """Replace the PEFT wrapper of every LinearResidual layer with JinaWqkvLora.

    Args:
        model: The PEFT-wrapped model to patch.

    Returns:
        The number of wrappers whose class changed.
    """
    swapped = 0
    for _name, module in model.named_modules():
        if (
            isinstance(module, PeftLoraLinear)
            and type(module) is not JinaWqkvLora
            and _is_linear_residual(module.base_layer)
        ):
            module.__class__ = JinaWqkvLora
            swapped += 1
    return swapped
