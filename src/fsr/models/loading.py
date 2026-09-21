"""Model and tokenizer loading for the mitigation phase."""

from __future__ import annotations

from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from fsr.models.jina_lora import patch_jina_lora
from fsr.models.mxbai_tanh_patch import apply_mxbai_tanh_patch

DEFAULT_DTYPE = torch.float32


def load_tokenizer(model_name: str, trust_remote_code: bool = True) -> Any:
    """Load the tokenizer for a model.

    Args:
        model_name: The model identifier.
        trust_remote_code: Whether to run the vendor's remote modeling code.

    Returns:
        The tokenizer.
    """
    return AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )


def load_model(
    model_name: str,
    device: str,
    trust_remote_code: bool = True,
    lora_adapter_path: str | None = None,
    eager_attn: bool = False,
    dtype: torch.dtype | str = DEFAULT_DTYPE,
    tanh_head: bool = False,
) -> Any:
    """Load a model in evaluation mode, with an optional LoRA adapter.

    Args:
        model_name: The model identifier.
        device: The device to move the model to.
        trust_remote_code: Whether to run the vendor's remote modeling code.
        lora_adapter_path: A PEFT adapter to load on top of the base model.
        eager_attn: Whether to request the eager attention kernel in place of
            the scaled dot-product kernel.
        dtype: The load dtype, default is float32.
        tanh_head: Whether to replace a single-Linear classifier head with the
            two-layer tanh head. The mxbai tanh adapters require it.

    Returns:
        The model, in evaluation mode.
    """
    dtype_str = (
        str(dtype).removeprefix("torch.") if isinstance(dtype, torch.dtype) else dtype
    )
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype_str,
    }
    if eager_attn:
        load_kwargs["attn_implementation"] = "eager"
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        ).to(device)
    except TypeError:
        load_kwargs.pop("attn_implementation", None)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **load_kwargs
        ).to(device)
    if tanh_head:
        apply_mxbai_tanh_patch(model)
    if lora_adapter_path is not None:
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        patch_jina_lora(model)
    return model.eval()
