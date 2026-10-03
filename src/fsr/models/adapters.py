"""Construction of the LoRA-adapted model a training run updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from peft import LoraConfig, get_peft_model

from fsr.models.jina_lora import patch_jina_lora
from fsr.models.loading import load_model

DEFAULT_LORA_RANK = 16
DEFAULT_LORA_ALPHA = 32
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_LORA_TARGETS = ("query", "key", "value")
TASK_TYPE = "SEQ_CLS"


@dataclass(frozen=True)
class AdaptedModel:
    """A model wrapped for training, and what the wrapping produced.

    Attributes:
        model: The adapted model, in training mode.
        jina_patched: The number of fused projection wrappers replaced.
        trainable: The parameters the optimiser updates.
        total: The parameters of the whole model.
        gradient_checkpointing: Whether activations are recomputed.
    """

    model: Any
    jina_patched: int
    trainable: int
    total: int
    gradient_checkpointing: bool

    @property
    def trainable_fraction(self) -> float:
        """Return the share of parameters the optimiser updates."""
        return self.trainable / self.total if self.total else 0.0


def lora_config(
    targets: tuple[str, ...] | list[str] = DEFAULT_LORA_TARGETS,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
) -> LoraConfig:
    """Return the adapter configuration for a sequence-classification model.

    Args:
        targets: The module name patterns an adapter attaches to.
        rank: The adapter rank.
        alpha: The adapter scaling factor.
        dropout: The adapter dropout probability.

    Returns:
        The configuration for the LoRA adapter.
    """
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(targets),
        bias="none",
        task_type=TASK_TYPE,
    )


def require_fp32(model: torch.nn.Module) -> None:
    """Refuse a model that carries half-precision parameters.

    Args:
        model: The model to check.

    Raises:
        ValueError: If any parameter is float16. AdamW needs float32.
    """
    dtypes = {p.dtype for p in model.parameters()}
    if torch.float16 in dtypes:
        raise ValueError(
            f"model holds float16 parameters {sorted(str(d) for d in dtypes)}; "
            "training needs float32"
        )


def parameter_counts(model: torch.nn.Module) -> tuple[int, int]:
    """Return the trainable and the total parameter count.

    Args:
        model: The model to count.

    Returns:
        The trainable count and the total count.
    """
    total = 0
    trainable = 0
    for param in model.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def adapt(
    base: torch.nn.Module,
    config: LoraConfig,
    *,
    gradient_checkpointing: bool = True,
) -> AdaptedModel:
    """Attach an adapter to a loaded base model and put it in training mode.

    Args:
        base: The loaded base model.
        config: The adapter configuration.
        gradient_checkpointing: Whether to recompute activations in the
            backward pass.

    Returns:
        The adapted model and its counts.

    Raises:
        ValueError: If the model carries half-precision parameters.
    """
    base.train()
    model = get_peft_model(base, config)
    patched = patch_jina_lora(model)
    if gradient_checkpointing:
        # PEFT needs a gradient on the input embeddings.
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    require_fp32(model)
    trainable, total = parameter_counts(model)
    return AdaptedModel(
        model=model,
        jina_patched=patched,
        trainable=trainable,
        total=total,
        gradient_checkpointing=gradient_checkpointing,
    )


def build_adapted_model(
    model_id: str,
    device: str,
    *,
    lora_targets: tuple[str, ...] | list[str] = DEFAULT_LORA_TARGETS,
    rank: int = DEFAULT_LORA_RANK,
    alpha: int = DEFAULT_LORA_ALPHA,
    dropout: float = DEFAULT_LORA_DROPOUT,
    tanh_head: bool = False,
    eager_attn: bool = False,
    gradient_checkpointing: bool = True,
) -> AdaptedModel:
    """Load a base model and attach a training adapter to it.

    Args:
        model_id: The Hugging Face identifier.
        device: The device to load onto.
        lora_targets: The module name patterns an adapter attaches to.
        rank: The adapter rank.
        alpha: The adapter scaling factor.
        dropout: The adapter dropout probability.
        tanh_head: Whether to replace the classifier head before the adapter
            attaches.
        eager_attn: Whether to request the eager attention kernel.
        gradient_checkpointing: Whether the model should use gradient checkpointing.

    Returns:
        The adapted model and its counts.
    """
    base = load_model(
        model_id,
        device,
        eager_attn=eager_attn,
        tanh_head=tanh_head,
    )
    config = lora_config(lora_targets, rank, alpha, dropout)
    return adapt(base, config, gradient_checkpointing=gradient_checkpointing)
