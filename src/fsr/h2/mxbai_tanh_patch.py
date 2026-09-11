"""Classifier patch that gives the mxbai reranker a jina-shaped head."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

RERANKER_OUT_FEATURES = 1


def apply_mxbai_tanh_patch(model: Any) -> None:
    """Replace a single-Linear classifier head with a two-layer tanh head.

    The new head is `Sequential(Linear(D, D), Tanh(), Linear(D, 1))`. The added
    dense layer takes PyTorch's default initialisation. The output projection
    copies the weight and the bias of the original classifier.

    Args:
        model: The model whose `classifier` attribute is replaced in place.

    Raises:
        RuntimeError: If the model has no classifier, if the classifier is not
            a plain `nn.Linear`, or if it does not produce a single output.
    """
    if not hasattr(model, "classifier"):
        raise RuntimeError("model has no `classifier` attribute")
    classifier = model.classifier
    if not isinstance(classifier, nn.Linear):
        raise RuntimeError(
            f"model.classifier is {type(classifier).__name__}, expected nn.Linear"
        )
    in_features, out_features = classifier.in_features, classifier.out_features
    if out_features != RERANKER_OUT_FEATURES:
        raise RuntimeError(
            f"expected classifier out_features={RERANKER_OUT_FEATURES} "
            f"(reranker head), got {out_features}"
        )

    device, dtype = classifier.weight.device, classifier.weight.dtype
    dense = nn.Linear(in_features, in_features, bias=True).to(
        device=device, dtype=dtype
    )
    out_proj = nn.Linear(in_features, out_features, bias=True)
    with torch.no_grad():
        out_proj.weight.copy_(classifier.weight)
        out_proj.bias.copy_(classifier.bias)
    out_proj = out_proj.to(device=device, dtype=dtype)

    model.classifier = nn.Sequential(dense, nn.Tanh(), out_proj)
