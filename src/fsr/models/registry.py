"""The rerankers under study and their details."""

from __future__ import annotations

import argparse
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """A reranker model and its associated training and evaluation settings.

    Attributes:
        slug: The short name used in output paths and adapter directories.
        model_id: The Hugging Face identifier.
        label: The name used in reported tables.
        lora_targets: The attention projection modules a LoRA adapter attaches
            to. The names differ between architectures.
        physical_batch: Training examples the device holds at one time.
        grad_accum: Micro-batches accumulated before the optimiser steps.
        eval_batch: Candidates scored at one time during evaluation.
        tanh_head: Whether the classifier head is replaced before an adapter
            attaches.
    """

    slug: str
    model_id: str
    label: str
    lora_targets: tuple[str, ...]
    physical_batch: int
    grad_accum: int
    eval_batch: int
    tanh_head: bool = False

    @property
    def effective_batch(self) -> int:
        """Return the training examples behind one optimiser step."""
        return self.physical_batch * self.grad_accum


MODELS = (
    Model(
        "minilm_l6",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "MiniLM-L6",
        ("query", "key", "value"),
        physical_batch=4,
        grad_accum=4,
        eval_batch=32,
    ),
    Model(
        "minilm_l12",
        "cross-encoder/ms-marco-MiniLM-L12-v2",
        "MiniLM-L12",
        ("query", "key", "value"),
        physical_batch=4,
        grad_accum=4,
        eval_batch=32,
    ),
    Model(
        "bge_base",
        "BAAI/bge-reranker-base",
        "bge-base",
        ("query", "key", "value"),
        physical_batch=2,
        grad_accum=8,
        eval_batch=16,
    ),
    Model(
        "bge_v2_m3",
        "BAAI/bge-reranker-v2-m3",
        "bge-v2-m3",
        ("query", "key", "value"),
        physical_batch=2,
        grad_accum=8,
        eval_batch=16,
    ),
    Model(
        "mxbai_v1",
        "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-v1",
        ("query_proj", "key_proj", "value_proj"),
        physical_batch=2,
        grad_accum=8,
        eval_batch=16,
    ),
    Model(
        "jina_v2",
        "jinaai/jina-reranker-v2-base-multilingual",
        "jina-v2",
        ("Wqkv",),
        physical_batch=2,
        grad_accum=8,
        eval_batch=16,
    ),
    Model(
        "mxbai_v1_tanh",
        "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-v1-tanh",
        ("query_proj", "key_proj", "value_proj"),
        physical_batch=2,
        grad_accum=8,
        eval_batch=16,
        tanh_head=True,
    ),
)

BASE_MODELS = tuple(m for m in MODELS if not m.tanh_head)
BASE_MODEL_IDS = tuple(m.model_id for m in BASE_MODELS)
_BY_SLUG = {m.slug: m for m in MODELS}


def by_slug(slug: str) -> Model:
    """Return the model with this slug.

    Args:
        slug: The short name.

    Returns:
        The model.

    Raises:
        ValueError: If no model carries the slug.
    """
    if slug not in _BY_SLUG:
        known = ", ".join(sorted(_BY_SLUG))
        raise ValueError(f"unknown model {slug!r}, expected one of: {known}")
    return _BY_SLUG[slug]


def by_id(model_id: str) -> Model | None:
    """Return the first model with this Hugging Face identifier, or None.

    Args:
        model_id: The Hugging Face identifier.

    Returns:
        The model, or None when the identifier is not in the registry.
    """
    return next((m for m in MODELS if m.model_id == model_id), None)


def label_of(model_id: str) -> str:
    """Return the reported label for an identifier, or the identifier itself."""
    model = by_id(model_id)
    return model.label if model else model_id


def config_lines(model: Model) -> list[str]:
    """Return one shell assignment per setting of a model.

    Args:
        model: The model to report.

    Returns:
        Lines a caller reads as shell assignments.
    """
    targets = " ".join(shlex.quote(t) for t in model.lora_targets)
    settings = {
        "MODEL_ID": model.model_id,
        "PHYSICAL_BATCH": str(model.physical_batch),
        "GRAD_ACCUM": str(model.grad_accum),
        "EVAL_BATCH": str(model.eval_batch),
        "TANH_HEAD": "1" if model.tanh_head else "0",
    }
    lines = [f"{name}={shlex.quote(value)}" for name, value in settings.items()]
    lines.insert(1, f"LORA_TARGETS=({targets})")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """Print the roster, or one model's settings.

    Args:
        argv: Command-line arguments. Defaults to the process arguments.

    Returns:
        The exit code. 2 when the slug is not in the registry.
    """
    ap = argparse.ArgumentParser(description="Report the roster of rerankers.")
    what = ap.add_mutually_exclusive_group(required=True)
    what.add_argument(
        "--list", action="store_true", help="Print the slug of every model."
    )
    what.add_argument(
        "--config",
        metavar="SLUG",
        help="Print one model's settings as shell assignments.",
    )
    args = ap.parse_args(argv)

    if args.list:
        for model in MODELS:
            print(model.slug)
        return 0

    try:
        model = by_slug(args.config)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for line in config_lines(model):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
