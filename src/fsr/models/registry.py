"""The rerankers under study, and the names each one is known by."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Model:
    """One reranker.

    Attributes:
        slug: The short name used in output paths and adapter directories.
        model_id: The Hugging Face identifier.
        label: The name used in reported tables.
        lora_targets: The attention projection modules a LoRA adapter attaches
            to. The names differ between architectures.
        tanh_head: Whether the classifier head is replaced before an adapter
            attaches.
    """

    slug: str
    model_id: str
    label: str
    lora_targets: tuple[str, ...]
    tanh_head: bool = False


MODELS = (
    Model(
        "minilm_l6",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "MiniLM-L6",
        ("query", "key", "value"),
    ),
    Model(
        "minilm_l12",
        "cross-encoder/ms-marco-MiniLM-L12-v2",
        "MiniLM-L12",
        ("query", "key", "value"),
    ),
    Model(
        "bge_base",
        "BAAI/bge-reranker-base",
        "bge-base",
        ("query", "key", "value"),
    ),
    Model(
        "bge_v2_m3",
        "BAAI/bge-reranker-v2-m3",
        "bge-v2-m3",
        ("query", "key", "value"),
    ),
    Model(
        "mxbai_v1",
        "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-v1",
        ("query_proj", "key_proj", "value_proj"),
    ),
    Model(
        "jina_v2",
        "jinaai/jina-reranker-v2-base-multilingual",
        "jina-v2",
        ("Wqkv",),
    ),
    Model(
        "mxbai_v1_tanh",
        "mixedbread-ai/mxbai-rerank-base-v1",
        "mxbai-v1-tanh",
        ("query_proj", "key_proj", "value_proj"),
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
