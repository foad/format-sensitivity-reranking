"""The metadata serialisation formats under study.

Each renderer turns the same `(key, value)` pairs into one surface form and
puts the body prose after it.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence

MetadataPairs = Sequence[tuple[str, str]]
Renderer = Callable[[MetadataPairs, str], str]


def render_yaml(md: MetadataPairs, body: str) -> str:
    """Render the metadata as a YAML front-matter block above the body."""
    return "---\n" + "\n".join(f"{k}: {v}" for k, v in md) + "\n---\n" + body


def render_json(md: MetadataPairs, body: str) -> str:
    """Render the metadata as a single-line JSON object above the body.

    A repeated key keeps only its last value.
    """
    return json.dumps(dict(md)) + "\n" + body


def render_toml(md: MetadataPairs, body: str) -> str:
    """Render the metadata as TOML key-value lines above the body."""
    return "\n".join(f'{k} = "{v}"' for k, v in md) + "\n" + body


def render_inline_kv(md: MetadataPairs, body: str) -> str:
    """Render the metadata as space-separated key=value pairs above the body."""
    return " ".join(f"{k}={v}" for k, v in md) + "\n" + body


def render_markdown(md: MetadataPairs, body: str) -> str:
    """Render the metadata as bold-key Markdown lines above the body."""
    return "\n".join(f"**{k}**: {v}" for k, v in md) + "\n\n" + body


FORMATS: dict[str, Renderer] = {
    "yaml": render_yaml,
    "json": render_json,
    "toml": render_toml,
    "inline_kv": render_inline_kv,
    "markdown": render_markdown,
}
FORMAT_NAMES = tuple(FORMATS.keys())
