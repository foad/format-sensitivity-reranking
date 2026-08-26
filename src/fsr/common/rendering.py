"""Format renderers, body-token budgeting, and semantic truncation."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, Protocol

MAX_TOKENS = 512
BUFFER_TOKENS = 5
SPECIAL_TOKEN_OVERHEAD = 3
MIN_BODY_TOKENS = 20
SENTENCE_TERMINATORS = frozenset(".!?")
SENTENCE_SEARCH_CHARS = 500
MIN_WORD_BOUNDARY_CHARS = 20

MetadataPairs = Sequence[tuple[str, str]]
Renderer = Callable[[MetadataPairs, str], str]


class Tokenizer(Protocol):
    """The tokenizer interface that this module requires."""

    def __call__(self, text: str, **kwargs: Any) -> Mapping[str, Any]:
        """Tokenize the text and return the encoding.

        Args:
            text: The text to tokenize.
            **kwargs: Tokenizer options. This module passes
                add_special_tokens, return_offsets_mapping, and truncation.

        Returns:
            A mapping that contains input_ids. The mapping also contains
            offset_mapping as (start, end) character pairs when the caller
            requests offsets.
        """
        ...


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


def compute_body_budget(
    question: str,
    pairs: MetadataPairs,
    tokenizers: Mapping[str, Tokenizer],
    *,
    max_tokens: int = MAX_TOKENS,
    formats: Mapping[str, Renderer] | None = None,
) -> tuple[int, str]:
    """Return the smallest body-token budget across the given tokenizers.

    The budget is the remainder of max_tokens after the question, the longest
    rendering of the metadata, the special tokens, and the buffer. The
    special-token count assumes the encoding [CLS] question [SEP] passage [SEP].

    Args:
        question: The query text.
        pairs: The metadata key-value pairs for one record.
        tokenizers: The tokenizers to measure, by name.
        max_tokens: The model context length.
        formats: The renderers to measure. The default is FORMATS.

    Returns:
        The budget, and the name of the tokenizer that gives it. The budget is
        negative if the question, the metadata, and the fixed overhead are
        longer than max_tokens.

    Raises:
        ValueError: If tokenizers is empty.
    """
    if not tokenizers:
        raise ValueError("tokenizers must not be empty")
    renderers = FORMATS if formats is None else formats
    budgets = {}
    for name, tok in tokenizers.items():
        q_tokens = len(tok(question, add_special_tokens=False)["input_ids"])
        max_meta = 0
        for renderer in renderers.values():
            meta_text = renderer(pairs, "")
            m_tokens = len(tok(meta_text, add_special_tokens=False)["input_ids"])
            if m_tokens > max_meta:
                max_meta = m_tokens
        budgets[name] = (
            max_tokens - q_tokens - max_meta - SPECIAL_TOKEN_OVERHEAD - BUFFER_TOKENS
        )
    tightest = min(budgets, key=budgets.get)
    return budgets[tightest], tightest


def truncate_body_semantic(
    body: str,
    budget_tokens: int,
    tokenizer: Tokenizer,
) -> str | None:
    """Truncate the body to the budget at the last sentence boundary.

    If no sentence terminator is in the last SENTENCE_SEARCH_CHARS characters,
    the function cuts at the last word boundary. If that word boundary is not
    more than MIN_WORD_BOUNDARY_CHARS characters into the text, the function
    cuts at the token limit.

    Args:
        body: The body prose.
        budget_tokens: The budget from compute_body_budget.
        tokenizer: The tokenizer that gives the budget.

    Returns:
        The truncated body. The function returns the body unchanged if the
        body is within the budget. The function returns an empty string if the
        body is empty. The function returns None if budget_tokens is less than
        MIN_BODY_TOKENS.
    """
    if budget_tokens < MIN_BODY_TOKENS:
        return None
    if not body:
        return ""
    enc = tokenizer(
        body,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = enc["offset_mapping"]
    if len(offsets) <= budget_tokens:
        return body
    char_end = offsets[budget_tokens - 1][1]
    truncated = body[:char_end]
    search_window = min(SENTENCE_SEARCH_CHARS, len(truncated))
    for i in range(len(truncated) - 1, max(0, len(truncated) - search_window) - 1, -1):
        if truncated[i] not in SENTENCE_TERMINATORS:
            continue
        if i == len(truncated) - 1 or truncated[i + 1] in " \t\n":
            return truncated[: i + 1].rstrip()
    last_space = truncated.rfind(" ")
    if last_space > MIN_WORD_BOUNDARY_CHARS:
        return truncated[:last_space].rstrip()
    return truncated.rstrip()


def prepare_records_with_body(
    records: Iterable[Mapping[str, Any]],
    tokenizers: Mapping[str, Tokenizer],
    *,
    max_tokens: int = MAX_TOKENS,
    formats: Mapping[str, Renderer] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add a shared body budget and a truncated body to each record.

    Args:
        records: Mappings that contain question, pairs, and body.
        tokenizers: The tokenizers to measure, by name.
        max_tokens: The model context length.
        formats: The renderers to measure. The default is FORMATS.

    Returns:
        The kept records, and the statistics. Each kept record is a copy with
        the added keys truncated_body, body_budget_tokens, and
        tightest_tokeniser. The function drops a record if its budget is less
        than MIN_BODY_TOKENS. The statistics contain the count dropped and the
        counts tightest_counts by tokenizer.
    """
    kept = []
    dropped = 0
    tightest_counts: dict[str, int] = {}
    for r in records:
        budget, tightest = compute_body_budget(
            r["question"],
            r["pairs"],
            tokenizers,
            max_tokens=max_tokens,
            formats=formats,
        )
        tightest_counts[tightest] = tightest_counts.get(tightest, 0) + 1
        truncated = truncate_body_semantic(r["body"], budget, tokenizers[tightest])
        if truncated is None:
            dropped += 1
            continue
        r2 = dict(r)
        r2["truncated_body"] = truncated
        r2["body_budget_tokens"] = budget
        r2["tightest_tokeniser"] = tightest
        kept.append(r2)
    return kept, {"dropped": dropped, "tightest_counts": tightest_counts}
