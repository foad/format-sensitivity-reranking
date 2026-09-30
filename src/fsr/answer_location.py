"""Classification of each corpus record by where its short answer appears."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

CATEGORIES = ("metadata_only", "body_only", "both", "no_answer", "neither")


def _contains_any(needles: Sequence[str], haystack: str) -> bool:
    """Report whether the text holds any of the strings, ignoring case."""
    if not haystack:
        return False
    hay = haystack.lower()
    return any(n and n.lower() in hay for n in needles)


def classify_record(record: Mapping[str, Any]) -> str:
    """Return where the short answer of one parsed record appears.

    Args:
        record: A parsed record with its pairs, body, and short answers.

    Returns:
        One of metadata_only, body_only, both, no_answer, or neither.
    """
    answers = [
        s
        for s in (record.get("short_answers") or [])
        if isinstance(s, str) and s.strip()
    ]
    if not answers:
        return "no_answer"
    in_metadata = _contains_any(answers, " ".join(str(v) for _, v in record["pairs"]))
    in_body = _contains_any(answers, record["body"])
    if in_metadata and in_body:
        return "both"
    if in_metadata:
        return "metadata_only"
    if in_body:
        return "body_only"
    return "neither"


def classify_records(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Classify every record that carries a short_answers field.

    Args:
        records: The parsed records to classify.

    Returns:
        The category of each record.
    """
    return {r["id"]: classify_record(r) for r in records if "short_answers" in r}


def category_counts(labels: Iterable[str]) -> dict[str, int]:
    """Total the categories, including the categories that no record holds.

    Args:
        labels: The category of each record.

    Returns:
        The count of each category, in CATEGORIES order.
    """
    counts = dict.fromkeys(CATEGORIES, 0)
    for label in labels:
        counts[label] += 1
    return counts
