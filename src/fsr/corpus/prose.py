"""Selection of the NQ examples answered in prose, not an infobox."""

from __future__ import annotations

import html
from collections.abc import Iterator
from dataclasses import dataclass, field

from datasets import load_dataset

from fsr.corpus.config import DEFAULT
from fsr.corpus.infobox import find_infobox_ranges, in_any_range
from fsr.corpus.nq import PROGRESS_EVERY, short_answer_texts
from fsr.corpus.wikitext import TAG_RE, WS_RE

NO_LONG_ANSWER = "no_long_answer"
ANSWER_IN_INFOBOX = "answer_in_infobox"
TOO_SHORT = "too_short"
NO_VERIFIED_ANSWER = "no_verified_answer"
SKIP_REASONS = (NO_LONG_ANSWER, ANSWER_IN_INFOBOX, TOO_SHORT, NO_VERIFIED_ANSWER)


@dataclass
class ProseStats:
    """The running counts of one prose scan.

    Attributes:
        scanned: The examples read from the dataset.
        matched: The examples kept.
        skipped: The count of each reason an example was rejected.
    """

    scanned: int = 0
    matched: int = 0
    skipped: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(SKIP_REASONS, 0)
    )

    def skip(self, reason: str) -> None:
        """Count one rejection.

        Args:
            reason: One of SKIP_REASONS.
        """
        self.skipped[reason] += 1


def clean_html_to_text(html_slice: str) -> str:
    """Turn an HTML fragment into plain prose.

    Args:
        html_slice: The fragment.

    Returns:
        The cleaned plain text.
    """
    text = TAG_RE.sub(" ", html_slice)
    text = html.unescape(text)
    return WS_RE.sub(" ", text).strip()


def answer_in_text(text: str, short_answers: list[str]) -> bool:
    """Report whether a verified short answer appears in the passage.

    Args:
        text: The passage prose.
        short_answers: The first annotator's short answers.

    Returns:
        True if any short answer appears in the passage, False otherwise.
    """
    if not short_answers:
        return False
    haystack = text.lower()
    return any(answer and answer.lower() in haystack for answer in short_answers)


def match_example(
    ex: dict,
    min_body_chars: int = DEFAULT.min_body_chars,
    body_chars: int = DEFAULT.body_chars,
    stats: ProseStats | None = None,
) -> dict | None:
    """Build a prose record for one example, or reject the example.

    Args:
        ex: One Natural Questions example.
        min_body_chars: The shortest passage kept.
        body_chars: The length at which a passage is cut.
        stats: Counters the scan updates as it runs.

    Returns:
        The record, or None when the example does not match.
    """
    html_bytes = ex["document"]["html"].encode("utf-8", errors="replace")

    longs = ex["annotations"].get("long_answer", [])
    if not (isinstance(longs, list) and longs):
        _count(stats, NO_LONG_ANSWER)
        return None
    first_long = longs[0]
    if not (isinstance(first_long, dict) and first_long.get("start_byte", -1) >= 0):
        _count(stats, NO_LONG_ANSWER)
        return None

    start, end = int(first_long["start_byte"]), int(first_long["end_byte"])
    ranges = find_infobox_ranges(html_bytes)
    if ranges and in_any_range(start, end, ranges):
        _count(stats, ANSWER_IN_INFOBOX)
        return None

    raw = html_bytes[start:end].decode("utf-8", errors="replace")
    text = clean_html_to_text(raw)
    if len(text) < min_body_chars:
        _count(stats, TOO_SHORT)
        return None

    truncated = text[:body_chars]
    answers = short_answer_texts(ex["annotations"])
    if not answer_in_text(truncated, answers):
        _count(stats, NO_VERIFIED_ANSWER)
        return None

    return {
        "id": str(ex.get("id", "")),
        "title": ex["document"].get("title", ""),
        "question": ex["question"]["text"],
        "long_answer_text": truncated,
        "long_answer_chars": len(text),
        "short_answers": answers,
    }


def _count(stats: ProseStats | None, reason: str) -> None:
    """Count one rejection when a scan is running."""
    if stats is not None:
        stats.skip(reason)


def iter_prose(
    split: str = "validation",
    n_limit: int = 0,
    min_body_chars: int = DEFAULT.min_body_chars,
    body_chars: int = DEFAULT.body_chars,
    dataset: str = DEFAULT.dataset,
    revision: str | None = DEFAULT.dataset_revision,
    stats: ProseStats | None = None,
    verbose: bool = True,
) -> Iterator[dict]:
    """Stream one split and yield each prose-answered record.

    Args:
        split: The Natural Questions split name.
        n_limit: The cap on examples scanned. 0 scans the whole split.
        min_body_chars: The shortest passage kept.
        body_chars: The length at which a passage is cut.
        dataset: The Hugging Face dataset to stream.
        revision: The dataset revision to pin. None uses the default branch.
        stats: Counters the scan updates as it runs.
        verbose: Whether to report progress while scanning.

    Yields:
        One record for each matching example.
    """
    stats = stats if stats is not None else ProseStats()
    if verbose:
        print(
            f"Streaming {dataset}@{revision or 'unpinned'} split={split} "
            f"(limit={n_limit or 'no limit'})..."
        )
    ds = load_dataset(dataset, split=split, streaming=True, revision=revision)

    for ex in ds:
        stats.scanned += 1
        if n_limit and stats.scanned > n_limit:
            break
        record = match_example(ex, min_body_chars, body_chars, stats)
        if record is not None:
            stats.matched += 1
            yield record
        if verbose and stats.scanned % PROGRESS_EVERY == 0:
            print(f"  scanned {stats.scanned:,}, matched {stats.matched:,}")

    if verbose:
        print(f"Done: scanned {stats.scanned:,}, matched {stats.matched:,}")
