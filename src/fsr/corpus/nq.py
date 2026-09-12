"""Selection of the Natural Questions examples that are answered in an infobox."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from datasets import load_dataset

from fsr.corpus.config import DEFAULT
from fsr.corpus.infobox import find_infobox_ranges, in_any_range

PROGRESS_EVERY = 1000


@dataclass
class ScanStats:
    """The running counts of one scan.

    Attributes:
        scanned: The examples read from the dataset.
        matched: The examples kept.
    """

    scanned: int = 0
    matched: int = 0


def short_answer_texts(annotations: dict) -> list[str]:
    """Return the first annotator's short-answer strings.

    Args:
        annotations: The annotations field of a Natural Questions example.

    Returns:
        The non-empty answer strings. The list is empty when the example has no
        short answer.
    """
    sa_ann = annotations.get("short_answers", [])
    if not (isinstance(sa_ann, list) and sa_ann):
        return []
    first_sa = sa_ann[0]
    if not isinstance(first_sa, dict):
        return []
    texts = first_sa.get("text", [])
    if not isinstance(texts, list):
        return []
    return [t for t in texts if t]


def match_example(ex: dict) -> dict | None:
    """Build a cache record for one example, or reject the example.

    An example matches when its HTML contains an infobox and the first
    annotator's long answer falls inside that infobox.

    Args:
        ex: One Natural Questions example.

    Returns:
        The cache record, or None when the example does not match.
    """
    html_bytes = ex["document"]["html"].encode("utf-8", errors="replace")
    ranges = find_infobox_ranges(html_bytes)
    if not ranges:
        return None

    longs = ex["annotations"].get("long_answer", [])
    if not (isinstance(longs, list) and longs):
        return None
    first_long = longs[0]
    if not (isinstance(first_long, dict) and first_long.get("start_byte", -1) >= 0):
        return None

    ls, le = int(first_long["start_byte"]), int(first_long["end_byte"])
    if not in_any_range(ls, le, ranges):
        return None

    return {
        "id": str(ex.get("id", "")),
        "title": ex["document"].get("title", ""),
        "question": ex["question"]["text"],
        "infobox_html_raw": html_bytes[ls:le].decode("utf-8", errors="replace"),
        "post_infobox_html_raw": html_bytes[le:].decode("utf-8", errors="replace"),
        "short_answers": short_answer_texts(ex["annotations"]),
    }


def iter_matched(
    split: str,
    n_limit: int = 0,
    dataset: str = DEFAULT.dataset,
    revision: str | None = DEFAULT.dataset_revision,
    stats: ScanStats | None = None,
    verbose: bool = True,
) -> Iterator[dict]:
    """Stream one split and yield each matching record.

    Args:
        split: The Natural Questions split name.
        n_limit: The cap on examples scanned. 0 scans the whole split.
        dataset: The Hugging Face dataset to stream.
        revision: The dataset revision to pin. None uses the default branch.
        stats: Counters the scan updates as it runs.
        verbose: Whether to report progress while scanning.

    Yields:
        One cache record for each matching example.
    """
    stats = stats if stats is not None else ScanStats()
    if verbose:
        pinned = revision or "unpinned"
        print(
            f"Streaming {dataset}@{pinned} split={split} "
            f"(limit={n_limit or 'no limit'})..."
        )
    ds = load_dataset(dataset, split=split, streaming=True, revision=revision)

    for ex in ds:
        stats.scanned += 1
        if n_limit and stats.scanned > n_limit:
            break
        record = match_example(ex)
        if record is not None:
            stats.matched += 1
            yield record
        if verbose and stats.scanned % PROGRESS_EVERY == 0:
            print(f"  scanned {stats.scanned:,}, matched {stats.matched:,}")

    if verbose:
        print(f"Done: scanned {stats.scanned:,}, matched {stats.matched:,}")
