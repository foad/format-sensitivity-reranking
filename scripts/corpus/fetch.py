"""Fetch the Natural Questions examples that are answered inside an infobox.

Streams a Natural Questions split and keeps each example whose first
annotator's long answer falls inside an infobox. Writes one JSON file per split
to `data/nq/matched_{split}.json`.

Each record stores the raw infobox HTML and the raw post-infobox HTML.
`scripts/corpus/parse.py` extracts the metadata pairs and the body from
those fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from fsr.corpus.config import DEFAULT
from fsr.corpus.infobox import find_infobox_ranges, in_any_range

DEFAULT_OUT_DIR = Path("data") / "nq"


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


def fetch_split(
    split: str,
    out_path: Path,
    n_limit: int,
    dataset: str = DEFAULT.dataset,
    revision: str | None = DEFAULT.dataset_revision,
) -> None:
    """Stream one split and write its infobox-answered examples to out_path.

    Args:
        split: The Natural Questions split name.
        out_path: The JSON file to write.
        n_limit: The cap on examples scanned. 0 scans the whole split.
        dataset: The Hugging Face dataset to stream.
        revision: The dataset revision to pin. None uses the default branch.
    """
    pinned = revision or "unpinned"
    print(
        f"Streaming {dataset}@{pinned} split={split} (limit={n_limit or 'no limit'})..."
    )
    ds = load_dataset(dataset, split=split, streaming=True, revision=revision)

    matched = []
    seen = 0
    for ex in ds:
        seen += 1
        if n_limit and seen > n_limit:
            break

        record = match_example(ex)
        if record is not None:
            matched.append(record)

        if seen % 1000 == 0:
            print(f"  scanned {seen:,}, matched {len(matched):,}")

    print(f"Done: scanned {seen:,}, matched {len(matched):,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "split": split,
                "n_scanned": seen,
                "n_matched": len(matched),
                "dataset": dataset,
                "dataset_revision": revision,
                "records": matched,
            },
            indent=2,
        )
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(matched):,} records -> {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    """Build the cache for each requested split."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
    )
    ap.add_argument(
        "--n-limit",
        type=int,
        default=0,
        help="Cap on examples scanned per split (0 = no cap)",
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--dataset", default=DEFAULT.dataset)
    ap.add_argument(
        "--revision",
        default=DEFAULT.dataset_revision,
        help="Dataset revision to pin (commit SHA, tag, or branch)",
    )
    args = ap.parse_args()

    for split in args.splits:
        out_path = args.out_dir / f"matched_{split}.json"
        if out_path.exists():
            print(f"Skipping {split}: cache already exists at {out_path}")
            continue
        fetch_split(split, out_path, args.n_limit, args.dataset, args.revision)


if __name__ == "__main__":
    main()
