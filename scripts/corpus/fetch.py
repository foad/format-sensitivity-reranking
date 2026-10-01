"""Write the raw cache of Natural Questions examples answered inside an infobox.

Each record holds the raw infobox HTML and the raw post-infobox HTML. The cache
is large. `scripts/corpus/parse.py` reads Natural Questions directly by default.
Run this stage to re-parse without re-streaming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.cli import (
    add_common_args,
    record_stage,
    report_written,
    skip_existing,
)
from fsr.corpus.config import DEFAULT
from fsr.corpus.files import write_atomic
from fsr.corpus.nq import ScanStats, iter_matched


def fetch_split(
    split: str,
    out_path: Path,
    limit: int,
    dataset: str = DEFAULT.dataset,
    revision: str | None = DEFAULT.dataset_revision,
) -> int:
    """Stream one split and write its infobox-answered examples to out_path.

    Args:
        split: The Natural Questions split name.
        out_path: The JSON file to write.
        limit: The cap on examples scanned. 0 scans the whole split.
        dataset: The Hugging Face dataset to stream.
        revision: The dataset revision to pin. None uses the default branch.

    Returns:
        The number of records written.
    """
    stats = ScanStats()
    matched = list(iter_matched(split, limit, dataset, revision, stats))

    write_atomic(
        out_path,
        json.dumps(
            {
                "split": split,
                "n_scanned": stats.scanned,
                "n_matched": stats.matched,
                "dataset": dataset,
                "dataset_revision": revision,
                "records": matched,
            },
            indent=2,
        ),
    )
    report_written(out_path, len(matched))
    return len(matched)


def main() -> None:
    """Write the raw cache for each requested split."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
    )
    ap.add_argument("--dataset", default=DEFAULT.dataset)
    ap.add_argument(
        "--revision",
        default=DEFAULT.dataset_revision,
        help="Dataset revision to pin (commit SHA, tag, or branch)",
    )
    add_common_args(
        ap, "Cap on examples scanned per split, before matching (0 = no cap)"
    )
    args = ap.parse_args()

    for split in args.splits:
        stage = f"fetch_{split}"
        out_path = args.data_root / f"matched_{split}.json"
        if skip_existing(stage, [out_path], args.force):
            continue
        written = fetch_split(split, out_path, args.limit, args.dataset, args.revision)
        record_stage(
            args.data_root,
            stage,
            {out_path: written},
            {"dataset": args.dataset, "dataset_revision": args.revision},
        )


if __name__ == "__main__":
    main()
