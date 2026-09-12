"""Write the raw cache of Natural Questions examples answered inside an infobox.

Each record holds the raw infobox HTML and the raw post-infobox HTML. The cache
is large. `scripts/corpus/parse.py` reads Natural Questions directly by default.
Run this stage to re-parse without re-streaming.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.config import DEFAULT
from fsr.corpus.nq import ScanStats, iter_matched

DEFAULT_OUT_DIR = Path("data") / "nq"


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
    stats = ScanStats()
    matched = list(iter_matched(split, n_limit, dataset, revision, stats))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
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
        )
    )
    size_mb = out_path.stat().st_size / 1e6
    print(f"Wrote {len(matched):,} records -> {out_path}  ({size_mb:.1f} MB)")


def main() -> None:
    """Write the raw cache for each requested split."""
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
