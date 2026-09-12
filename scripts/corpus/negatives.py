"""Mine BM25 hard negatives for every query in every split.

The retrieval corpus is the quality-passed training records only, so a
validation record never appears as a training negative.

Writes `bm25_negatives.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.config import DEFAULT
from fsr.corpus.negatives import build_negatives
from fsr.corpus.splitting import strict_gated

DEFAULT_IN_DIR = Path("data") / "nq"
DEFAULT_SPLIT_DIR = Path("data") / "nq" / "h2_splits"
SPLIT_NAMES = ("train", "dev", "test", "nq_val")


def load_queries(split_dir: Path, names: tuple[str, ...] = SPLIT_NAMES) -> list[dict]:
    """Return every record of the named splits, in split order."""
    return [
        record
        for name in names
        for record in json.loads((split_dir / f"{name}.json").read_text())["records"]
    ]


def main() -> None:
    """Mine negatives for every split and write the cache."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    ap.add_argument("--cache-k", type=int, default=DEFAULT.cache_k)
    ap.add_argument("--seed", type=int, default=DEFAULT.negatives_seed)
    args = ap.parse_args()

    parsed = json.loads((args.in_dir / "parsed_train.json").read_text())
    corpus_records = strict_gated(parsed["records"])
    queries = load_queries(args.split_dir)
    print(f"Queries: {len(queries):,} across {', '.join(SPLIT_NAMES)}")

    negatives, filled_count = build_negatives(
        corpus_records, queries, args.cache_k, args.seed
    )
    print(f"\nQueries needing a random fill: {filled_count:,} / {len(queries):,}")

    out_path = args.split_dir / "bm25_negatives.json"
    out_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "cache_k": args.cache_k,
                "n_queries": len(negatives),
                "n_corpus": len(corpus_records),
                "random_fill_count": filled_count,
                "negatives": negatives,
            }
        )
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
