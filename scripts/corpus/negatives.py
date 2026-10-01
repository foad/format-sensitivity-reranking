"""Mine BM25 hard negatives for every query in every split.

The retrieval corpus is the quality-passed training records only, so a
validation record never appears as a training negative.

Writes `bm25_negatives.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.cli import (
    NEGATIVES_NAME,
    SPLIT_NAMES,
    add_common_args,
    record_stage,
    report_written,
    skip_existing,
    split_dir,
    take,
)
from fsr.corpus.config import DEFAULT
from fsr.corpus.files import write_atomic
from fsr.corpus.negatives import build_negatives
from fsr.corpus.splitting import strict_gated


def load_queries(split_path: Path, names: tuple[str, ...] = SPLIT_NAMES) -> list[dict]:
    """Return every record of the named splits, in split order."""
    return [
        record
        for name in names
        for record in json.loads((split_path / f"{name}.json").read_text())["records"]
    ]


def check_corpus_size(corpus_records: list[dict], cache_k: int) -> None:
    """Reject a corpus that cannot supply the negatives each query needs.

    Args:
        corpus_records: The records to retrieve from.
        cache_k: How many negatives each query needs.

    Raises:
        SystemExit: If the corpus holds too few distinct titles.
    """
    available = len({r["title"] for r in corpus_records}) - 1
    if available < cache_k:
        raise SystemExit(
            f"Corpus holds {available + 1} distinct titles, which gives at most "
            f"{available} negatives per query, short of the {cache_k} requested."
        )


def main() -> None:
    """Mine negatives for every split and write the cache."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-k", type=int, default=DEFAULT.cache_k)
    ap.add_argument("--seed", type=int, default=DEFAULT.negatives_seed)
    add_common_args(ap, "Cap on queries mined, across all splits (0 = no cap)")
    args = ap.parse_args()

    split_path = split_dir(args.data_root)
    out_path = split_path / NEGATIVES_NAME
    if skip_existing("negatives", [out_path], args.force):
        return

    parsed = json.loads((args.data_root / "parsed_train.json").read_text())
    corpus_records = strict_gated(parsed["records"])
    check_corpus_size(corpus_records, args.cache_k)

    queries = take(load_queries(split_path), args.limit)
    print(f"Queries: {len(queries):,} across {', '.join(SPLIT_NAMES)}")

    negatives, filled_count = build_negatives(
        corpus_records, queries, args.cache_k, args.seed
    )
    print(f"\nQueries needing a random fill: {filled_count:,} / {len(queries):,}")

    write_atomic(
        out_path,
        json.dumps(
            {
                "seed": args.seed,
                "cache_k": args.cache_k,
                "n_queries": len(negatives),
                "n_corpus": len(corpus_records),
                "random_fill_count": filled_count,
                "negatives": negatives,
            }
        ),
    )
    report_written(out_path, len(negatives))
    record_stage(
        args.data_root,
        "negatives",
        {out_path: len(negatives)},
        {"negatives_seed": args.seed, "cache_k": args.cache_k},
    )


if __name__ == "__main__":
    main()
