"""Attach mined negatives to the prose records for the capability check.

Writes `prose_eval.json`.
"""

from __future__ import annotations

import argparse
import json

from fsr.cli import add_common_args, report_written, skip_existing, take
from fsr.corpus.files import write_atomic
from fsr.corpus.layout import PROSE_NAME, prose_eval_path, prose_path
from fsr.corpus.manifest import record_stage
from fsr.corpus.prose import PROSE_NEGATIVES, PROSE_SEED, build_prose_eval


def main() -> None:
    """Mine negatives for every prose record and write the ranking set."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--negatives",
        type=int,
        default=PROSE_NEGATIVES,
        help="How many negatives each record keeps",
    )
    ap.add_argument("--seed", type=int, default=PROSE_SEED)
    add_common_args(ap, "Cap on prose records read (0 = no cap)")
    args = ap.parse_args()

    out_path = prose_eval_path(args.data_root)
    if skip_existing("prose_eval", [out_path], args.force):
        return

    source = prose_path(args.data_root)
    if not source.exists():
        raise SystemExit(f"no prose records at {source}; run the prose stage first.")

    records = take(json.loads(source.read_text())["records"], args.limit)
    print(f"Loaded {len(records):,} prose records from {source}")

    resolved, filled, n_articles = build_prose_eval(records, args.negatives, args.seed)
    print(f"\n{len(resolved):,} records over {n_articles:,} articles")
    print(f"  needed the random fill: {filled:,}")

    write_atomic(
        out_path,
        json.dumps(
            {
                "source": PROSE_NAME,
                "n_records": len(resolved),
                "k_negatives": args.negatives,
                "seed": args.seed,
                "n_random_fill": filled,
                "n_distinct_articles": n_articles,
                "records": resolved,
            },
            ensure_ascii=False,
        ),
    )
    report_written(out_path, len(resolved))

    record_stage(
        args.data_root,
        "prose_eval",
        {out_path: len(resolved)},
        {
            "prose_negatives": args.negatives,
            "prose_negatives_seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
