"""Script to extract and write prose-answered records from NQ.

Writes `prose.json`.
"""

from __future__ import annotations

import argparse
import json

from fsr.cli import add_common_args, report_written, skip_existing
from fsr.corpus.config import DEFAULT
from fsr.corpus.files import write_atomic
from fsr.corpus.layout import prose_path
from fsr.corpus.manifest import record_stage
from fsr.corpus.prose import ProseStats, iter_prose

SPLIT = "validation"


def main() -> None:
    """Stream the validation split and write the prose-answered records."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--min-body-chars",
        type=int,
        default=DEFAULT.min_body_chars,
        help="The shortest passage kept",
    )
    ap.add_argument(
        "--body-chars",
        type=int,
        default=DEFAULT.body_chars,
        help="The length at which a passage is cut",
    )
    ap.add_argument("--revision", default=DEFAULT.dataset_revision)
    add_common_args(ap, "Cap on examples scanned (0 = no cap)")
    args = ap.parse_args()

    out_path = prose_path(args.data_root)
    if skip_existing("prose", [out_path], args.force):
        return

    stats = ProseStats()
    records = list(
        iter_prose(
            SPLIT,
            args.limit,
            args.min_body_chars,
            args.body_chars,
            revision=args.revision,
            stats=stats,
        )
    )
    print(f"\nkept {len(records):,} of {stats.scanned:,} scanned")
    for reason, count in stats.skipped.items():
        print(f"  {reason}: {count:,}")

    if not records:
        raise SystemExit("no record passed the prose gates; nothing to write.")

    write_atomic(
        out_path,
        json.dumps(
            {
                "split": SPLIT,
                "n_scanned": stats.scanned,
                "n_records": len(records),
                "skipped": stats.skipped,
                "records": records,
            },
            ensure_ascii=False,
        ),
    )
    report_written(out_path, len(records))

    record_stage(
        args.data_root,
        "prose",
        {out_path: len(records)},
        {
            "dataset": DEFAULT.dataset,
            "dataset_revision": args.revision,
            "split": SPLIT,
            "min_body_chars": args.min_body_chars,
            "body_chars": args.body_chars,
        },
    )


if __name__ == "__main__":
    main()
