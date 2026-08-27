"""Parse the cached Wikipedia HTML into metadata pairs and body text.

Reads `data/nq/matched_{split}.json`, parses each record's infobox
and prose, applies the quality gate, and writes `data/nq/parsed_{split}.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fsr.h1.wikitext import (
    DEFAULT_BODY_CHARS,
    DEFAULT_MIN_BODY_CHARS,
    DEFAULT_MIN_INFOBOX_CHARS,
    DEFAULT_MIN_PAIRS,
    extract_body,
    parse_infobox,
    quality_check,
)

DEFAULT_IN_DIR = Path("data") / "nq"
DEFAULT_OUT_DIR = Path("data") / "nq"
STRICT_GATE_SAMPLE = 5
DRY_RUN_BODY_CHARS = 1000


def parse_record(
    rec: dict,
    target_body_chars: int = DEFAULT_BODY_CHARS,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_body: int = DEFAULT_MIN_BODY_CHARS,
    min_infobox: int = DEFAULT_MIN_INFOBOX_CHARS,
) -> tuple[dict, Counter]:
    """Parse one cache record and apply the quality gate.

    Args:
        rec: One record from the matched cache.
        target_body_chars: The body length at which collection stops.
        min_pairs: The fewest pairs a record may hold.
        min_body: The shortest body a record may hold.
        min_infobox: The shortest raw infobox HTML a record may hold.

    Returns:
        The parsed record with its quality_flags, and a count of dropped rows by
        reason.
    """
    pairs, parser_stats = parse_infobox(rec["infobox_html_raw"])
    body = extract_body(rec["post_infobox_html_raw"], target_chars=target_body_chars)
    parsed = {
        "id": rec["id"],
        "title": rec["title"],
        "question": rec["question"],
        "pairs": pairs,
        "body": body,
    }
    parsed["quality_flags"] = quality_check(
        parsed, rec, min_pairs, min_body, min_infobox
    )
    return parsed, parser_stats


def parse_records(
    raw_records: list[dict],
    target_body_chars: int = DEFAULT_BODY_CHARS,
    min_pairs: int = DEFAULT_MIN_PAIRS,
    min_body: int = DEFAULT_MIN_BODY_CHARS,
    min_infobox: int = DEFAULT_MIN_INFOBOX_CHARS,
) -> tuple[list[dict], Counter, Counter]:
    """Parse every cache record and total the parser and quality counts.

    Args:
        raw_records: The records from the matched cache.
        target_body_chars: The body length at which collection stops.
        min_pairs: The fewest pairs a record may hold.
        min_body: The shortest body a record may hold.
        min_infobox: The shortest raw infobox HTML a record may hold.

    Returns:
        The parsed records, the dropped-row counts, and the quality-flag counts.
    """
    parsed_records: list[dict] = []
    parser_totals: Counter = Counter()
    quality_totals: Counter = Counter()
    for rec in raw_records:
        parsed, parser_stats = parse_record(
            rec, target_body_chars, min_pairs, min_body, min_infobox
        )
        parser_totals.update(parser_stats)
        quality_totals.update(parsed["quality_flags"])
        parsed_records.append(parsed)
    return parsed_records, parser_totals, quality_totals


def strict_gate_available(raw_records: list[dict]) -> bool:
    """Report whether the cache carries the short answers the strict gate needs."""
    return any("short_answers" in r for r in raw_records[:STRICT_GATE_SAMPLE])


def write_dry_run(
    out_path: Path, parsed_records: list[dict], raw_records: list[dict], n: int
) -> None:
    """Write a readable sample of parsed records for inspection.

    Args:
        out_path: The text file to write.
        parsed_records: The parsed records.
        raw_records: The matching cache records.
        n: The number of samples to write.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rule = "=" * 80
    with out_path.open("w") as fh:
        for i, r in enumerate(parsed_records[:n]):
            src = raw_records[i]
            fh.write(f"{rule}\n")
            fh.write(
                f"{i + 1}/{n}   quality_flags={r['quality_flags'] or '(passed)'}\n"
            )
            fh.write(f"{rule}\n")
            fh.write(f"title:    {r['title']}\n")
            fh.write(f"question: {r['question']}\n")
            fh.write(f"n_pairs:  {len(r['pairs'])}\n")
            fh.write(f"raw_infobox_bytes: {len(src['infobox_html_raw'])}\n")
            fh.write(f"body ({len(r['body'])} chars):\n")
            fh.write(r["body"][:DRY_RUN_BODY_CHARS])
            if len(r["body"]) > DRY_RUN_BODY_CHARS:
                extra = len(r["body"]) - DRY_RUN_BODY_CHARS
                fh.write(f"\n... [{extra} more chars truncated in this dump]")
            fh.write("\npairs:\n")
            for k, v in r["pairs"]:
                fh.write(f"  {k!r}: {v!r}\n")
            fh.write("\n")


def main() -> None:
    """Parse each requested split and write its parsed cache."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
    )
    ap.add_argument("--target-body-chars", type=int, default=DEFAULT_BODY_CHARS)
    ap.add_argument("--min-pairs", type=int, default=DEFAULT_MIN_PAIRS)
    ap.add_argument("--min-body", type=int, default=DEFAULT_MIN_BODY_CHARS)
    ap.add_argument("--min-infobox", type=int, default=DEFAULT_MIN_INFOBOX_CHARS)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="If >0, write N parsed samples to --dry-run-out and exit",
    )
    ap.add_argument(
        "--dry-run-out", type=Path, default=DEFAULT_OUT_DIR / "parse_dry_run.txt"
    )
    args = ap.parse_args()

    for split_idx, split in enumerate(args.splits):
        in_path = args.in_dir / f"matched_{split}.json"
        out_path = args.out_dir / f"parsed_{split}.json"
        if not in_path.exists():
            print(f"Skipping {split}: {in_path} does not exist")
            continue

        print(f"\n=== {split} ===")
        print(f"Loading {in_path} ({in_path.stat().st_size / 1e6:.1f} MB)...")
        data = json.loads(in_path.read_text())
        raw_records = data["records"]
        print(f"  {len(raw_records):,} raw matched records")

        parsed_records, parser_totals, quality_totals = parse_records(
            raw_records,
            args.target_body_chars,
            args.min_pairs,
            args.min_body,
            args.min_infobox,
        )
        n_passed = sum(1 for r in parsed_records if not r["quality_flags"])
        strict_gate_active = strict_gate_available(raw_records)

        print("\nParser stats (totals across all records):")
        for k, v in sorted(parser_totals.items()):
            print(f"  {k}: {v:,}")
        gate = "strict + loose" if strict_gate_active else "loose only"
        print(f"\nQuality gate ({gate}):")
        share = 100 * n_passed / len(raw_records) if raw_records else 0.0
        print(f"  passed: {n_passed:,} ({share:.1f}%)")
        for f, n in sorted(quality_totals.items()):
            print(f"  flagged {f}: {n:,}")

        if args.dry_run > 0 and split_idx == 0:
            n = min(args.dry_run, len(parsed_records))
            write_dry_run(args.dry_run_out, parsed_records, raw_records, n)
            print(f"\nDry-run: wrote {n} parsed samples -> {args.dry_run_out}")
            print("(No parsed_*.json written. Re-run without --dry-run.)")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "source_cache": in_path.name,
                    "n_records": len(parsed_records),
                    "n_passed_quality": n_passed,
                    "target_body_chars": args.target_body_chars,
                    "min_pairs": args.min_pairs,
                    "min_body": args.min_body,
                    "min_infobox": args.min_infobox,
                    "parser_stats": dict(parser_totals),
                    "quality_flag_counts": dict(quality_totals),
                    "strict_gate_active": strict_gate_active,
                    "records": parsed_records,
                },
                indent=2,
            )
        )
        size_mb = out_path.stat().st_size / 1e6
        print(
            f"\nWrote {len(parsed_records):,} records -> {out_path}  ({size_mb:.1f} MB)"
        )


if __name__ == "__main__":
    main()
