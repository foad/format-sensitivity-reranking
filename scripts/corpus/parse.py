"""Parse Wikipedia HTML into metadata pairs and body text.

Reads Natural Questions directly by default.

Writes `parsed_{split}.json`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from fsr.corpus.cli import (
    add_common_args,
    record_stage,
    report_written,
    skip_existing,
    take,
)
from fsr.corpus.config import DEFAULT
from fsr.corpus.nq import ScanStats, iter_matched
from fsr.corpus.wikitext import extract_body, parse_infobox, quality_check

STRICT_GATE_SAMPLE = 5
DRY_RUN_BODY_CHARS = 1000


def parse_record(
    rec: dict,
    target_body_chars: int = DEFAULT.body_chars,
    min_pairs: int = DEFAULT.min_pairs,
    min_body: int = DEFAULT.min_body_chars,
    min_infobox: int = DEFAULT.min_infobox_chars,
) -> tuple[dict, Counter]:
    """Parse one cache record and apply the quality gate.

    Args:
        rec: One record from the matched cache.
        target_body_chars: The body length at which collection stops.
        min_pairs: The fewest pairs a record may hold.
        min_body: The shortest body a record may hold.
        min_infobox: The shortest raw infobox HTML a record may hold.

    Returns:
        The parsed record with its quality_flags, a count of dropped rows by
        reason, and the source short answers.
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
    if "short_answers" in rec:
        parsed["short_answers"] = rec["short_answers"]
    parsed["quality_flags"] = quality_check(
        parsed, rec, min_pairs, min_body, min_infobox
    )
    return parsed, parser_stats


def parse_records(
    raw_records: Iterable[dict],
    target_body_chars: int = DEFAULT.body_chars,
    min_pairs: int = DEFAULT.min_pairs,
    min_body: int = DEFAULT.min_body_chars,
    min_infobox: int = DEFAULT.min_infobox_chars,
    keep_sources: int = 0,
) -> tuple[list[dict], Counter, Counter, list[dict]]:
    """Parse every cache record and total the parser and quality counts.

    Args:
        raw_records: The records to parse, from a cache or a live stream.
        target_body_chars: The body length at which collection stops.
        min_pairs: The fewest pairs a record may hold.
        min_body: The shortest body a record may hold.
        min_infobox: The shortest raw infobox HTML a record may hold.
        keep_sources: How many source records to retain for the dry run.

    Returns:
        The parsed records, the dropped-row counts, the quality-flag counts,
        and the retained sources.
    """
    parsed_records: list[dict] = []
    parser_totals: Counter = Counter()
    quality_totals: Counter = Counter()
    sources: list[dict] = []
    for rec in raw_records:
        if len(sources) < keep_sources:
            sources.append(rec)
        parsed, parser_stats = parse_record(
            rec, target_body_chars, min_pairs, min_body, min_infobox
        )
        parser_totals.update(parser_stats)
        quality_totals.update(parsed["quality_flags"])
        parsed_records.append(parsed)
    return parsed_records, parser_totals, quality_totals, sources


def strict_gate_available(raw_records: list[dict]) -> bool:
    """Report whether the cache carries the short answers the strict gate needs."""
    return any("short_answers" in r for r in raw_records[:STRICT_GATE_SAMPLE])


def open_source(
    source: str,
    split: str,
    in_path: Path,
    limit: int,
    dataset: str,
    revision: str | None,
) -> tuple[Iterable[dict] | None, bool]:
    """Return the records to parse, and whether the strict gate applies.

    Args:
        source: Either `stream` or `cache`.
        split: The Natural Questions split name.
        in_path: The cache file, used by the cache source.
        limit: The cap on examples scanned, or on cache records read.
        dataset: The Hugging Face dataset to stream.
        revision: The dataset revision to pin.

    Returns:
        The records, or None when the cache file is missing.
    """
    if source == "cache":
        if not in_path.exists():
            print(f"Skipping {split}: {in_path} does not exist")
            return None, False
        print(f"Loading {in_path} ({in_path.stat().st_size / 1e6:.1f} MB)...")
        records = take(json.loads(in_path.read_text())["records"], limit)
        print(f"  {len(records):,} raw matched records")
        return records, strict_gate_available(records)
    return iter_matched(split, limit, dataset, revision, ScanStats()), True


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


def report_gate(
    parser_totals: Counter,
    quality_totals: Counter,
    n_records: int,
    n_passed: int,
    strict_gate_active: bool,
) -> None:
    """Print the parser counts and the quality-gate outcome of one split.

    Args:
        parser_totals: The dropped-row counts.
        quality_totals: The quality-flag counts.
        n_records: The records parsed.
        n_passed: The records that raised no flag.
        strict_gate_active: Whether the answer check ran.
    """
    print("\nParser stats (totals across all records):")
    for name, count in sorted(parser_totals.items()):
        print(f"  {name}: {count:,}")
    gate = "strict + loose" if strict_gate_active else "loose only"
    print(f"\nQuality gate ({gate}):")
    share = 100 * n_passed / n_records if n_records else 0.0
    print(f"  passed: {n_passed:,} ({share:.1f}%)")
    for flag, count in sorted(quality_totals.items()):
        print(f"  flagged {flag}: {count:,}")


def main() -> None:
    """Parse each requested split and write its parsed cache."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        choices=["train", "validation"],
    )
    ap.add_argument("--target-body-chars", type=int, default=DEFAULT.body_chars)
    ap.add_argument("--min-pairs", type=int, default=DEFAULT.min_pairs)
    ap.add_argument("--min-body", type=int, default=DEFAULT.min_body_chars)
    ap.add_argument("--min-infobox", type=int, default=DEFAULT.min_infobox_chars)
    ap.add_argument(
        "--source",
        choices=["stream", "cache"],
        default="stream",
        help="Read Natural Questions directly, or the raw cache from fetch.py",
    )
    ap.add_argument("--dataset", default=DEFAULT.dataset)
    ap.add_argument(
        "--revision",
        default=DEFAULT.dataset_revision,
        help="Dataset revision to pin (commit SHA, tag, or branch)",
    )
    ap.add_argument(
        "--dry-run",
        type=int,
        default=0,
        help="If >0, write N parsed samples to --dry-run-out and exit",
    )
    ap.add_argument("--dry-run-out", type=Path, default=None)
    add_common_args(
        ap,
        "Cap on examples scanned when streaming, or on records read from the "
        "cache (0 = no cap)",
    )
    args = ap.parse_args()

    dry_run_out = args.dry_run_out or args.data_root / "parse_dry_run.txt"

    for split_idx, split in enumerate(args.splits):
        stage = f"parse_{split}"
        in_path = args.data_root / f"matched_{split}.json"
        out_path = args.data_root / f"parsed_{split}.json"

        print(f"\n=== {split} ===")
        if not args.dry_run and skip_existing(stage, [out_path], args.force):
            continue
        records, strict_gate_active = open_source(
            args.source, split, in_path, args.limit, args.dataset, args.revision
        )
        if records is None:
            continue

        parsed_records, parser_totals, quality_totals, sources = parse_records(
            records,
            args.target_body_chars,
            args.min_pairs,
            args.min_body,
            args.min_infobox,
            keep_sources=args.dry_run if split_idx == 0 else 0,
        )
        n_passed = sum(1 for r in parsed_records if not r["quality_flags"])
        report_gate(
            parser_totals,
            quality_totals,
            len(parsed_records),
            n_passed,
            strict_gate_active,
        )

        if args.dry_run > 0 and split_idx == 0:
            n = min(args.dry_run, len(parsed_records))
            write_dry_run(dry_run_out, parsed_records, sources, n)
            print(f"\nDry-run: wrote {n} parsed samples -> {dry_run_out}")
            print("(No parsed_*.json written. Re-run without --dry-run.)")
            return

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "split": split,
                    "source": args.source,
                    "source_cache": in_path.name if args.source == "cache" else None,
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
        report_written(out_path, len(parsed_records))
        record_stage(
            args.data_root,
            stage,
            {out_path: len(parsed_records)},
            {
                "body_chars": args.target_body_chars,
                "min_pairs": args.min_pairs,
                "min_body_chars": args.min_body,
                "min_infobox_chars": args.min_infobox,
            },
        )


if __name__ == "__main__":
    main()
