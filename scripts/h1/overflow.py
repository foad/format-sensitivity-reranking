"""Find the H1 passages that exceed a model's context after budgeting.

Rebuilds the score-axis records the way `cross_query.py` does and reads the
answer-axis candidate cache, then tokenizes every rendering with every model's
tokenizer. Needs no GPU and no model weights.

Writes `{split}_overflow.json` to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.cli import add_data_root_arg
from fsr.corpus.files import write_atomic
from fsr.corpus.splitting import load_records
from fsr.h1_results import overflow_path, result_path
from fsr.models.loading import load_tokenizer
from fsr.models.registry import BASE_MODEL_IDS, BASE_MODELS
from fsr.overflow import query_items, record_items, scan
from fsr.passages import MAX_TOKENS, prepare_records_with_body

SPLITS = ("test", "nq_val")


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--split", choices=SPLITS, default="test")
    return ap


def check_order(found: list[str], path: Path, key: str) -> None:
    """Stop if the rebuilt items differ from those a result file measured.

    Args:
        found: The ids rebuilt here, in order.
        path: A result file of the same split.
        key: The field of that file holding its ids.

    Raises:
        SystemExit: If the ids or their order differ.
    """
    recorded = json.loads(path.read_text())[key]
    if found != recorded:
        raise SystemExit(
            f"{path.name} measured {len(recorded)} items in a different order "
            f"from the {len(found)} rebuilt here. Rebuild from the same corpus."
        )


def main() -> None:
    """Scan both axes of one split and write the overflow record."""
    args = build_parser().parse_args()
    probe = BASE_MODELS[0].slug

    print(f"Loading tokenizers for {len(BASE_MODEL_IDS)} models...")
    tokenizers = {m: load_tokenizer(m, trust_remote_code=True) for m in BASE_MODEL_IDS}

    records, _ = prepare_records_with_body(
        load_records(args.data_root, args.split), tokenizers
    )
    record_ids = [r["id"] for r in records]
    check_order(
        record_ids,
        result_path(args.data_root, args.split, "cross", "with_body", probe),
        "record_ids",
    )
    print(f"Scanning {len(records):,} score-axis records...")
    cross = scan(record_items(records), tokenizers)

    cache = args.data_root / "h1" / f"{args.split}_candidates.json"
    prepared = json.loads(cache.read_text())["prepared"]
    query_ids = [q["id"] for q in prepared]
    check_order(
        query_ids,
        result_path(args.data_root, args.split, "within", "with_body", probe),
        "query_ids",
    )
    print(f"Scanning {len(prepared):,} answer-axis queries...")
    within = scan(query_items(prepared), tokenizers)

    for name in BASE_MODEL_IDS:
        c, w = cross[name], within[name]
        print(
            f"  {name}: {len(c['ids'])} records, {len(w['ids'])} queries "
            f"({w['renderings_over']}/{w['renderings']} candidate renderings)"
        )

    out = overflow_path(args.data_root, args.split)
    write_atomic(
        out,
        json.dumps(
            {
                "split": args.split,
                "max_tokens": MAX_TOKENS,
                "budget_models": list(BASE_MODEL_IDS),
                "cross": {"record_ids": record_ids, "by_model": cross},
                "within": {"query_ids": query_ids, "by_model": within},
            },
            indent=2,
        ),
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
