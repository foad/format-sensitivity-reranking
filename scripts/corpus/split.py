"""Partition the parsed corpus into train, dev, and test by article.

The Natural Questions validation split is kept as a second, independent test set.

Writes `train.json`, `dev.json`, `test.json`, `nq_val.json`, and `meta.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.cli import (
    META_NAME,
    SPLIT_NAMES,
    add_common_args,
    record_stage,
    skip_existing,
    split_dir,
    take,
)
from fsr.corpus.config import DEFAULT
from fsr.corpus.files import write_atomic
from fsr.corpus.splitting import split_by_article, strict_gated


def load_passed(path: Path) -> list[dict]:
    """Return the quality-passed records of a parsed corpus file.

    Args:
        path: The parsed corpus file.

    Returns:
        The records that raised no quality flag.

    Raises:
        ValueError: If the count disagrees with the file's own total.
    """
    data = json.loads(path.read_text())
    passed = strict_gated(data["records"])
    expected = data.get("n_passed_quality")
    if expected is not None and len(passed) != expected:
        raise ValueError(
            f"{path.name}: counted {len(passed)} passed records, "
            f"file reports {expected}"
        )
    return passed


def write_split(out_dir: Path, name: str, records: list[dict], seed: int) -> Path:
    """Write one split file and return its path."""
    path = out_dir / f"{name}.json"
    write_atomic(
        path,
        json.dumps(
            {
                "split": name,
                "seed": seed,
                "n_records": len(records),
                "records": records,
            },
            ensure_ascii=False,
        ),
    )
    return path


def main() -> None:
    """Partition the parsed corpus and write the split files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=DEFAULT.split_seed)
    ap.add_argument("--frac-train", type=float, default=DEFAULT.frac_train)
    ap.add_argument("--frac-dev", type=float, default=DEFAULT.frac_dev)
    add_common_args(ap, "Cap on records read from each parsed file (0 = no cap)")
    args = ap.parse_args()

    out_dir = split_dir(args.data_root)
    outputs = [out_dir / f"{name}.json" for name in SPLIT_NAMES]
    outputs.append(out_dir / META_NAME)
    if skip_existing("split", outputs, args.force):
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    train_passed = take(load_passed(args.data_root / "parsed_train.json"), args.limit)
    splits = split_by_article(train_passed, args.seed, args.frac_train, args.frac_dev)
    nq_val = take(load_passed(args.data_root / "parsed_validation.json"), args.limit)

    written = dict(
        zip(
            SPLIT_NAMES,
            [splits.train, splits.dev, splits.test, nq_val],
            strict=True,
        )
    )
    recorded: dict[Path, int | None] = {}
    for name, records in written.items():
        path = write_split(out_dir, name, records, args.seed)
        recorded[path] = len(records)
        print(f"{name:>8}: {len(records):>5} records -> {path}")

    meta = {
        "seed": args.seed,
        "frac_train": args.frac_train,
        "frac_dev": args.frac_dev,
        "nq_val_n_records": len(nq_val),
        **splits.counts,
    }
    meta_path = out_dir / META_NAME
    write_atomic(meta_path, json.dumps(meta, indent=2))
    recorded[meta_path] = None
    print(f"\nmeta: {json.dumps(meta, indent=2)}")

    record_stage(
        args.data_root,
        "split",
        recorded,
        {
            "split_seed": args.seed,
            "frac_train": args.frac_train,
            "frac_dev": args.frac_dev,
            "frac_test": 1.0 - args.frac_train - args.frac_dev,
        },
    )


if __name__ == "__main__":
    main()
