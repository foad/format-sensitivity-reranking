"""Partition the parsed corpus into train, dev, and test by article.

The Natural Questions validation split is kept as a second, independent test set.

Writes `train.json`, `dev.json`, `test.json`, `nq_val.json`, and `meta.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.corpus.config import DEFAULT
from fsr.corpus.splitting import split_by_article, strict_gated

DEFAULT_IN_DIR = Path("data") / "nq"
DEFAULT_OUT_DIR = Path("data") / "nq" / "h2_splits"


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
    path.write_text(
        json.dumps(
            {
                "split": name,
                "seed": seed,
                "n_records": len(records),
                "records": records,
            },
            ensure_ascii=False,
        )
    )
    return path


def main() -> None:
    """Partition the parsed corpus and write the split files."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=DEFAULT.split_seed)
    ap.add_argument("--frac-train", type=float, default=DEFAULT.frac_train)
    ap.add_argument("--frac-dev", type=float, default=DEFAULT.frac_dev)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_passed = load_passed(args.in_dir / "parsed_train.json")
    splits = split_by_article(train_passed, args.seed, args.frac_train, args.frac_dev)
    nq_val = load_passed(args.in_dir / "parsed_validation.json")

    written = [
        ("train", splits.train),
        ("dev", splits.dev),
        ("test", splits.test),
        ("nq_val", nq_val),
    ]
    for name, records in written:
        path = write_split(args.out_dir, name, records, args.seed)
        print(f"{name:>8}: {len(records):>5} records -> {path}")

    meta = {
        "seed": args.seed,
        "frac_train": args.frac_train,
        "frac_dev": args.frac_dev,
        "nq_val_n_records": len(nq_val),
        **splits.counts,
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nmeta: {json.dumps(meta, indent=2)}")


if __name__ == "__main__":
    main()
