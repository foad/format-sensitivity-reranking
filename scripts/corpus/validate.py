"""Check that a built corpus is internally consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.cli import add_data_root_arg
from fsr.corpus.layout import META_NAME, NEGATIVES_NAME, SPLIT_NAMES, split_dir
from fsr.corpus.manifest import MANIFEST_NAME, Manifest
from fsr.corpus.splitting import strict_gated
from fsr.corpus.validate import (
    check_ids_unique,
    check_manifest_digests,
    check_meta,
    check_negatives,
    check_quality_gate,
    check_titles_disjoint,
)

PARSED_NAME = "parsed_train.json"


def load_json(path: Path, problems: list[str]) -> dict | None:
    """Read a JSON file, or record that it is missing.

    Args:
        path: The file to read.
        problems: The list to add to when the file is absent.

    Returns:
        The contents, or None when the file is absent.
    """
    if not path.exists():
        problems.append(f"{path} is missing")
        return None
    return json.loads(path.read_text())


def collect_problems(data_root: Path) -> list[str]:
    """Run every check over a built corpus.

    Args:
        data_root: The directory holding the corpus files and the manifest.

    Returns:
        The problems found, in the order the checks ran.
    """
    problems: list[str] = []
    split_path = split_dir(data_root)

    parsed = load_json(data_root / PARSED_NAME, problems)
    splits = {}
    for name in SPLIT_NAMES:
        data = load_json(split_path / f"{name}.json", problems)
        if data is not None:
            splits[name] = data["records"]
    meta = load_json(split_path / META_NAME, problems)
    negatives = load_json(split_path / NEGATIVES_NAME, problems)
    manifest_path = data_root / MANIFEST_NAME
    if not manifest_path.exists():
        problems.append(f"{manifest_path} is missing")
    if problems:
        return problems

    for name, records in splits.items():
        problems += check_quality_gate(name, records)
    problems += check_titles_disjoint(splits)
    problems += check_ids_unique(splits)
    problems += check_meta(meta, splits)
    problems += check_negatives(
        negatives,
        [r for name in SPLIT_NAMES for r in splits[name]],
        strict_gated(parsed["records"]),
    )
    problems += check_manifest_digests(Manifest.load(manifest_path), data_root)
    return problems


def main() -> None:
    """Check the corpus and report what is wrong with it.

    Raises:
        SystemExit: If any check found a problem.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    args = ap.parse_args()

    problems = collect_problems(args.data_root)
    for problem in problems:
        print(f"  {problem}")
    if problems:
        raise SystemExit(f"{len(problems)} problems found in {args.data_root}")
    print(f"{args.data_root}: every check passed")


if __name__ == "__main__":
    main()
