"""Shared command-line behaviour of the corpus build stages."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from itertools import islice
from pathlib import Path

from fsr.corpus.config import DEFAULT
from fsr.corpus.manifest import Manifest, Output

DEFAULT_DATA_ROOT = Path("data") / "nq"
SPLIT_SUBDIR = "splits"
MANIFEST_NAME = "manifest.json"


def add_common_args(ap: argparse.ArgumentParser, limit_help: str) -> None:
    """Add the arguments that every stage accepts.

    Args:
        ap: The parser to add to.
        limit_help: The help text for --limit, which counts different things
            in a stage that streams and a stage that reads a file.
    """
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory holding the corpus files and the manifest",
    )
    ap.add_argument("--limit", type=int, default=0, help=limit_help)
    ap.add_argument(
        "--force", action="store_true", help="Rebuild outputs that already exist"
    )


def split_dir(data_root: Path) -> Path:
    """Return the directory that holds the split files."""
    return data_root / SPLIT_SUBDIR


def take[T](records: Iterable[T], limit: int) -> list[T]:
    """Return the first records of a sequence or a stream.

    Args:
        records: The records to take from.
        limit: How many to take. 0 takes every record.

    Returns:
        The records taken.
    """
    return list(islice(records, limit)) if limit else list(records)


def skip_existing(name: str, outputs: Sequence[Path], force: bool) -> bool:
    """Report whether a stage may leave its outputs as they are.

    Args:
        name: The stage name, for the message.
        outputs: The files the stage writes.
        force: Whether the caller asked for a rebuild.

    Returns:
        True when the stage may skip its work.
    """
    if force or not outputs or not all(p.exists() for p in outputs):
        return False
    listed = ", ".join(str(p) for p in outputs)
    print(f"Skipping {name}: {listed} already present. Use --force to rebuild.")
    return True


def report_written(path: Path, records: int | None = None) -> None:
    """Print the size of a written file, and the records it holds.

    Args:
        path: The file that was written.
        records: The number of records the file holds.
    """
    detail = f"{path.stat().st_size / 1e6:.1f} MB"
    if records is not None:
        detail = f"{records:,} records, {detail}"
    print(f"Wrote {path}  ({detail})")


def record_stage(
    data_root: Path,
    name: str,
    outputs: Mapping[Path, int | None],
    config: Mapping[str, object],
) -> Path:
    """Add a completed stage to the build manifest and save it.

    Args:
        data_root: Manifest directory, with outputs recorded relative to it.
        name: The stage name. An earlier stage of this name is replaced.
        outputs: The records each written file holds, or None.
        config: The parameter values the stage ran with.

    Returns:
        The manifest file.
    """
    path = data_root / MANIFEST_NAME
    manifest = Manifest.load_or_new(path, DEFAULT.as_dict())
    manifest.config.update(config)
    manifest.record(
        name, [Output.describe(p, data_root, n) for p, n in outputs.items()]
    )
    manifest.save(path)
    return path
