"""The command-line behaviour shared by the pipeline scripts."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from itertools import islice
from pathlib import Path

from fsr.models.registry import MODELS, Model, by_id, by_slug

DEFAULT_DATA_ROOT = Path("data") / "nq"


def add_data_root_arg(ap: argparse.ArgumentParser) -> None:
    """Add the argument that names the corpus directory."""
    ap.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Directory holding the corpus files and the manifest",
    )


def add_common_args(ap: argparse.ArgumentParser, limit_help: str) -> None:
    """Add the arguments that every stage accepts.

    Args:
        ap: The parser to add to.
        limit_help: The help text for --limit.
    """
    add_data_root_arg(ap)
    ap.add_argument("--limit", type=int, default=0, help=limit_help)
    ap.add_argument(
        "--force", action="store_true", help="Rebuild outputs that already exist"
    )


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
    if len(outputs) == 1:
        present = f"{outputs[0]} is"
    else:
        present = f"{len(outputs)} outputs in {outputs[0].parent} are"
    print(f"Skipping {name}: {present} already present. Use --force to rebuild.")
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


def resolve_model(name: str) -> Model:
    """Return the registry entry for a slug or an identifier.

    Args:
        name: A registry slug or a Hugging Face identifier.

    Returns:
        The model.

    Raises:
        SystemExit: If the name is in neither form.
    """
    found = by_id(name) if "/" in name else None
    if found is not None:
        return found
    try:
        return by_slug(name)
    except ValueError:
        known = ", ".join(m.slug for m in MODELS)
        raise SystemExit(f"unknown model {name!r}. Expected one of: {known}") from None
