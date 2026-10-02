"""The names and the directory layout of the corpus files."""

from __future__ import annotations

from pathlib import Path

SPLIT_SUBDIR = "splits"
META_NAME = "meta.json"
NEGATIVES_NAME = "bm25_negatives.json"
SPLIT_NAMES = ("train", "dev", "test", "nq_val")


def split_dir(data_root: Path) -> Path:
    """Return the directory that holds the split files."""
    return data_root / SPLIT_SUBDIR
