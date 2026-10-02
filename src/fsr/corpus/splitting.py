"""Article-level partition of the corpus into train, dev, and test."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from fsr.corpus.config import DEFAULT
from fsr.corpus.layout import split_dir


@dataclass
class Splits:
    """The three partitions and their article and record counts.

    Attributes:
        train: The training records.
        dev: The development records.
        test: The held-out test records.
        counts: The article and record count of each partition.
    """

    train: list[dict]
    dev: list[dict]
    test: list[dict]
    counts: dict[str, int]


SPLIT_CHOICES = ("train", "dev", "test", "nq_val", "all")
PARTITION = ("train", "dev", "test")


def load_records(data_root: Path, split: str, verbose: bool = True) -> list[dict]:
    """Return the records of one split.

    Args:
        data_root: The corpus directory.
        split: A split name, or `all` for the train, dev and test partition
            together. That is the whole quality-passed training corpus.
        verbose: Whether to name each file as it is read.

    Returns:
        The records, in split order.
    """
    names = PARTITION if split == "all" else (split,)
    records: list[dict] = []
    for name in names:
        path = split_dir(data_root) / f"{name}.json"
        if verbose:
            print(f"Loading {path}...")
        records += json.loads(path.read_text())["records"]
    return records


def strict_gated(records: list[dict]) -> list[dict]:
    """Return the records that raised no quality flag."""
    return [r for r in records if not r["quality_flags"]]


def split_by_article(
    records: list[dict],
    seed: int = DEFAULT.split_seed,
    frac_train: float = DEFAULT.frac_train,
    frac_dev: float = DEFAULT.frac_dev,
) -> Splits:
    """Partition records by article title, keeping every title in one split.

    Args:
        records: The records to partition, each carrying a title.
        seed: The seed for the title shuffle.
        frac_train: The share of articles for the train split.
        frac_dev: The share of articles for the dev split.

    Returns:
        The three partitions and their counts.
    """
    by_title: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_title[r["title"]].append(r)

    titles = sorted(by_title)
    random.Random(seed).shuffle(titles)

    n = len(titles)
    n_train = int(n * frac_train)
    n_dev = int(n * frac_dev)
    groups = {
        "train": titles[:n_train],
        "dev": titles[n_train : n_train + n_dev],
        "test": titles[n_train + n_dev :],
    }

    parts = {k: [r for t in v for r in by_title[t]] for k, v in groups.items()}
    counts = {"n_articles_total": n}
    for name in ("train", "dev", "test"):
        counts[f"n_articles_{name}"] = len(groups[name])
        counts[f"n_records_{name}"] = len(parts[name])
    return Splits(parts["train"], parts["dev"], parts["test"], counts)
