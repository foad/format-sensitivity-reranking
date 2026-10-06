"""Reading the cached representations a frozen model gave its head."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from fsr.h2_layout import feature_meta_path, feature_path, feature_score_path

GOLD_COLUMN = 0


@dataclass(frozen=True)
class FeatureStore:
    """The representations and scores of one model on one split.

    Attributes:
        features: The representations.
        scores: The score the frozen model gave each representation
        record_ids: The identifier of each record, in row order.
        formats: The format of each column of the format axis.
        model: The registry slug the representations come from.
        split: The split the records come from.
    """

    features: np.ndarray
    scores: np.ndarray
    record_ids: list[str]
    formats: list[str]
    model: str
    split: str

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self.record_ids)

    @property
    def dim(self) -> int:
        """Return the width of the representation."""
        return int(self.features.shape[-1])

    @property
    def n_negatives(self) -> int:
        """Return the negatives cached for each record."""
        return int(self.features.shape[1]) - 1

    def format_columns(self, names: Sequence[str]) -> list[int]:
        """Return the column of each named format.

        Args:
            names: The format names to locate.

        Returns:
            One column index per name, in the order given.

        Raises:
            ValueError: If a name is not one the cache covers.
        """
        columns = []
        for name in names:
            if name not in self.formats:
                raise ValueError(
                    f"the cache holds {self.formats}, which does not include {name!r}"
                )
            columns.append(self.formats.index(name))
        return columns

    def batch(
        self,
        rows: Sequence[int],
        format_names: Sequence[str],
        n_negatives: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the representations of one batch, ready for a head.

        Args:
            rows: The record rows to take.
            format_names: The formats to take, in the order wanted.
            n_negatives: The negatives to take, counting from the first mined.

        Returns:
            The gold representations of shape (B, F, D) and the negative
            representations of shape (B, F, K, D).

        Raises:
            ValueError: If more negatives are asked for than the cache holds.
        """
        if n_negatives > self.n_negatives:
            raise ValueError(
                f"the cache holds {self.n_negatives} negatives, "
                f"{n_negatives} were asked for"
            )
        columns = self.format_columns(format_names)
        taken = np.asarray(self.features[list(rows)])[:, :, columns, :]
        gold = torch.from_numpy(np.ascontiguousarray(taken[:, GOLD_COLUMN]))
        negatives = taken[:, GOLD_COLUMN + 1 : GOLD_COLUMN + 1 + n_negatives]
        return gold, torch.from_numpy(
            np.ascontiguousarray(negatives.transpose(0, 2, 1, 3))
        )

    def gold_scores(self, format_names: Sequence[str]) -> np.ndarray:
        """Return the frozen model's own gold score, per record and format.

        Args:
            format_names: The formats to take, in the order wanted.

        Returns:
            An array of shape (records, formats).
        """
        return self.scores[:, GOLD_COLUMN, self.format_columns(format_names)]


def load_store(data_root: Path, split: str, model: str) -> FeatureStore:
    """Read one model's cached representations from the corpus directory.

    Args:
        data_root: The corpus directory.
        split: The split the representations cover.
        model: The registry slug.

    Returns:
        The store, with the representations memory mapped.

    Raises:
        SystemExit: If the cache is absent.
    """
    path = feature_path(data_root, split, model)
    meta_path = feature_meta_path(data_root, split, model)
    if not path.exists() or not meta_path.exists():
        raise SystemExit(
            f"no cached representations at {path}; run capture_features.py first."
        )
    meta = json.loads(meta_path.read_text())
    return FeatureStore(
        features=np.load(path, mmap_mode="r"),
        scores=np.load(feature_score_path(data_root, split, model)),
        record_ids=list(meta["record_ids"]),
        formats=list(meta["formats"]),
        model=meta["model"],
        split=meta["split"],
    )
