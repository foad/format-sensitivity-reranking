"""The score tensor one condition of the standardisation study produces."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fsr.standardisation.layout import tensor_meta_path, tensor_path

GOLD = 0


@dataclass(frozen=True)
class ScoreTensor:
    """Every candidate of every query, scored under every format.

    Attributes:
        scores: The scores, shaped (queries, candidates, formats). The gold
            passage is candidate GOLD.
        query_ids: The query of each row.
        formats: The format of each column of the last axis, in order.
        model: The registry slug.
        arm: The trained condition, or BASE_ARM for the untrained model.
        split: The split the queries come from.
    """

    scores: np.ndarray
    query_ids: list[str]
    formats: list[str]
    model: str
    arm: str
    split: str

    def __post_init__(self) -> None:
        """Check the axes against the labels.

        Raises:
            ValueError: If the scores are not one value per query, candidate
                and format.
        """
        if self.scores.ndim != 3:
            raise ValueError(
                f"the scores have {self.scores.ndim} axes, expected 3: "
                "queries, candidates and formats"
            )
        queries, candidates, formats = self.scores.shape
        if len(self.query_ids) != queries:
            raise ValueError(
                f"{len(self.query_ids)} query ids for {queries} rows of scores"
            )
        if len(self.formats) != formats:
            raise ValueError(
                f"{len(self.formats)} format names for {formats} score columns"
            )
        if candidates < 2:
            raise ValueError(
                f"each query carries {candidates} candidates, expected the gold "
                "passage and at least one negative"
            )

    @property
    def n_queries(self) -> int:
        """Return the number of queries."""
        return self.scores.shape[0]

    @property
    def n_candidates(self) -> int:
        """Return the number of candidates of each query."""
        return self.scores.shape[1]

    @property
    def n_formats(self) -> int:
        """Return the number of formats."""
        return self.scores.shape[2]

    @property
    def gold(self) -> np.ndarray:
        """Return the gold score of each query, by format."""
        return self.scores[:, GOLD]

    @property
    def negatives(self) -> np.ndarray:
        """Return the negative scores of each query, by format."""
        return self.scores[:, GOLD + 1 :]


def save_tensor(
    data_root: Path, tensor: ScoreTensor, extra: dict[str, Any] | None = None
) -> Path:
    """Write one condition's scores and the description beside them.

    Args:
        data_root: The corpus directory.
        tensor: The scores to write.
        extra: More fields for the description.

    Returns:
        The path of the scores.
    """
    path = tensor_path(data_root, tensor.split, tensor.model, tensor.arm)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.scores)
    meta = {
        "model": tensor.model,
        "arm": tensor.arm,
        "split": tensor.split,
        "formats": tensor.formats,
        "n_queries": tensor.n_queries,
        "n_candidates": tensor.n_candidates,
        "gold_index": GOLD,
        "query_ids": tensor.query_ids,
        **(extra or {}),
    }
    meta_path = tensor_meta_path(data_root, tensor.split, tensor.model, tensor.arm)
    meta_path.write_text(json.dumps(meta, indent=2))
    return path


def load_tensor(data_root: Path, split: str, model: str, arm_name: str) -> ScoreTensor:
    """Read one condition's scores.

    Args:
        data_root: The corpus directory.
        split: The split the queries come from.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The scores.

    Raises:
        SystemExit: If the condition was not captured.
    """
    path = tensor_path(data_root, split, model, arm_name)
    meta_path = tensor_meta_path(data_root, split, model, arm_name)
    if not path.exists() or not meta_path.exists():
        raise SystemExit(
            f"no scores for {model} {arm_name} on {split} at {path}; "
            "capture the condition first."
        )
    meta = json.loads(meta_path.read_text())
    return ScoreTensor(
        scores=np.load(path),
        query_ids=meta["query_ids"],
        formats=meta["formats"],
        model=meta["model"],
        arm=meta["arm"],
        split=meta["split"],
    )
