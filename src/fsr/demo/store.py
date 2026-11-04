"""Reading and writing the demonstration's score tensors.

The tensors share `fsr.standardisation.store.ScoreTensor`, so the
standardisation study's conditions and metrics apply to them unchanged. Only
the location differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from fsr.demo.layout import tensor_meta_path, tensor_path
from fsr.standardisation.store import GOLD, ScoreTensor


def save_tensor(
    demo_root: Path, tensor: ScoreTensor, extra: dict[str, Any] | None = None
) -> Path:
    """Write one condition's scores and the description beside them.

    Args:
        demo_root: The demonstration directory.
        tensor: The scores to write.
        extra: More fields for the description.

    Returns:
        The path of the scores.
    """
    path = tensor_path(demo_root, tensor.model, tensor.arm)
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
    tensor_meta_path(demo_root, tensor.model, tensor.arm).write_text(
        json.dumps(meta, indent=2)
    )
    return path


def load_tensor(demo_root: Path, model: str, arm_name: str) -> ScoreTensor:
    """Read one condition's scores.

    Args:
        demo_root: The demonstration directory.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The scores.

    Raises:
        SystemExit: If the condition was not scored.
    """
    path = tensor_path(demo_root, model, arm_name)
    meta_path = tensor_meta_path(demo_root, model, arm_name)
    if not path.exists() or not meta_path.exists():
        raise SystemExit(
            f"no scores for {model} {arm_name} at {path}; score the condition first."
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
