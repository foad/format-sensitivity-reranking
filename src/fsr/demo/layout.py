"""Where the pipeline demonstration keeps its artefacts.

The demonstration's data is synthetic, so it lives apart from the corpus. The
adapters it scores with stay in the corpus directory, under `fsr.h2_layout`.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_DEMO_ROOT = Path("data") / "demo"
SPLIT = "demo"


def scenarios_path(demo_root: Path) -> Path:
    """Return the file holding every generated pool."""
    return demo_root / "scenarios.json"


def tensor_path(demo_root: Path, model: str, arm_name: str) -> Path:
    """Return the file holding the scores of one condition.

    Args:
        demo_root: The demonstration directory.
        model: The registry slug.
        arm_name: The trained condition, or BASE_ARM for the untrained model.

    Returns:
        The path.
    """
    return demo_root / f"scores_{model}_{arm_name}.npy"


def tensor_meta_path(demo_root: Path, model: str, arm_name: str) -> Path:
    """Return the file describing the scores of one condition."""
    return tensor_path(demo_root, model, arm_name).with_suffix(".json")


def selection_path(demo_root: Path, model: str) -> Path:
    """Return the file holding the selection funnel and the matching pools."""
    return demo_root / f"selection_{model}.json"


def topk_path(demo_root: Path, model: str, scenario_id: str) -> Path:
    """Return the file holding what each arm passes on for one pool."""
    return demo_root / f"topk_{model}_{scenario_id}.md"


def loop_root(demo_root: Path) -> Path:
    """Return the directory the iteration loop exchanges its jobs through."""
    return demo_root / "loop"


def inbox_dir(demo_root: Path) -> Path:
    """Return the directory the jobs are pushed to. The worker only reads it."""
    return loop_root(demo_root) / "inbox"


def outbox_dir(demo_root: Path) -> Path:
    """Return the directory the worker writes its claims, results and cache to."""
    return loop_root(demo_root) / "outbox"
