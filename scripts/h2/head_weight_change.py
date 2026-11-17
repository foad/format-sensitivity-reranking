"""Measure how far training moved the head layers away from their pretrained values.

Writes `h2/head_weight_change.json`.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open

import fsr.h2_results as h2r
from fsr.cli import add_data_root_arg, resolve_model
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import format_lambda, h2_dir
from fsr.reporting import heading, report_saved, table

PREFIX = "base_model.model."
WEIGHTS_FILE = "model.safetensors"
ARM = re.compile(r"_lam(?P<weight>[\d.]+)(?:_s(?P<seed>\d+))?$")
MODELS = ("minilm_l6", "minilm_l6_pooler", "minilm_l12", "minilm_l12_pooler")
COLUMNS = [("model", 18), ("tensor", 26), ("n", 4), ("control", 8)]
COLUMNS += [("chosen", 8), ("min", 8), ("max", 8)]
MODELS += ("mxbai_v1", "mxbai_v1_pooler", "bge_base", "bge_v2_m3", "jina_v2")


def relative_change(trained: np.ndarray, pretrained: np.ndarray) -> float:
    """Return the norm of the change as a share of the pretrained norm."""
    return float(np.linalg.norm(trained - pretrained) / np.linalg.norm(pretrained))


def model_change(data_root: Path, slug: str) -> dict[str, Any]:
    """Return the relative change of each head tensor over a model's adapters.

    Args:
        data_root: The corpus directory.
        slug: The registry slug.

    Returns:
        The relative change of each head tensor and the number of adapters read.
    """
    model = resolve_model(slug)
    pretrained = safe_open(
        hf_hub_download(model.model_id, WEIGHTS_FILE, local_files_only=True), "pt"
    )
    stem = f"{slug}_"
    chosen = h2r.winner_weight(h2r.load_selection(data_root, slug))
    groups = {
        format_lambda(h2r.CONTROL_WEIGHT): "control",
        format_lambda(chosen): "chosen",
    }
    changes: dict[str, dict[str, list[float]]] = {}
    count = 0
    for run in sorted((h2_dir(data_root) / "train").glob(f"{stem}*")):
        match = ARM.search(run.name)
        path = run / "adapter" / "adapter_model.safetensors"
        fold = run.name[len(stem) :].split("_lam")[0]
        if not (path.exists() and match and fold in FORMAT_NAMES):
            continue
        if match["weight"] not in groups:
            continue
        count += 1
        trained = safe_open(path, "pt")
        names = trained.keys()
        for key in names:
            name = key.removeprefix(PREFIX)
            if "lora_" in name or not name.endswith("weight"):
                continue
            before = pretrained.get_tensor(name).float().numpy()
            value = relative_change(trained.get_tensor(key).float().numpy(), before)
            per_weight = changes.setdefault(name, {})
            per_weight.setdefault(groups[match["weight"]], []).append(value)
    return {
        "adapters": count,
        "tensors": {
            name: {
                group: {
                    "mean": float(np.mean(v)),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                }
                for group, v in groups.items()
            }
            for name, groups in changes.items()
        },
    }


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Report the head weight change of every model."""
    args = build_parser().parse_args()
    results = {slug: model_change(args.data_root, slug) for slug in args.models}
    heading("Relative change of the trained head weights")
    table(
        COLUMNS,
        [
            (
                slug,
                name,
                str(result["adapters"]),
                f"{stats['control']['mean']:.4f}",
                f"{stats['chosen']['mean']:.4f}",
                f"{min(g['min'] for g in stats.values()):.4f}",
                f"{max(g['max'] for g in stats.values()):.4f}",
            )
            for slug, result in results.items()
            for name, stats in result["tensors"].items()
        ],
    )
    out = h2_dir(args.data_root) / "head_weight_change.json"
    out.write_text(json.dumps(results, indent=2))
    report_saved(out)


if __name__ == "__main__":
    main()
