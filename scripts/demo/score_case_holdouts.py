"""Score the case under every leave-one-format-out pair of adapters.

H2 trained each fine-tuned arm once per held-out format. Scoring the case with
all of them shows whether the gap between the two weights depends on which
format the models never saw. Writes `case_scores_<arm>.npy` for each pair not
already present, and adds the arms to `case.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.demo.layout import DEFAULT_DEMO_ROOT
from fsr.demo.results import HOLD_OUTS, case_dir, scores_path
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.h2_layout import adapter_dir
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import report_saved
from fsr.scoring import score_batch


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="bge_base")
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--force", action="store_true", help="Rescore present arms")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Score the case under each hold-out pair."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    meta_path = case_dir(args.demo_root) / "case.json"
    meta = json.loads(meta_path.read_text())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer(entry.model_id)
    pairs = [
        (meta["question"], FORMATS[name](c["pairs"], c["truncated_body"]))
        for c in meta["candidates"]
        for name in FORMAT_NAMES
    ]
    arms = [a for pair in HOLD_OUTS.values() for a in pair]
    for arm in arms:
        out = scores_path(args.demo_root, "case", arm)
        if out.exists() and not args.force:
            print(f"skipping {arm}: present")
            continue
        model = load_model(
            entry.model_id,
            device,
            lora_adapter_path=str(adapter_dir(args.data_root, entry.slug, arm)),
            tanh_head=entry.tanh_head,
        )
        flat = score_batch(
            model, tokenizer, pairs, args.batch_size or entry.eval_batch, device
        )
        np.save(out, np.asarray(flat, dtype=np.float32).reshape(-1, len(FORMAT_NAMES)))
        report_saved(out, f"{arm} case scores")
        del model
    meta["arms"] = list(dict.fromkeys([*meta["arms"], *arms]))
    meta_path.write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
