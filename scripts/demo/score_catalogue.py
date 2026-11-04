"""Score every wording against every passage version in the grid's catalogue.

Writes `data/demo/grid/scores_<arm>.npy`, shaped (wordings, texts, formats), for
each arm. The grid's rankings are assembled from these offline.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.demo.grid import grid_scores_path, load_catalogue
from fsr.demo.layout import DEFAULT_DEMO_ROOT
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.h2_layout import BASE_ARM, adapter_dir
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import ProgressCounter, report_saved
from fsr.scoring import MAX_TOKENS, score_batch

ARMS = ("base", "5fmt_lam0", "5fmt_lam1", "json_lam0", "json_lam1")


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="bge_base")
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--progress-file", type=Path, default=None)
    ap.add_argument("--force", action="store_true")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Score the catalogue under each arm."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    catalogue = load_catalogue(args.demo_root)
    by_id = {p["id"]: p for p in catalogue.passages}
    texts = [by_id[pid]["versions"][v] for pid, v in catalogue.texts()]
    wordings = [w["text"] for w in catalogue.wordings]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch = args.batch_size or entry.eval_batch
    tokenizer = load_tokenizer(entry.model_id)

    pairs = [
        (w, FORMATS[name](t["pairs"], t["body"]))
        for w in wordings
        for t in texts
        for name in FORMAT_NAMES
    ]
    long = sum(len(tokenizer(q, p)["input_ids"]) > MAX_TOKENS for q, p in pairs)
    if long:
        raise SystemExit(f"{long} pairs would be truncated")
    arms = args.arms.split(",")
    print(
        f"{len(wordings)} wordings x {len(texts)} texts x {len(FORMAT_NAMES)} formats"
    )
    print(f"= {len(pairs):,} pairs per arm, {len(arms)} arms, on {device}", flush=True)

    counter = ProgressCounter(len(arms), args.progress_file)
    for arm in arms:
        out = grid_scores_path(args.demo_root, arm)
        if out.exists() and not args.force:
            print(f"skipping {arm}: {out} is present")
            counter.step(f"{arm} present")
            continue
        adapter = (
            None if arm == BASE_ARM else adapter_dir(args.data_root, entry.slug, arm)
        )
        model = load_model(
            entry.model_id,
            device,
            lora_adapter_path=str(adapter) if adapter else None,
            tanh_head=entry.tanh_head,
        )
        started = time.time()
        flat = score_batch(model, tokenizer, pairs, batch, device)
        scores = np.asarray(flat, dtype=np.float32).reshape(
            len(wordings), len(texts), len(FORMAT_NAMES)
        )
        np.save(out, scores)
        print(f"{arm}: {time.time() - started:.0f}s", flush=True)
        report_saved(out, f"{arm} scores")
        counter.step(arm)
        del model


if __name__ == "__main__":
    main()
