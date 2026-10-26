"""Score one model and arm over every candidate, under every format.

Writes `h2/standardisation/{split}_scores_{model}_{arm}.npy` and the matching
description. The study reads the candidate lists the answer axis built, so its
queries match the main H2 result.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, adapter_dir
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import ProgressCounter, heading, report_elapsed, report_saved
from fsr.standardisation.layout import tensor_path
from fsr.standardisation.scoring import (
    load_candidates,
    query_ids,
    score_every_format,
)
from fsr.standardisation.store import ScoreTensor, save_tensor

DEFAULT_SPLIT = "test"


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--arm",
        default=BASE_ARM,
        help="The trained arm to score with. The default is the untrained model",
    )
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--tanh-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the classifier head the registry records",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help="Load with eager attention",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap records for a check")
    ap.add_argument("--progress-file", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild a present capture")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Capture one condition of the standardisation study."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    out_path = tensor_path(args.data_root, args.split, entry.slug, args.arm)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {args.arm}: {out_path} is present.")
        return

    adapter = None
    if args.arm != BASE_ARM:
        adapter = adapter_dir(args.data_root, entry.slug, args.arm)
        if not adapter.exists():
            raise SystemExit(f"no adapter at {adapter}; train the arm first.")

    prepared = load_candidates(args.data_root, args.split, args.limit)
    batch_size = args.batch_size or entry.eval_batch
    tanh_head = entry.tanh_head if args.tanh_head is None else args.tanh_head
    device = "cuda" if torch.cuda.is_available() else "cpu"

    heading(f"STANDARDISATION  {entry.label}  arm={args.arm}", width=70)
    print(f"  identifier: {entry.model_id}")
    print(f"  adapter:    {adapter or 'none, untrained baseline'}")
    print(f"  queries:    {len(prepared):,}")
    print(f"  candidates: {len(prepared[0]['candidates'])} each")
    print(f"  formats:    {' '.join(FORMAT_NAMES)}")
    print(f"  batch:      {batch_size}")
    print(f"  device:     {device}")

    tokenizer = load_tokenizer(entry.model_id)
    model = load_model(
        entry.model_id,
        device,
        lora_adapter_path=str(adapter) if adapter else None,
        eager_attn=args.eager_attn,
        tanh_head=tanh_head,
    )

    counter = ProgressCounter(len(FORMAT_NAMES), args.progress_file)
    started = time.time()
    scores = score_every_format(
        model,
        tokenizer,
        prepared,
        batch_size,
        device,
        on_format=lambda name: counter.step(f"scored/{name}"),
    )

    tensor = ScoreTensor(
        scores=scores,
        query_ids=query_ids(prepared),
        formats=list(FORMAT_NAMES),
        model=entry.slug,
        arm=args.arm,
        split=args.split,
    )
    written = save_tensor(
        args.data_root,
        tensor,
        extra={
            "model_id": entry.model_id,
            "adapter": str(adapter) if adapter else None,
            "tanh_head": tanh_head,
            "batch_size": batch_size,
            "device": device,
            "limit": args.limit,
        },
    )
    report_elapsed("standardisation capture", started)
    report_saved(written, f"{entry.slug} {args.arm} scores")


if __name__ == "__main__":
    main()
