"""Score every demonstration pool with one model and arm, under every format.

Writes `scores_{model}_{arm}.npy` and the matching description to the
demonstration directory. Adapters load from the corpus directory, where H2
trained them.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.demo.layout import DEFAULT_DEMO_ROOT, SPLIT, scenarios_path, tensor_path
from fsr.demo.scenarios import overlong
from fsr.demo.store import save_tensor
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, adapter_dir
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import ProgressCounter, heading, report_elapsed, report_saved
from fsr.standardisation.scoring import query_ids, score_every_format
from fsr.standardisation.store import ScoreTensor


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--arm",
        default=BASE_ARM,
        help="The trained arm to score with. The default is the untrained model",
    )
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--tanh-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the classifier head the registry records",
    )
    ap.add_argument(
        "--eager-attn", action="store_true", help="Load with eager attention"
    )
    ap.add_argument("--progress-file", type=Path, default=None)
    ap.add_argument("--force", action="store_true", help="Rebuild a present tensor")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Score one condition of the demonstration."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    out_path = tensor_path(args.demo_root, entry.slug, args.arm)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {args.arm}: {out_path} is present.")
        return

    pools_path = scenarios_path(args.demo_root)
    if not pools_path.exists():
        raise SystemExit(
            f"no pools at {pools_path}; run scripts/demo/build_scenarios.py first."
        )
    pools = json.loads(pools_path.read_text())["scenarios"]

    adapter = None
    if args.arm != BASE_ARM:
        adapter = adapter_dir(args.data_root, entry.slug, args.arm)
        if not adapter.exists():
            raise SystemExit(f"no adapter at {adapter}; train the arm first.")

    batch_size = args.batch_size or entry.eval_batch
    tanh_head = entry.tanh_head if args.tanh_head is None else args.tanh_head
    device = "cuda" if torch.cuda.is_available() else "cpu"

    heading(f"DEMO SCORING  {entry.label}  arm={args.arm}", width=70)
    print(f"  identifier: {entry.model_id}")
    print(f"  adapter:    {adapter or 'none, untrained baseline'}")
    print(f"  pools:      {len(pools):,}")
    print(f"  candidates: {len(pools[0]['candidates'])} each")
    print(f"  formats:    {' '.join(FORMAT_NAMES)}")
    print(f"  batch:      {batch_size}")
    print(f"  device:     {device}")

    tokenizer = load_tokenizer(entry.model_id)
    long_pairs = overlong(pools, tokenizer)
    if long_pairs:
        raise SystemExit(
            f"{len(long_pairs)} pairs would be truncated, the first "
            f"{long_pairs[0]}; shorten the pools."
        )
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
        pools,
        batch_size,
        device,
        on_format=lambda name: counter.step(f"scored/{name}"),
    )

    tensor = ScoreTensor(
        scores=scores,
        query_ids=query_ids(pools),
        formats=list(FORMAT_NAMES),
        model=entry.slug,
        arm=args.arm,
        split=SPLIT,
    )
    written = save_tensor(
        args.demo_root,
        tensor,
        extra={
            "model_id": entry.model_id,
            "adapter": str(adapter) if adapter else None,
            "tanh_head": tanh_head,
            "batch_size": batch_size,
            "device": device,
        },
    )
    report_elapsed("demo scoring", started)
    report_saved(written, f"{entry.slug} {args.arm} scores")


if __name__ == "__main__":
    main()
