"""Train one LoRA adapter under the ranking and invariance objective.

Writes `h2/train/{model}_{arm}/` holding the adapter and logs.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import torch
from transformers import get_scheduler

from fsr.cli import add_data_root_arg, resolve_model
from fsr.corpus.splitting import load_records
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import ADAPTER_NAME, adapter_config_path, arm, train_dir
from fsr.models.adapters import build_adapted_model
from fsr.models.loading import load_tokenizer
from fsr.reporting import ProgressCounter, heading, report_elapsed
from fsr.training.checkpoint import CHECKPOINT_NAME, load_checkpoint
from fsr.training.data import (
    DEFAULT_TRAIN_NEGATIVES,
    load_corpus_index,
    load_negatives,
    prepare_dev_records,
    prepare_train_records,
)
from fsr.training.log import LOG_NAME, RUN, TrainLog, rewind
from fsr.training.loop import LoopConfig, train

DEFAULT_LR = 2e-4
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_MAX_STEPS = 500
DEFAULT_WARMUP_STEPS = 30
DEFAULT_EVAL_EVERY = 50
HELD_OUT_NONE = "none"


def training_formats(held_out: str | None) -> list[str]:
    """Return the formats a run trains on.

    Args:
        held_out: The format to withhold, or None to train on every format.

    Returns:
        The format names, in the published order.
    """
    if held_out is None:
        return list(FORMAT_NAMES)
    return [name for name in FORMAT_NAMES if name != held_out]


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--held-out-format",
        default=HELD_OUT_NONE,
        choices=[*FORMAT_NAMES, HELD_OUT_NONE],
        help="Format withheld from training. 'none' trains on every format",
    )
    ap.add_argument("--lambda-inv", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    ap.add_argument("--lora-rank", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=None)
    ap.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Module name patterns. The default is the registry entry",
    )
    ap.add_argument(
        "--rank-tag",
        type=int,
        default=None,
        help="Record the adapter rank in the arm name, for a rank sweep",
    )
    ap.add_argument("--physical-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--neg-k", type=int, default=DEFAULT_TRAIN_NEGATIVES)
    ap.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    ap.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)
    ap.add_argument("--eval-every", type=int, default=DEFAULT_EVAL_EVERY)
    ap.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Steps between resume checkpoints. 0 writes none",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the checkpoint in the run directory, if there is one",
    )
    ap.add_argument(
        "--no-grad-checkpoint",
        dest="grad_checkpoint",
        action="store_false",
        help="Keep activations instead of recomputing them",
    )
    ap.set_defaults(grad_checkpoint=True)
    ap.add_argument(
        "--tanh-head",
        action="store_true",
        default=None,
        help="Replace the classifier head. The default is the registry entry",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help="Request the eager attention kernel in place of the fused one",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="File to keep the live progress of this job in",
    )
    ap.add_argument("--limit-train", type=int, default=None, help="Cap train records")
    ap.add_argument("--limit-dev", type=int, default=None, help="Cap dev records")
    ap.add_argument(
        "--force", action="store_true", help="Train again over a finished run"
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Train one arm and save its adapter."""
    args = build_parser().parse_args()
    model_entry = resolve_model(args.model)
    held_out = None if args.held_out_format == HELD_OUT_NONE else args.held_out_format
    formats = training_formats(held_out)

    arm_name = arm(held_out, args.lambda_inv, args.rank_tag)
    run_dir = train_dir(args.data_root, model_entry.slug, arm_name)
    adapter_path = run_dir / ADAPTER_NAME
    log_path = run_dir / LOG_NAME
    checkpoint_path = run_dir / CHECKPOINT_NAME

    saved = adapter_config_path(args.data_root, model_entry.slug, arm_name)
    if saved.exists() and not args.force:
        print(f"Skipping {model_entry.slug} {arm_name}: {adapter_path} is present.")
        return

    rank = args.lora_rank
    alpha = args.lora_alpha
    dropout = args.lora_dropout
    targets = args.lora_target_modules or model_entry.lora_targets
    physical_batch = args.physical_batch or model_entry.physical_batch
    grad_accum = args.grad_accum or model_entry.grad_accum
    tanh_head = model_entry.tanh_head if args.tanh_head is None else args.tanh_head

    heading(f"TRAIN  {model_entry.label}  arm={arm_name}", width=70)
    print(f"  identifier:  {model_entry.model_id}")
    print(f"  formats:     {len(formats)} of {len(FORMAT_NAMES)}  {formats}")
    if held_out is not None:
        print(f"  held out:    {held_out}")
    print(f"  lambda_inv:  {args.lambda_inv}")
    print(f"  batch:       {physical_batch} x {grad_accum}")
    print(f"  steps:       {args.max_steps}, warmup {args.warmup_steps}")
    print(f"  output:      {run_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device:      {device}")

    tokenizer = load_tokenizer(model_entry.model_id)
    adapter_kwargs = {
        key: value
        for key, value in (("rank", rank), ("alpha", alpha), ("dropout", dropout))
        if value is not None
    }
    adapted = build_adapted_model(
        model_entry.model_id,
        device,
        lora_targets=targets,
        tanh_head=tanh_head,
        eager_attn=args.eager_attn,
        gradient_checkpointing=args.grad_checkpoint,
        **adapter_kwargs,
    )
    if adapted.jina_patched:
        print(f"  fused projections patched: {adapted.jina_patched}")
    print(
        f"  trainable:   {adapted.trainable:,} / {adapted.total:,} "
        f"({100 * adapted.trainable_fraction:.2f}%)"
    )

    started = time.time()
    print("\nLoading the corpus...")
    train_split = load_records(args.data_root, "train", verbose=False)
    dev_split = load_records(args.data_root, "dev", verbose=False)
    if args.limit_train:
        train_split = train_split[: args.limit_train]
    if args.limit_dev:
        dev_split = dev_split[: args.limit_dev]
    negatives = load_negatives(args.data_root)
    corpus = load_corpus_index(args.data_root, verbose=False)

    records, dropped = prepare_train_records(
        train_split, negatives, corpus, tokenizer, args.neg_k
    )
    print(f"  train: {len(records):,} kept, {dropped:,} dropped")
    dev_records = prepare_dev_records(dev_split, tokenizer)
    print(f"  dev:   {len(dev_records):,} kept")
    report_elapsed("corpus loading", started)
    if not records or not dev_records:
        raise SystemExit("no records survived preparation; nothing to train on.")

    optimizer = torch.optim.AdamW(
        (p for p in adapted.model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )
    sampler = np.random.default_rng(args.seed)

    start_step = 0
    if args.resume and checkpoint_path.exists():
        start_step = load_checkpoint(
            checkpoint_path,
            model=adapted.model,
            optimizer=optimizer,
            scheduler=scheduler,
            sampler=sampler,
        )
        dropped_lines = rewind(log_path, start_step)
        print(
            f"\nResuming at step {start_step}; "
            f"dropped {dropped_lines} log records past it."
        )
    elif log_path.exists():
        # A fresh run must not read as a continuation of an earlier one.
        log_path.unlink()

    config = LoopConfig(
        max_steps=args.max_steps,
        physical_batch=physical_batch,
        grad_accum=grad_accum,
        eval_every=args.eval_every,
        lambda_inv=args.lambda_inv,
        checkpoint_every=args.checkpoint_every,
    )
    counter = ProgressCounter(config.max_steps, args.progress_file, done=start_step)

    print(f"\nTraining {model_entry.slug} {arm_name}")
    with TrainLog(log_path) as log:
        if start_step == 0:
            log.write(
                RUN,
                model=model_entry.slug,
                model_id=model_entry.model_id,
                arm=arm_name,
                held_out=held_out,
                formats=formats,
                lambda_inv=args.lambda_inv,
                lr=args.lr,
                weight_decay=args.weight_decay,
                lora_targets=list(targets),
                physical_batch=physical_batch,
                grad_accum=grad_accum,
                max_steps=args.max_steps,
                warmup_steps=args.warmup_steps,
                neg_k=args.neg_k,
                seed=args.seed,
                tanh_head=tanh_head,
                train_records=len(records),
                dev_records=len(dev_records),
            )
        try:
            result = train(
                adapted.model,
                tokenizer,
                records=records,
                dev_records=dev_records,
                formats=formats,
                optimizer=optimizer,
                scheduler=scheduler,
                sampler=sampler,
                config=config,
                device=device,
                log=log,
                checkpoint_path=(checkpoint_path if args.checkpoint_every else None),
                start_step=start_step,
                counter=counter,
            )
        except BaseException:
            counter.fail()
            raise

    adapted.model.save_pretrained(adapter_path)
    checkpoint_path.unlink(missing_ok=True)
    print(
        f"\nFinished at step {result.step} in "
        f"{result.elapsed:.0f}s, dev max|d|={result.dev_max_abs_d:.4f}"
    )
    print(f"Saved adapter -> {adapter_path}")


if __name__ == "__main__":
    main()
