"""The training loop that optimises the ranking and invariance objective."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fsr.formats import FORMATS
from fsr.metrics import pairwise_cohen_d, per_format_summary
from fsr.reporting import ProgressCounter
from fsr.scoring import score_batch
from fsr.training.batch import TrainRecord, compute_loss, forward_batch
from fsr.training.checkpoint import save_checkpoint, should_checkpoint
from fsr.training.log import END, EVAL, STEP, TrainLog, step_line

DEFAULT_LOG_EVERY = 20
DEFAULT_MAX_GRAD_NORM = 1.0


@dataclass(frozen=True)
class LoopConfig:
    """The settings of one training run.

    Attributes:
        max_steps: The optimiser steps the run takes.
        physical_batch: The records in one micro-batch.
        grad_accum: The micro-batches behind one optimiser step.
        eval_every: The steps between development evaluations.
        lambda_inv: The weight of the invariance term.
        log_every: The steps between progress lines.
        checkpoint_every: The steps between checkpoints. 0 disables them.
        max_grad_norm: The norm gradients are clipped to.
    """

    max_steps: int
    physical_batch: int
    grad_accum: int
    eval_every: int
    lambda_inv: float
    log_every: int = DEFAULT_LOG_EVERY
    checkpoint_every: int = 0
    max_grad_norm: float = DEFAULT_MAX_GRAD_NORM

    @property
    def effective_batch(self) -> int:
        """Return the records behind one optimiser step."""
        return self.physical_batch * self.grad_accum


@dataclass(frozen=True)
class TrainResult:
    """What a finished run reports.

    Attributes:
        step: The steps completed.
        elapsed: The seconds the loop ran.
        dev_max_abs_d: The last development score sensitivity, or None when
            the run made no evaluation.
    """

    step: int
    elapsed: float
    dev_max_abs_d: float | None


def autocast_for(device: str) -> Any:
    """Return the mixed-precision context for a device.

    Args:
        device: The device the run uses.

    Returns:
        A bfloat16 autocast context on a CUDA device, and a context that does
        nothing anywhere else.
    """
    if str(device).startswith("cuda"):
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def dev_sensitivity(
    model: Any,
    tokenizer: Any,
    records: list[TrainRecord],
    formats: list[str] | tuple[str, ...],
    device: str,
    batch_size: int,
) -> tuple[float, dict[str, Any]]:
    """Score the development records in each training format.

    Args:
        model: The model to score with.
        tokenizer: The tokenizer that matches the model.
        records: The prepared development records.
        formats: The formats the run trains on.
        device: The device to score on.
        batch_size: The pairs scored at one time.

    Returns:
        The statistics of the development sensitivity.
    """
    scores_per_fmt = {
        fmt: score_batch(
            model,
            tokenizer,
            [(r.question, FORMATS[fmt](r.pairs, r.truncated_body)) for r in records],
            batch_size,
            device,
        )
        for fmt in formats
    }
    pairwise, max_abs_d, max_d_pair = pairwise_cohen_d(scores_per_fmt, formats=formats)
    return max_abs_d, {
        "per_format": per_format_summary(scores_per_fmt),
        "pairwise": pairwise,
        "max_d_pair": max_d_pair,
    }


def train(
    model: Any,
    tokenizer: Any,
    *,
    records: list[TrainRecord],
    dev_records: list[TrainRecord],
    formats: list[str] | tuple[str, ...],
    optimizer: Any,
    scheduler: Any,
    sampler: np.random.Generator,
    config: LoopConfig,
    device: str,
    log: TrainLog,
    checkpoint_path: Path | None = None,
    start_step: int = 0,
    counter: ProgressCounter | None = None,
) -> TrainResult:
    """Run the optimiser until the step budget is spent.

    Every step accumulates `grad_accum` micro-batches, clips the gradients,
    and advances the optimiser and the scheduler. The run evaluates the
    development records on each evaluation step and on the last step.

    Args:
        model: The adapted model, in training mode.
        tokenizer: The tokenizer that matches the model.
        records: The prepared training records.
        dev_records: The prepared development records.
        formats: The formats the run trains on.
        optimizer: The optimiser.
        scheduler: The learning-rate scheduler.
        sampler: The generator that draws each micro-batch.
        config: The settings of the run.
        device: The device to train on.
        log: The log to write to.
        checkpoint_path: The checkpoint file. None disables checkpointing.
        start_step: The steps a resumed run has already taken.
        counter: The progress counter of the job, if one is running.

    Returns:
        The steps completed, the seconds spent, and the last development
        score sensitivity.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    step = start_step
    started = time.time()
    max_abs_d: float | None = None

    while step < config.max_steps:
        loss_total = 0.0
        rank_total = 0.0
        inv_total = 0.0
        for _ in range(config.grad_accum):
            drawn = sampler.choice(
                len(records), size=config.physical_batch, replace=False
            )
            batch = [records[i] for i in drawn]
            with autocast_for(device):
                pos_scores, neg_scores = forward_batch(
                    model, tokenizer, batch, formats, device
                )
                loss, l_rank, l_inv = compute_loss(
                    pos_scores, neg_scores, config.lambda_inv
                )
                scaled = loss / config.grad_accum
            scaled.backward()
            loss_total += loss.item()
            rank_total += l_rank.item()
            inv_total += l_inv.item()

        torch.nn.utils.clip_grad_norm_(trainable, config.max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        step += 1

        loss_avg = loss_total / config.grad_accum
        rank_avg = rank_total / config.grad_accum
        inv_avg = inv_total / config.grad_accum
        elapsed = time.time() - started
        learning_rate = scheduler.get_last_lr()[0]

        if step % config.log_every == 0 or step == start_step + 1:
            print(
                step_line(
                    step,
                    config.max_steps,
                    loss_avg,
                    rank_avg,
                    inv_avg,
                    learning_rate,
                    elapsed,
                ),
                flush=True,
            )
            log.write(
                STEP,
                step=step,
                loss=loss_avg,
                l_rank=rank_avg,
                l_inv=inv_avg,
                lr=learning_rate,
                elapsed_s=elapsed,
            )

        evaluated = step % config.eval_every == 0 or step == config.max_steps
        if evaluated:
            model.eval()
            with torch.no_grad():
                max_abs_d, summary = dev_sensitivity(
                    model,
                    tokenizer,
                    dev_records,
                    formats,
                    device,
                    config.effective_batch,
                )
            model.train()
            print(
                f"  dev max|d|={max_abs_d:.4f}  (pair={summary['max_d_pair']})",
                flush=True,
            )
            log.write(
                EVAL,
                step=step,
                loss=loss_avg,
                l_rank=rank_avg,
                l_inv=inv_avg,
                dev_max_abs_d=max_abs_d,
                dev_max_d_pair=summary["max_d_pair"],
                dev_summary=summary,
            )

        if counter is not None:
            label = f"loss={loss_avg:.4f}"
            if evaluated:
                label = f"{label} dev|d|={max_abs_d:.4f}"
            counter.step(label)

        if checkpoint_path is not None and should_checkpoint(
            step, config.checkpoint_every
        ):
            save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                sampler=sampler,
            )

    elapsed = time.time() - started
    log.write(END, step=step, elapsed_s=elapsed, dev_max_abs_d=max_abs_d)
    return TrainResult(step=step, elapsed=elapsed, dev_max_abs_d=max_abs_d)
