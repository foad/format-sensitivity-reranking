"""Score knowledge-base variants as they are pushed, under every arm and format.

The worker keeps every arm loaded and watches the inbox of the iteration
loop. It scores each ready job, writes the scores to the outbox, and stops
once it has been idle for a while or has run for its wall-clock cap, so it
never holds a GPU that is doing nothing. Launch it again to resume: the score
cache in the outbox means no pair is scored twice.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.demo.layout import DEFAULT_DEMO_ROOT, inbox_dir, outbox_dir
from fsr.demo.loop import DEFAULT_IDLE, DEFAULT_POLL, DEFAULT_WALL, serve
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM, adapter_dir, arm
from fsr.h2_results import load_selection, winner_weight
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import ProgressCounter, heading, stamp
from fsr.scoring import MAX_TOKENS, score_batch
from fsr.standardisation.results import arm_names

HOLD_OUT = "json"


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="bge_base", help="Registry slug")
    ap.add_argument(
        "--arms",
        default=None,
        help="Comma-separated arms. The default is the study's three and the "
        "JSON hold-out pair",
    )
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--worker", default="w0", help="Names this worker's claims")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--poll", type=float, default=DEFAULT_POLL)
    ap.add_argument("--idle-minutes", type=float, default=DEFAULT_IDLE / 60)
    ap.add_argument("--wall-hours", type=float, default=DEFAULT_WALL / 3600)
    ap.add_argument("--progress-file", type=Path, default=None)
    add_data_root_arg(ap)
    return ap


def default_arms(data_root: Path, model: str) -> list[str]:
    """Return the study's arms, then the JSON hold-out pair at the same weights."""
    names = list(arm_names(data_root, model).values())
    weight = winner_weight(load_selection(data_root, model))
    return [*names, arm(HOLD_OUT, 0), arm(HOLD_OUT, weight)]


def main() -> None:
    """Load every arm and serve the inbox."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    arms = (
        args.arms.split(",") if args.arms else default_arms(args.data_root, entry.slug)
    )
    adapters = {
        name: None
        if name == BASE_ARM
        else adapter_dir(args.data_root, entry.slug, name)
        for name in arms
    }
    absent = [str(path) for path in adapters.values() if path and not path.exists()]
    if absent:
        raise SystemExit(f"no adapter at {', '.join(absent)}; train the arm first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.batch_size or entry.eval_batch
    inbox, outbox = inbox_dir(args.demo_root), outbox_dir(args.demo_root)

    heading(f"DEMO LOOP WORKER  {entry.label}  worker={args.worker}", width=70)
    print(f"  identifier: {entry.model_id}")
    print(f"  arms:       {' '.join(arms)}")
    print(f"  formats:    {' '.join(FORMAT_NAMES)}")
    print(f"  inbox:      {inbox}")
    print(f"  outbox:     {outbox}")
    print(f"  batch:      {batch_size}")
    print(f"  device:     {device}")
    print(f"  stops:      {args.idle_minutes:g} min idle, {args.wall_hours:g} h in all")

    tokenizer = load_tokenizer(entry.model_id)
    models = {
        name: load_model(
            entry.model_id,
            device,
            lora_adapter_path=str(path) if path else None,
            tanh_head=entry.tanh_head,
        )
        for name, path in adapters.items()
    }
    if device == "cuda":
        print(f"  memory:     {torch.cuda.memory_allocated() / 2**30:.1f} GiB loaded")

    def scorer(name, pairs):
        return score_batch(models[name], tokenizer, pairs, batch_size, device)

    def too_long(question, text):
        return len(tokenizer(question, text)["input_ids"]) > MAX_TOKENS

    progress: dict[str, ProgressCounter] = {}

    def on_arm(name, scored):
        progress["job"].step(f"{name}: {scored} pairs")

    def on_job(summary):
        if summary["failed"]:
            progress["job"].fail()
            print(f"{stamp()} {summary['job']} FAILED: {summary['error']}", flush=True)
            return
        print(
            f"{stamp()} {summary['job']}: {summary['pairs']} pairs, "
            f"{summary['scored']} scored, {summary['truncated_pairs']} truncated, "
            f"{summary['seconds']}s",
            flush=True,
        )

    def on_start(_name):
        progress["job"] = ProgressCounter(len(arms), args.progress_file)

    started = time.time()
    print(f"{stamp()} watching the inbox", flush=True)
    reason = serve(
        inbox,
        outbox,
        args.worker,
        arms,
        scorer,
        too_long=too_long,
        poll=args.poll,
        idle=args.idle_minutes * 60,
        wall=args.wall_hours * 3600,
        on_job=on_job,
        on_arm=on_arm,
        on_start=on_start,
    )
    print(
        f"{stamp()} stopping: "
        + ("idle" if reason == "idle" else "wall-clock cap reached")
        + f" after {(time.time() - started) / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
