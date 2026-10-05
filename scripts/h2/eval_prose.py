"""Rank ordinary prose without relying on infobox metadata.

The mitigation phase trains on infobox metadata. This evaluates the models
on ordinary prose passages. It detects whether a model has achieved invariance
at the cost of forgetting how to rank non-metadata passages.

Writes `h2/prose_mrr_{model}_{arm}.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("RAYON_NUM_THREADS", "1")

import numpy as np
import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.corpus.layout import prose_eval_path
from fsr.corpus.prose import PROSE_NEGATIVES
from fsr.h2_layout import adapter_dir, result_path
from fsr.metrics import reciprocal_ranks
from fsr.models.loading import load_model, load_tokenizer
from fsr.reporting import ProgressCounter, heading, report_saved
from fsr.scoring import score_batch

PROSE_SPLIT = "prose"
MRR_AXIS = "mrr"


def ranking_pairs(records: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Return every query and passage pair, gold first within each record.

    Args:
        records: The prose ranking records.

    Returns:
        The pairs, in record order.
    """
    pairs = []
    for record in records:
        pairs.append((record["question"], record["positive"]["text"]))
        pairs.extend((record["question"], neg["text"]) for neg in record["negatives"])
    return pairs


def score_prose(
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    k: int,
    batch_size: int,
    device: str,
    counter: ProgressCounter | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every record against its negatives.

    Args:
        model: The model to score with.
        tokenizer: The tokenizer that matches the model.
        records: The prose ranking records.
        k: The negatives each record carries.
        batch_size: The pairs scored at one time.
        device: The device to score on.
        counter: The progress counter of the job, if one is running.

    Returns:
        The gold score of each record, and the negative scores of each record.

    Raises:
        ValueError: If a record does not carry exactly k negatives.
    """
    pairs = ranking_pairs(records)
    expected = len(records) * (k + 1)
    if len(pairs) != expected:
        raise ValueError(f"expected {expected} pairs, built {len(pairs)}")

    done = 0

    def advance(scored: int) -> None:
        nonlocal done
        while done < scored // batch_size:
            done += 1
            counter.step(f"{scored:,} pairs")

    scores = score_batch(
        model,
        tokenizer,
        pairs,
        batch_size,
        device,
        on_batch=advance if counter is not None else None,
    )
    reshaped = np.asarray(scores).reshape(len(records), k + 1)
    return reshaped[:, 0], reshaped[:, 1:]


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Rank with the untrained model instead of a trained arm",
    )
    ap.add_argument("--arm", default=None, help="The trained arm to rank with")
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--tanh-head",
        action="store_true",
        default=None,
        help="Replace the classifier head. The default is the registry entry",
    )
    ap.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help="File to keep the live progress of this job in",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help="Request the eager attention kernel in place of the fused one",
    )
    ap.add_argument("--limit", type=int, default=0, help="Cap records for a check")
    ap.add_argument(
        "--force", action="store_true", help="Rank again over a finished run"
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Rank the prose subset with one arm and write the result."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    if args.baseline == bool(args.arm):
        raise SystemExit("give either --baseline or --arm, not both or neither.")

    arm_name = "base" if args.baseline else args.arm
    out_path = result_path(args.data_root, PROSE_SPLIT, MRR_AXIS, entry.slug, arm_name)
    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {arm_name}: {out_path} is present.")
        return

    adapter = (
        None if args.baseline else adapter_dir(args.data_root, entry.slug, arm_name)
    )
    if adapter is not None and not adapter.exists():
        raise SystemExit(f"no adapter at {adapter}; train the arm first.")

    source = prose_eval_path(args.data_root)
    if not source.exists():
        raise SystemExit(f"no prose ranking set at {source}; build the corpus first.")
    corpus = json.loads(source.read_text())
    records = corpus["records"][: args.limit] if args.limit else corpus["records"]
    k = corpus["k_negatives"]
    if k != PROSE_NEGATIVES:
        print(f"Note: the ranking set carries {k} negatives, not {PROSE_NEGATIVES}.")

    batch_size = args.batch_size or entry.eval_batch
    tanh_head = entry.tanh_head if args.tanh_head is None else args.tanh_head

    heading(f"PROSE  {entry.label}  arm={arm_name}", width=70)
    print(f"  identifier: {entry.model_id}")
    print(f"  adapter:    {adapter or 'none, untrained baseline'}")
    print(f"  records:    {len(records):,} against {k} negatives each")
    print(f"  batch:      {batch_size}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device:     {device}")

    tokenizer = load_tokenizer(entry.model_id)
    model = load_model(
        entry.model_id,
        device,
        lora_adapter_path=str(adapter) if adapter else None,
        eager_attn=args.eager_attn,
        tanh_head=tanh_head,
    )

    n_pairs = len(records) * (k + 1)
    counter = ProgressCounter(-(-n_pairs // batch_size), args.progress_file)
    started = time.time()
    gold, negatives = score_prose(
        model, tokenizer, records, k, batch_size, device, counter
    )
    elapsed = time.time() - started
    print(f"\nscored {n_pairs:,} pairs in {elapsed:.1f}s")

    rr = reciprocal_ranks(gold, negatives)
    print(f"MRR: {rr.mean():.4f}  over {len(records):,} records")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model": entry.slug,
                "model_id": entry.model_id,
                "arm": arm_name,
                "adapter": str(adapter) if adapter else None,
                "split": PROSE_SPLIT,
                "source": source.name,
                "n_records_input": corpus["n_records"],
                "n_records_kept": len(records),
                "k_negatives": k,
                "batch_size": batch_size,
                "scoring_seconds": elapsed,
                "mrr": float(rr.mean()),
                "reciprocal_ranks": rr.tolist(),
                "record_ids": [r["id"] for r in records],
                "gold_scores": gold.tolist(),
                "neg_score_means": negatives.mean(axis=1).tolist(),
            }
        )
    )
    report_saved(out_path, arm_name)


if __name__ == "__main__":
    main()
