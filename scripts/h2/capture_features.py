"""Cache the encoder representation a frozen model's head reads, per format.

Scores the gold passage and its mined negatives in every format, and keeps
the first-position encoder output for each one. Cuts at the CLS token.

Writes `h2/head_probe/{split}_features_{model}.npy`, the matching `_scores_`
file, and a JSON description.
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
from fsr.corpus.splitting import load_records
from fsr.features import features_and_scores
from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import (
    HEAD_PROBE_SPLITS,
    feature_meta_path,
    feature_path,
    feature_score_path,
)
from fsr.models.loading import load_model, load_tokenizer
from fsr.models.registry import BASE_MODEL_IDS
from fsr.reporting import ProgressCounter, heading, report_elapsed, report_saved
from fsr.training.batch import render_negative, render_positive
from fsr.training.data import (
    load_corpus_index,
    load_negatives,
    prepare_train_records,
)

HEAD_PROBE_NEGATIVES = 15


def load_budget_tokenizers(names: list[str]) -> dict[str, Any]:
    """Load the tokenizers the body budget is measured across.

    Args:
        names: Registry slugs or Hugging Face identifiers.

    Returns:
        The tokenizers, by identifier.

    Raises:
        SystemExit: If any tokenizer fails to load.
    """
    loaded = {}
    for name in names:
        model_id = resolve_model(name).model_id
        if model_id in loaded:
            continue
        try:
            loaded[model_id] = load_tokenizer(model_id)
        except Exception as error:
            raise SystemExit(
                f"the budget tokenizer {model_id} did not load ({error}). The "
                "body budget comes from the whole roster, so fix the load and "
                "run again."
            ) from error
    return loaded


def candidate_pairs(records: list, format_name: str) -> list[tuple[str, str]]:
    """Return the query and passage pairs of one format, gold first.

    Args:
        records: The prepared records.
        format_name: The format to render in.

    Returns:
        One pair per candidate, grouped by record in the order given.
    """
    pairs = []
    for record in records:
        pairs.append((record.question, render_positive(record, format_name)))
        for index in range(len(record.neg_pairs_list)):
            pairs.append((record.question, render_negative(record, index, format_name)))
    return pairs


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="Registry slug or identifier")
    ap.add_argument(
        "--split",
        default="train",
        choices=list(HEAD_PROBE_SPLITS),
        help="Split to cache",
    )
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument(
        "--negatives",
        type=int,
        default=HEAD_PROBE_NEGATIVES,
        help="Negatives cached for each record",
    )
    ap.add_argument(
        "--budget-models",
        nargs="+",
        default=list(BASE_MODEL_IDS),
        help="Tokenizers the per-record body budgets come from",
    )
    ap.add_argument(
        "--eager-attn",
        action="store_true",
        help="Request the eager attention kernel in place of the fused one",
    )
    ap.add_argument(
        "--tanh-head",
        action="store_true",
        default=None,
        help="Replace the classifier head. The default is the registry entry",
    )
    ap.add_argument("--progress-file", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0, help="Cap records for a check")
    ap.add_argument("--force", action="store_true", help="Rebuild a present cache")
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Cache one model's representations and write them with their scores."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    out_path = feature_path(args.data_root, args.split, entry.slug)
    score_path = feature_score_path(args.data_root, args.split, entry.slug)
    meta_path = feature_meta_path(args.data_root, args.split, entry.slug)

    if out_path.exists() and not args.force:
        print(f"Skipping {entry.slug} {args.split}: {out_path} is present.")
        return

    batch_size = args.batch_size or entry.eval_batch
    tanh_head = entry.tanh_head if args.tanh_head is None else args.tanh_head
    device = "cuda" if torch.cuda.is_available() else "cpu"

    heading(f"FEATURES  {entry.label}  split={args.split}", width=70)
    print(f"  identifier:  {entry.model_id}")
    print(f"  device:      {device}")
    print(f"  batch:       {batch_size}")
    print(f"  output:      {out_path}")

    records = load_records(args.data_root, args.split, verbose=False)
    if args.limit:
        records = records[: args.limit]
    budget_tokenizers = load_budget_tokenizers(args.budget_models)
    prepared, dropped = prepare_train_records(
        records,
        load_negatives(args.data_root),
        load_corpus_index(args.data_root, verbose=False),
        budget_tokenizers,
        neg_k=args.negatives,
    )
    print(f"\n{len(prepared):,} records kept, {dropped:,} dropped")
    if not prepared:
        raise SystemExit("no records survived preparation; nothing to cache.")

    tokenizer = load_tokenizer(entry.model_id)
    model = load_model(
        entry.model_id, device, eager_attn=args.eager_attn, tanh_head=tanh_head
    )

    n_candidates = 1 + args.negatives
    counter = ProgressCounter(len(FORMAT_NAMES), args.progress_file)
    started = time.time()
    features = None
    scores = np.empty(
        (len(prepared), n_candidates, len(FORMAT_NAMES)), dtype=np.float32
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    for index, name in enumerate(FORMAT_NAMES):
        print(f"\nScoring {name}...")
        rows, row_scores = features_and_scores(
            model, tokenizer, candidate_pairs(prepared, name), batch_size, device
        )
        if features is None:
            features = np.lib.format.open_memmap(
                out_path,
                mode="w+",
                dtype=np.float32,
                shape=(
                    len(prepared),
                    n_candidates,
                    len(FORMAT_NAMES),
                    rows.shape[1],
                ),
            )
        features[:, :, index, :] = rows.reshape(len(prepared), n_candidates, -1)
        scores[:, :, index] = np.asarray(row_scores, dtype=np.float32).reshape(
            len(prepared), n_candidates
        )
        counter.step(f"features/{name}")

    features.flush()
    np.save(score_path, scores)
    meta_path.write_text(
        json.dumps(
            {
                "model": entry.slug,
                "model_id": entry.model_id,
                "split": args.split,
                "tanh_head": tanh_head,
                "formats": list(FORMAT_NAMES),
                "n_records": len(prepared),
                "n_candidates": n_candidates,
                "n_negatives": args.negatives,
                "feature_dim": int(features.shape[-1]),
                "budget_models": [
                    resolve_model(n).model_id for n in args.budget_models
                ],
                "record_ids": [r.id for r in prepared],
                "dropped": dropped,
                "limit": args.limit,
            },
            indent=2,
        )
    )
    report_elapsed("feature capture", started)
    report_saved(out_path, f"{entry.slug} {args.split}")


if __name__ == "__main__":
    main()
