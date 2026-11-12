"""Re-score the H1 passages that exceeded one model's context.

The body budget is the smallest over the roster, but the body was cut by
counting tokens with the one tokenizer that set the budget, so a few renderings
still exceed another model's context and that model truncated the end of the
body itself. This script takes the records and queries `overflow.py` found for
one model, cuts every passage of them so it fits every tokenizer, and scores
them again under the five formats.

A control set of records that never overflowed is scored too. Their scores
should match the original run, which confirms the scoring set-up is unchanged.

Writes `{split}_rescore_{model}.json` to the output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from fsr.cli import add_data_root_arg, resolve_model
from fsr.corpus.files import write_atomic
from fsr.corpus.splitting import load_records
from fsr.formats import FORMAT_NAMES, FORMATS
from fsr.h1_results import load_overflow, rescore_path, result_path
from fsr.models.loading import load_model, load_tokenizer
from fsr.models.registry import BASE_MODEL_IDS
from fsr.overflow import query_items, record_items, refit_body, scan
from fsr.passages import prepare_records_with_body
from fsr.reporting import ProgressCounter
from fsr.scoring import score_batch
from fsr.within_query import gold_leads, within_query_mrr

SPLITS = ("test", "nq_val")
CONTROL_RECORDS = 20


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument("--split", choices=SPLITS, default="test")
    ap.add_argument("--model", required=True, help="Registry slug of the model")
    ap.add_argument("--cross-batch-size", type=int, default=16)
    ap.add_argument("--within-batch-size", type=int, default=32)
    ap.add_argument("--progress-file", type=Path, default=None)
    return ap


def refit(question: str, record: dict[str, Any], tokenizers: dict[str, Any]) -> str:
    """Return a body that fits every tokenizer, or stop.

    Args:
        question: The query text.
        record: The corpus record, with its pairs and full body.
        tokenizers: The tokenizers to fit.

    Returns:
        The truncated body.

    Raises:
        SystemExit: If a tokenizer leaves too small a budget.
    """
    body = refit_body(question, record["pairs"], record["body"], tokenizers)
    if body is None:
        raise SystemExit(f"{record['id']} leaves too small a budget once refitted")
    return body


def control_ids(ids: list[str], excluded: set[str], n: int) -> list[str]:
    """Return n ids spread evenly over those not excluded.

    Args:
        ids: Every id, in result order.
        excluded: The ids to pass over.
        n: How many to return.

    Returns:
        The chosen ids, in result order.
    """
    eligible = [i for i in ids if i not in excluded]
    if len(eligible) <= n:
        return eligible
    picks = np.linspace(0, len(eligible) - 1, n).round().astype(int)
    return [eligible[i] for i in picks]


def score_records(
    model: Any,
    tokenizer: Any,
    records: list[dict],
    batch_size: int,
    device: str,
    counter: ProgressCounter,
) -> dict[str, list[float]]:
    """Score each record under each format.

    Args:
        model: The reranker.
        tokenizer: Its tokenizer.
        records: Records with question, pairs, and truncated_body.
        batch_size: Pairs scored at once.
        device: The device to score on.
        counter: The progress counter of the job.

    Returns:
        One score per record, by format.
    """
    scores = {}
    for name in FORMAT_NAMES:
        render = FORMATS[name]
        pairs = [
            (r["question"], render(r["pairs"], r["truncated_body"])) for r in records
        ]
        scores[name] = score_batch(model, tokenizer, pairs, batch_size, device)
        counter.step(f"cross/{name}")
    return scores


def score_queries(
    model: Any,
    tokenizer: Any,
    prepared: list[dict],
    batch_size: int,
    device: str,
    counter: ProgressCounter,
) -> dict[str, np.ndarray]:
    """Score every candidate of each query under each format.

    Args:
        model: The reranker.
        tokenizer: Its tokenizer.
        prepared: Queries with their candidate lists, gold first.
        batch_size: Pairs scored at once.
        device: The device to score on.
        counter: The progress counter of the job.

    Returns:
        One (queries, candidates) matrix per format.
    """
    n_candidates = len(prepared[0]["candidates"])
    matrices = {}
    for name in FORMAT_NAMES:
        render = FORMATS[name]
        pairs = [
            (q["question"], render(c["pairs"], c["truncated_body"]))
            for q in prepared
            for c in q["candidates"]
        ]
        flat = score_batch(model, tokenizer, pairs, batch_size, device)
        matrices[name] = np.asarray(flat, dtype=float).reshape(-1, n_candidates)
        counter.step(f"within/{name}")
    return matrices


def check_fit(items: list, tokenizers: dict[str, Any], what: str) -> None:
    """Stop if any refitted rendering still exceeds a model's context.

    Args:
        items: The refitted scan items.
        tokenizers: Every tokenizer of the roster.
        what: The name of the items, for the message.

    Raises:
        SystemExit: If a rendering still overflows.
    """
    left = {n: len(v["ids"]) for n, v in scan(items, tokenizers).items() if v["ids"]}
    if left:
        raise SystemExit(f"refitted {what} still overflow: {left}")


def main() -> None:
    """Re-score one model's overflowing items and write the result."""
    args = build_parser().parse_args()
    model_id = resolve_model(args.model).model_id
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\nModel: {model_id}")

    overflow = load_overflow(args.data_root, args.split)
    cross_drop = set(overflow["cross"]["by_model"][model_id]["ids"])
    within_drop = set(overflow["within"]["by_model"][model_id]["ids"])
    print(f"  {len(cross_drop)} records, {len(within_drop)} queries overflowed")

    # The control pass always runs; the other two run when there is anything
    # to re-score. The counter starts here so the watcher sees the job while
    # the passages are refitted, which takes minutes for the larger models.
    phases = 1 + int(bool(cross_drop)) + int(bool(within_drop))
    counter = ProgressCounter(phases * len(FORMAT_NAMES), args.progress_file)
    counter.label = "refitting"
    counter.write()

    print(f"Loading tokenizers for {len(BASE_MODEL_IDS)} models...")
    tokenizers = {m: load_tokenizer(m, trust_remote_code=True) for m in BASE_MODEL_IDS}
    corpus = {r["id"]: r for r in load_records(args.data_root, "all", verbose=False)}

    eligible, _ = prepare_records_with_body(
        load_records(args.data_root, args.split, verbose=False), tokenizers
    )
    by_id = {r["id"]: r for r in eligible}
    cross_ids = [i for i in overflow["cross"]["record_ids"] if i in cross_drop]
    cross_records = [
        {
            **by_id[i],
            "truncated_body": refit(by_id[i]["question"], by_id[i], tokenizers),
        }
        for i in cross_ids
    ]
    controls = [
        by_id[i]
        for i in control_ids(
            overflow["cross"]["record_ids"], cross_drop, CONTROL_RECORDS
        )
    ]
    check_fit(record_items(cross_records), tokenizers, "records")

    cache = args.data_root / "h1" / f"{args.split}_candidates.json"
    prepared = json.loads(cache.read_text())["prepared"]
    queries = [
        {
            **q,
            "candidates": [
                {
                    **c,
                    "truncated_body": refit(q["question"], corpus[c["id"]], tokenizers),
                }
                for c in q["candidates"]
            ],
        }
        for q in prepared
        if q["id"] in within_drop
    ]
    check_fit(query_items(queries), tokenizers, "queries")
    print(f"  refitted {len(cross_records)} records and {len(queries)} queries")

    tokenizer = tokenizers[model_id]
    model = load_model(model_id, device, trust_remote_code=True)

    control_scores = score_records(
        model, tokenizer, controls, args.cross_batch_size, device, counter
    )
    original = json.loads(
        result_path(
            args.data_root, args.split, "cross", "with_body", args.model
        ).read_text()
    )
    position = {i: n for n, i in enumerate(original["record_ids"])}
    stored = original["results"][model_id]["scores"]
    control_diff = max(
        abs(control_scores[f][k] - stored[f][position[r["id"]]])
        for f in FORMAT_NAMES
        for k, r in enumerate(controls)
    )
    print(f"  control records: largest score difference {control_diff:.2e}")

    out: dict[str, Any] = {
        "split": args.split,
        "model": model_id,
        "control": {
            "record_ids": [r["id"] for r in controls],
            "max_abs_diff": control_diff,
        },
        "cross": {"record_ids": cross_ids, "scores": {}},
        "within": {"query_ids": [q["id"] for q in queries]},
    }
    if cross_records:
        out["cross"]["scores"] = score_records(
            model, tokenizer, cross_records, args.cross_batch_size, device, counter
        )
    if queries:
        matrices = score_queries(
            model, tokenizer, queries, args.within_batch_size, device, counter
        )
        out["within"]["reciprocal_ranks"] = {
            f: v["reciprocal_ranks"] for f, v in within_query_mrr(matrices).items()
        }
        out["within"]["gold_leads"] = {
            f: v.tolist() for f, v in gold_leads(matrices).items()
        }
        out["within"]["matrices"] = {f: m.tolist() for f, m in matrices.items()}

    path = rescore_path(args.data_root, args.split, args.model)
    write_atomic(path, json.dumps(out))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
