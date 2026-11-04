"""What each arm passes on to the generator, for one pool.

The export renders each candidate exactly as the reranker scored it, so the
top-k text can go to any LLM by hand.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from fsr.demo.selection import (
    CONDITIONS,
    K,
    mixed_indices,
    ranks,
    standard_assignment,
)
from fsr.formats import FORMATS
from fsr.standardisation.conditions import realise
from fsr.standardisation.results import MIXED, STANDARDISED
from fsr.standardisation.store import GOLD, ScoreTensor

PROMPT = (
    "Answer the question using only the passages below. "
    "If they do not contain the answer, say so."
)


def ranked_candidates(
    tensor: ScoreTensor, pool: dict[str, Any], row: int, assignment: np.ndarray
) -> list[dict[str, Any]]:
    """Return every candidate of one pool, in the order the reranker puts them.

    Args:
        tensor: The scores of one condition.
        pool: The pool, at row `row` of the tensor.
        row: The pool's row in the tensor.
        assignment: The format column of every candidate of every pool.

    Returns:
        One entry per candidate, best first, with its rank, role, format,
        score and rendered text.
    """
    realised = realise(tensor.scores, assignment)[row]
    ranked = ranks(realised[None, :])[0]
    out = []
    for index in np.argsort(ranked):
        candidate = pool["candidates"][index]
        name = tensor.formats[assignment[row, index]]
        out.append(
            {
                "rank": int(ranked[index]),
                "index": int(index),
                "role": candidate["role"],
                "format": name,
                "score": float(realised[index]),
                "text": FORMATS[name](candidate["pairs"], candidate["truncated_body"]),
            }
        )
    return out


def pipeline_views(
    tensors: Mapping[str, ScoreTensor],
    pools: list[dict[str, Any]],
    standards: Mapping[str, str],
    scenario_id: str,
) -> list[dict[str, Any]]:
    """Return the full reranked list of one pool, per condition and handling.

    Args:
        tensors: The scores of each condition.
        pools: The pools the tensors score, in order.
        standards: The standard format of each condition.
        scenario_id: The pool to show.

    Returns:
        One view per condition and handling, holding its ranked candidates
        and the gold passage's rank.
    """
    row = next(i for i, pool in enumerate(pools) if pool["id"] == scenario_id)
    views = []
    for condition in CONDITIONS:
        tensor = tensors[condition]
        assignments = {
            MIXED: mixed_indices(pools, tensor.formats),
            STANDARDISED: standard_assignment(tensor, standards[condition]),
        }
        for handling, assignment in assignments.items():
            ranked = ranked_candidates(tensor, pools[row], row, assignment)
            views.append(
                {
                    "condition": condition,
                    "arm": tensor.arm,
                    "handling": handling,
                    "standard": standards[condition]
                    if handling == STANDARDISED
                    else None,
                    "gold_rank": next(c["rank"] for c in ranked if c["index"] == GOLD),
                    "candidates": ranked,
                }
            )
    return views


def topk_markdown(pool: dict[str, Any], views: list[dict[str, Any]], k: int = K) -> str:
    """Return the top-k text each view passes on, ready to paste into an LLM.

    Each view's notes sit outside its fenced passages, so the passages can be
    copied without them.

    Args:
        pool: The pool shown.
        views: The views from `pipeline_views`.
        k: The number of candidates passed on.

    Returns:
        The Markdown document.
    """
    lines = [
        f"# {pool['id']}: {pool['question']}",
        "",
        f"Expected answer: {pool['answer']}. "
        f"Wrong answer on offer: {pool['wrong_answer']}.",
        "",
        f"Prompt: {PROMPT}",
    ]
    for view in views:
        handling = view["handling"]
        if view["standard"]:
            handling = f"{handling} to {view['standard']}"
        kept = "kept" if view["gold_rank"] <= k else "lost"
        lines += [
            "",
            f"## {view['condition']} ({view['arm']}), {handling}",
            "",
            f"Gold passage {kept} at rank {view['gold_rank']} of "
            f"{len(view['candidates'])}.",
            "",
            "```text",
            f"Question: {pool['question']}",
        ]
        for candidate in view["candidates"][:k]:
            lines += ["", f"Passage {candidate['rank']}:", candidate["text"]]
        lines.append("```")
    return "\n".join(lines) + "\n"
