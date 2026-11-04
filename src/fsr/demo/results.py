"""The demonstration case and the evidence that it is not noise.

The case is one question over the synthetic knowledge base, scored by every arm
under every format. The conditions each apply a single mitigation to a system
whose sources arrive in mixed formats: none, standardising the passages,
fine-tuning without the invariance term, and fine-tuning with it.
Standardising is shown at every format, ordered from the one that serves the
key passage best to the one that serves it worst, since which format a team
picks is arbitrary.

Over every assignment of formats to the source systems, the share that places
the key passage in the top k shows how far the outcome depends on the formats.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import BASE_ARM

SOURCES = ("api_reference", "changelog", "wiki", "tickets")
CONTROL_ARM = "5fmt_lam0"
WINNER_ARM = "5fmt_lam1"
# The leave-one-format-out pair of each weight, by the format withheld in training.
HOLD_OUTS = {name: (f"{name}_lam0", f"{name}_lam1") for name in FORMAT_NAMES}
MIXED = "mixed"
OTHER = "other"
FORMAT_LABELS = {
    "yaml": "YAML",
    "json": "JSON",
    "toml": "TOML",
    "inline_kv": "inline key-value",
    "markdown": "Markdown",
}


@dataclass(frozen=True)
class Condition:
    """One mitigation, as an arm and the way the passages are rendered.

    The handling is MIXED for the case's own formats, or the name of the one
    format every passage is converted to.
    """

    label: str
    arm: str
    handling: str


def case_dir(demo_root: Path) -> Path:
    """Return the directory holding the exported case."""
    return demo_root / "case"


def scores_path(demo_root: Path, stem: str, arm: str) -> Path:
    """Return the file holding one arm's scores of the case."""
    return case_dir(demo_root) / f"{stem}_scores_{arm}.npy"


@dataclass(frozen=True)
class Case:
    """The case: its pool, what each passage says, and every arm's scores.

    Attributes:
        question: The question asked.
        k: The number of passages the reranker passes on.
        key: The position of the key passage in the pool.
        passage_ids: The passages of the pool, in retrieval order.
        sources: The source system of each passage.
        claims: What each passage says the answer is, or OTHER.
        assignment: The format of each source system in the mixed deployment.
        expected: The current answer.
        stale: The answer the rest of the knowledge base supports.
        scores: Each arm's scores, shaped (passages, formats).
    """

    question: str
    k: int
    key: int
    passage_ids: list[str]
    sources: list[str]
    claims: list[str]
    assignment: dict[str, str]
    expected: str
    stale: str
    scores: dict[str, np.ndarray]


def load_case(demo_root: Path) -> Case:
    """Read the exported case.

    Args:
        demo_root: The demonstration directory.

    Returns:
        The case.
    """
    meta = json.loads((case_dir(demo_root) / "case.json").read_text())
    ids = [c["passage_id"] for c in meta["candidates"]]
    return Case(
        question=meta["question"],
        k=meta["k"],
        key=ids.index(meta["key"]),
        passage_ids=ids,
        sources=[c["source"] for c in meta["candidates"]],
        claims=[meta["claims"].get(pid, OTHER) for pid in ids],
        assignment=meta["assignment"],
        expected=meta["expected"],
        stale=meta["stale"],
        scores={
            arm: np.load(scores_path(demo_root, "case", arm)) for arm in meta["arms"]
        },
    )


def all_assignments() -> np.ndarray:
    """Return every assignment of a format to each source, shaped (625, sources)."""
    return np.array(
        list(itertools.product(range(len(FORMAT_NAMES)), repeat=len(SOURCES)))
    )


def deployed(
    scores: np.ndarray, sources: Sequence[str], assignment: np.ndarray
) -> np.ndarray:
    """Return each passage's score with its source in the assigned format.

    Args:
        scores: One arm's scores, shaped (passages, formats).
        sources: The source system of each passage.
        assignment: The format index of each source, or a stack of them.

    Returns:
        The scores, shaped (passages,) or (assignments, passages).
    """
    src = np.array([SOURCES.index(s) for s in sources])
    return scores[np.arange(len(src)), np.asarray(assignment)[..., src]]


def rank_of(scores: np.ndarray, index: int) -> np.ndarray:
    """Return the rank of one passage, 1 for the best, along the last axis."""
    return (scores > scores[..., [index]]).sum(axis=-1) + 1


def standard_order(case: Case, arm: str = BASE_ARM) -> list[str]:
    """Return the formats from the one that serves the key passage best to worst.

    Formats are ordered by the key passage's rank when every passage is in that
    format, then by its margin over the passage it must beat to make the top k.
    """
    uniform = case.scores[arm].T
    ranks = rank_of(uniform, case.key)
    others = np.sort(np.delete(uniform, case.key, axis=1), axis=1)[:, ::-1]
    gaps = uniform[:, case.key] - others[:, case.k - 1]
    order = sorted(range(len(FORMAT_NAMES)), key=lambda f: (ranks[f], -gaps[f]))
    return [FORMAT_NAMES[f] for f in order]


def conditions(case: Case) -> list[Condition]:
    """Return the conditions shown for a case, standardising to every format."""
    return [
        Condition("mixed formats, untrained", BASE_ARM, MIXED),
        *(
            Condition(f"standardise to {FORMAT_LABELS[f]}, untrained", BASE_ARM, f)
            for f in standard_order(case)
        ),
        Condition("fine-tune, $\\lambda$=0", CONTROL_ARM, MIXED),
        Condition("fine-tune, chosen $\\lambda$", WINNER_ARM, MIXED),
    ]


def condition_scores(case: Case, condition: Condition) -> tuple[np.ndarray, list[str]]:
    """Return the scores one condition gives the pool, and each passage's format."""
    scores = case.scores[condition.arm]
    if condition.handling != MIXED:
        f = FORMAT_NAMES.index(condition.handling)
        return scores[:, f], [condition.handling] * len(case.passage_ids)
    assignment = np.array([FORMAT_NAMES.index(case.assignment[s]) for s in SOURCES])
    formats = [case.assignment[s] for s in case.sources]
    return deployed(scores, case.sources, assignment), formats


def ladder(case: Case, depth: int = 6) -> pd.DataFrame:
    """Return the top of the ranking under each condition.

    Args:
        case: The case.
        depth: The number of ranks to keep.

    Returns:
        One row per condition and rank: the passage, its source, format and
        claim, whether it is the key passage, and whether it is passed on.
    """
    rows: list[dict[str, Any]] = []
    for condition in conditions(case):
        scores, formats = condition_scores(case, condition)
        for rank, i in enumerate(np.argsort(-scores, kind="stable")[:depth], 1):
            rows.append(
                {
                    "condition": condition.label,
                    "rank": rank,
                    "passage": case.passage_ids[i],
                    "source": case.sources[i],
                    "format": formats[i],
                    "claim": case.claims[i],
                    "key": i == case.key,
                    "passed_on": rank <= case.k,
                }
            )
    return pd.DataFrame(rows)


def keep_rate(scores: np.ndarray, sources: Sequence[str], key: int, k: int) -> float:
    """Return the share of format assignments that pass the key passage on.

    Args:
        scores: One arm's scores, shaped (passages, formats).
        sources: The source system of each passage.
        key: The position of the key passage.
        k: The number of passages passed on.

    Returns:
        The share of the 625 assignments with the key passage in the top k.
    """
    ranks = rank_of(deployed(scores, sources, all_assignments()), key)
    return float((ranks <= k).mean())


def case_keep_rates(case: Case) -> pd.Series:
    """Return each arm's keep rate on the case, over every format assignment."""
    return pd.Series(
        {
            arm: keep_rate(s, case.sources, case.key, case.k)
            for arm, s in case.scores.items()
        },
        name="keep_rate",
    )
