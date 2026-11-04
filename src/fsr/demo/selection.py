"""Finding the pools whose answer the serialisation decides.

A pool matches when the untrained model and the rank-only control both trim
the gold passage below the cut, on the mixed pool as retrieved and after
standardising every candidate to that arm's best format, while the
invariance-trained arm keeps it under the mixed pool and every uniform
format. A last check re-renders only the gold passage on the untrained model:
some other format must bring it back above the cut, so the loss is the
format's and not the content's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from fsr.demo.scenarios import WRONG_ROLE, role_index
from fsr.standardisation.conditions import realise, uniform_assignment
from fsr.standardisation.metrics import uniform_table
from fsr.standardisation.results import (
    CONTROL,
    MIXED,
    STANDARDISED,
    UNTRAINED,
    WINNER,
)
from fsr.standardisation.store import GOLD, ScoreTensor

K = 3
CONDITIONS = (UNTRAINED, CONTROL, WINNER)
HANDLINGS = (MIXED, STANDARDISED)


def ranks(realised: np.ndarray) -> np.ndarray:
    """Return the rank of every candidate of every pool, from 1.

    The sort is stable, so a tie goes to the earlier candidate. The gold
    passage is first, so a tie counts in its favour, as in
    `fsr.metrics.reciprocal_ranks`.

    Args:
        realised: One score per pool and candidate.

    Returns:
        One rank per pool and candidate.
    """
    order = np.argsort(-realised, axis=1, kind="stable")
    out = np.empty_like(order)
    rows = np.arange(len(realised))[:, None]
    out[rows, order] = np.arange(1, realised.shape[1] + 1)[None, :]
    return out


def gold_rank(realised: np.ndarray) -> np.ndarray:
    """Return the rank of the gold passage in every pool."""
    return ranks(realised)[:, GOLD]


def mixed_indices(
    pools: Sequence[dict[str, Any]], formats: Sequence[str]
) -> np.ndarray:
    """Return the mixed assignment of every pool, as format columns.

    Args:
        pools: The pools, each holding its mixed assignment by name.
        formats: The format of each column of the score tensor.

    Returns:
        One format column per pool and candidate.
    """
    column = {name: index for index, name in enumerate(formats)}
    return np.array([[column[name] for name in pool["mixed"]] for pool in pools])


def best_standard(tensor: ScoreTensor) -> str:
    """Return the format whose uniform rendering ranks best for this arm.

    Standardising to it is the strongest form of the alternative.
    """
    return str(uniform_table(tensor)["mrr"].idxmax())


def standard_assignment(tensor: ScoreTensor, name: str) -> np.ndarray:
    """Return the assignment that gives every candidate the named format."""
    return uniform_assignment(
        tensor.n_queries, tensor.n_candidates, tensor.formats.index(name)
    )


def uniform_ranks(tensor: ScoreTensor) -> np.ndarray:
    """Return the gold rank under each uniform format.

    Returns:
        One row per pool, one column per format.
    """
    return np.stack(
        [
            gold_rank(
                realise(
                    tensor.scores,
                    uniform_assignment(tensor.n_queries, tensor.n_candidates, index),
                )
            )
            for index in range(tensor.n_formats)
        ],
        axis=1,
    )


def swap_ranks(tensor: ScoreTensor, assignment: np.ndarray) -> np.ndarray:
    """Return the gold rank with only the gold passage re-rendered.

    Every negative keeps its assigned format and the gold passage takes each
    format in turn, so the content of the pool is unchanged.

    Args:
        tensor: The scores of one condition.
        assignment: The format column of every candidate.

    Returns:
        One row per pool, one column per format of the gold passage.
    """
    realised = realise(tensor.scores, assignment)
    out = []
    for index in range(tensor.n_formats):
        swapped = realised.copy()
        swapped[:, GOLD] = tensor.scores[:, GOLD, index]
        out.append(gold_rank(swapped))
    return np.stack(out, axis=1)


def check_aligned(
    tensors: Mapping[str, ScoreTensor], pools: Sequence[dict[str, Any]]
) -> None:
    """Refuse tensors that do not score the pools in order.

    Raises:
        ValueError: If any tensor's rows are not the pools, in order.
    """
    ids = [pool["id"] for pool in pools]
    for condition, tensor in tensors.items():
        if tensor.query_ids != ids:
            raise ValueError(
                f"the {condition} scores ({tensor.arm}) do not cover the pools in order"
            )


def select(
    tensors: Mapping[str, ScoreTensor],
    pools: Sequence[dict[str, Any]],
    k: int = K,
) -> dict[str, Any]:
    """Find the pools that show the failure and the fix.

    Args:
        tensors: The scores of each condition, keyed by UNTRAINED, CONTROL
            and WINNER.
        pools: The pools the tensors score, in order.
        k: The number of candidates the reranker passes on.

    Returns:
        The cut, the arm and standard format of each condition, the funnel
        of counts as each requirement is added, and the matching pools in
        order of preference.
    """
    check_aligned(tensors, pools)
    wrong = np.array([role_index(pool, WRONG_ROLE) for pool in pools])
    rows = np.arange(len(pools))

    standards, gold, wrong_kept, uniform = {}, {}, {}, {}
    swaps = {}
    for condition in CONDITIONS:
        tensor = tensors[condition]
        standards[condition] = best_standard(tensor)
        assignments = {
            MIXED: mixed_indices(pools, tensor.formats),
            STANDARDISED: standard_assignment(tensor, standards[condition]),
        }
        for handling, assignment in assignments.items():
            ranked = ranks(realise(tensor.scores, assignment))
            gold[condition, handling] = ranked[:, GOLD]
            wrong_kept[condition, handling] = ranked[rows, wrong] <= k
            swaps[condition, handling] = swap_ranks(tensor, assignment)
        uniform[condition] = uniform_ranks(tensor)

    steps = [("generated", np.ones(len(pools), dtype=bool))]
    for condition in (UNTRAINED, CONTROL):
        for handling in HANDLINGS:
            steps.append(
                (f"{condition} loses it, {handling}", gold[condition, handling] > k)
            )
    steps.append((f"{WINNER} keeps it, {MIXED}", gold[WINNER, MIXED] <= k))
    steps.append(
        (f"{WINNER} keeps it, every uniform format", uniform[WINNER].max(axis=1) <= k)
    )
    steps.append(
        (
            f"re-rendering the gold alone rescues it, {UNTRAINED}",
            (swaps[UNTRAINED, MIXED].min(axis=1) <= k)
            & (swaps[UNTRAINED, STANDARDISED].min(axis=1) <= k),
        )
    )

    keep = np.ones(len(pools), dtype=bool)
    funnel = []
    for label, flags in steps:
        keep &= flags
        funnel.append({"step": label, "n": int(keep.sum())})

    misled = np.ones(len(pools), dtype=bool)
    for condition in (UNTRAINED, CONTROL):
        for handling in HANDLINGS:
            misled &= wrong_kept[condition, handling]

    tensor_formats = tensors[UNTRAINED].formats
    matches = []
    for i in np.flatnonzero(keep):
        pool = pools[i]
        matches.append(
            {
                "id": pool["id"],
                "kind": pool["kind"],
                "target": pool["target"],
                "question": pool["question"],
                "answer": pool["answer"],
                "wrong_answer": pool["wrong_answer"],
                "wrong_in_topk": bool(misled[i]),
                "ranks": {
                    condition: {
                        MIXED: int(gold[condition, MIXED][i]),
                        STANDARDISED: int(gold[condition, STANDARDISED][i]),
                        "uniform": dict(
                            zip(
                                tensors[condition].formats,
                                map(int, uniform[condition][i]),
                                strict=True,
                            )
                        ),
                    }
                    for condition in CONDITIONS
                },
                "rescued_by": {
                    handling: [
                        name
                        for name, rank in zip(
                            tensor_formats,
                            swaps[UNTRAINED, handling][i],
                            strict=True,
                        )
                        if rank <= k
                    ]
                    for handling in HANDLINGS
                },
            }
        )
    matches.sort(
        key=lambda m: (
            not m["wrong_in_topk"],
            max(m["ranks"][WINNER]["uniform"].values()),
            -min(m["ranks"][c][h] for c in (UNTRAINED, CONTROL) for h in HANDLINGS),
        )
    )
    return {
        "k": k,
        "n_candidates": tensors[UNTRAINED].n_candidates,
        "arms": {condition: tensors[condition].arm for condition in CONDITIONS},
        "standards": standards,
        "funnel": funnel,
        "n_wrong_in_topk": int(sum(m["wrong_in_topk"] for m in matches)),
        "matches": matches,
    }
