"""Selection of the value a sweep keeps."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from fsr.formats import FORMAT_NAMES
from fsr.metrics import DEFAULT_SEED, bootstrap_ci_of_mean

NI_MARGIN = 0.03


def _median(ordered: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the middle entry, taking the lower one of an even pair."""
    return ordered[(len(ordered) - 1) // 2]


def _smallest(ordered: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Return the first entry."""
    return ordered[0]


@dataclass(frozen=True)
class Sweep:
    """One swept quantity, and how a tie between its values is broken.

    Attributes:
        name: The quantity, which names the output fields.
        pattern: The expression that finds the value in a file name.
        parse: The conversion from the captured text to a sortable value.
        pick: The choice among the tied values, once they are in order.
    """

    name: str
    pattern: re.Pattern[str]
    parse: Callable[[str], float]
    pick: Callable[[Sequence[dict[str, Any]]], dict[str, Any]]


LAMBDA = Sweep(
    name="lambda",
    pattern=re.compile(r"lam([0-9.]+?)(?:_.*)?\.json$"),
    parse=float,
    pick=_median,
)

RANK = Sweep(
    name="rank",
    pattern=re.compile(r"_r(\d+)\.json$"),
    parse=int,
    pick=_smallest,
)

SWEEPS = {sweep.name: sweep for sweep in (LAMBDA, RANK)}


def extract_value(path: Path, sweep: Sweep) -> str:
    """Return the swept value a file name carries.

    Args:
        path: The evaluation file.
        sweep: The quantity to read.

    Returns:
        The value, as it appears in the name.

    Raises:
        ValueError: If the name carries no such value.
    """
    found = sweep.pattern.search(path.name)
    if not found:
        raise ValueError(f"no {sweep.name} in {path.name}")
    return found.group(1)


def mean_mrr_per_query(guardrail: dict[str, Any]) -> np.ndarray:
    """Return the reciprocal rank of each query, averaged over the formats.

    Args:
        guardrail: The ranking guardrail section of an evaluation.

    Returns:
        One value per query.
    """
    by_format = guardrail["per_format_reciprocal_ranks"]
    return np.stack(
        [np.asarray(by_format[name]) for name in FORMAT_NAMES], axis=0
    ).mean(axis=0)


def delta_mrr_ci(
    rr_base: np.ndarray,
    rr_trained: np.ndarray,
    seed: int = DEFAULT_SEED,
) -> tuple[float, float, float]:
    """Compare a trained arm against the baseline, query by query.

    Args:
        rr_base: The baseline reciprocal ranks.
        rr_trained: The trained reciprocal ranks, for the same queries.
        seed: The seed for the resample.

    Returns:
        The mean change, and the lower and upper bounds of its interval.
    """
    delta = rr_trained - rr_base
    low, high = bootstrap_ci_of_mean(delta, seed=seed)
    return float(delta.mean()), low, high


def cis_overlap(a_low: float, a_high: float, b_low: float, b_high: float) -> bool:
    """Report whether two closed intervals meet at any point."""
    return a_low <= b_high and b_low <= a_high


def candidate_row(
    path: Path,
    rr_base: np.ndarray,
    sweep: Sweep,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Summarise one candidate against the baseline.

    Args:
        path: The development evaluation of the candidate.
        rr_base: The baseline reciprocal ranks.
        sweep: The quantity being swept.
        seed: The seed for the resample.

    Returns:
        The swept value of the candidate.

    Raises:
        ValueError: If the candidate covers a different set of queries.
    """
    value = extract_value(path, sweep)
    candidate = json.loads(path.read_text())
    rr_candidate = mean_mrr_per_query(candidate["mrr_guardrail"])
    if len(rr_candidate) != len(rr_base):
        raise ValueError(
            f"{path.name} covers {len(rr_candidate)} queries, "
            f"the baseline covers {len(rr_base)}"
        )
    max_abs_d = float(candidate["format_sensitivity"]["summary"]["max_abs_cohen_d"])
    interval = candidate.get("max_abs_d_ci95")
    ci_missing = interval is None
    low, high = (
        (max_abs_d, max_abs_d)
        if ci_missing
        else (
            float(interval[0]),
            float(interval[1]),
        )
    )
    delta_mean, delta_low, delta_high = delta_mrr_ci(rr_base, rr_candidate, seed)
    return {
        sweep.name: value,
        "path": str(path),
        "dev_mean_mrr": float(rr_candidate.mean()),
        "dev_max_abs_d": max_abs_d,
        "dev_max_abs_d_ci_lo": low,
        "dev_max_abs_d_ci_hi": high,
        "ci_missing": ci_missing,
        "delta_mrr_mean": delta_mean,
        "delta_mrr_ci_lo": delta_low,
        "delta_mrr_ci_hi": delta_high,
        "mrr_ni_pass": delta_low > -NI_MARGIN,
    }


def select_winner(
    survivors: Sequence[dict[str, Any]], sweep: Sweep
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Choose among the candidates that keep ranking quality.

    Args:
        survivors: The candidates that passed the non-inferiority margin.
        sweep: The quantity being swept.

    Returns:
        The winner, and the tied group.
    """
    leader = min(survivors, key=lambda r: r["dev_max_abs_d"])
    tied = [
        r
        for r in survivors
        if cis_overlap(
            r["dev_max_abs_d_ci_lo"],
            r["dev_max_abs_d_ci_hi"],
            leader["dev_max_abs_d_ci_lo"],
            leader["dev_max_abs_d_ci_hi"],
        )
        and cis_overlap(
            r["delta_mrr_ci_lo"],
            r["delta_mrr_ci_hi"],
            leader["delta_mrr_ci_lo"],
            leader["delta_mrr_ci_hi"],
        )
    ]
    if len(tied) == 1:
        return leader, tied
    ordered = sorted(tied, key=lambda r: sweep.parse(r[sweep.name]))
    return sweep.pick(ordered), tied


def select(
    baseline_path: Path,
    candidate_paths: Sequence[Path],
    sweep: Sweep,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Run the selection rule over one sweep.

    Args:
        baseline_path: The development evaluation of the untrained model.
        candidate_paths: The development evaluation of each candidate.
        sweep: The quantity being swept.
        seed: The seed for the resample.

    Returns:
        The selection. `status` is `selected` with a winner, or `halt`.
    """
    baseline = json.loads(baseline_path.read_text())
    rr_base = mean_mrr_per_query(baseline["mrr_guardrail"])
    summary = {
        "candidates": [
            candidate_row(path, rr_base, sweep, seed) for path in candidate_paths
        ],
        "baseline_mean_mrr": float(rr_base.mean()),
        "baseline_max_abs_d": float(
            baseline["format_sensitivity"]["summary"]["max_abs_cohen_d"]
        ),
        "ni_margin": NI_MARGIN,
        "swept": sweep.name,
    }
    survivors = [r for r in summary["candidates"] if r["mrr_ni_pass"]]
    if not survivors:
        return {
            "status": "halt",
            "reason": f"no_{sweep.name}_passed_mrr_non_inferiority",
            **summary,
        }
    winner, tied = select_winner(survivors, sweep)
    tie_broken = len(tied) > 1
    return {
        "status": "selected",
        f"winner_{sweep.name}": winner[sweep.name],
        "winner_summary": winner,
        "tie_broken": tie_broken,
        f"tied_{sweep.name}s": [r[sweep.name] for r in tied] if tie_broken else [],
        **summary,
    }
