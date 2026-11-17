"""Compare a variant that also trains the pooler with its base model.

Writes `h2/pooler/{variant}.json`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import binomtest

import fsr.h2_results as h2r
from fsr.cli import add_data_root_arg, resolve_model
from fsr.comparison import IN_TRAINING, OOD
from fsr.h2_layout import h2_dir
from fsr.models.registry import by_slug
from fsr.reporting import heading, report_saved, table

OUT_SUBDIR = "pooler"
SUBSETS = {"held_out": OOD, "training": IN_TRAINING}
N_BOOT = 1000
COLUMNS = [("subset / statistic", 26), ("base", 16), ("variant", 16)]
COLUMNS += [("lower", 7), ("p", 7)]
DIGITS = {"score_pct": 1, "answer_pct": 1, "mrr_trained": 3}


def seed_row(
    data_root: Path, model: str, subset: str, seed: int, n_boot: int
) -> dict[str, float]:
    """Return the relative change on both axes for one model and seed.

    Args:
        data_root: The corpus directory.
        model: The registry slug.
        subset: The comparison subset.
        seed: The training seed.
        n_boot: The number of replicates.

    Returns:
        The per-seed statistics for the given model, subset, and seed.
    """
    label = by_slug(model).label
    kwargs = {"subset": subset, "n_boot": n_boot, "training_seed": seed}
    score = h2r.score_table(data_root, [model], **kwargs).loc[label]
    answer = h2r.answer_table(data_root, [model], **kwargs).loc[label]
    control_mrr = h2r.fold_mean_mrr(data_root, model, h2r.CONTROL_WEIGHT, seed)
    return {
        "score_control": float(score["control_max_d"]),
        "score_trained": float(score["trained_max_d"]),
        "score_pct": float(score["delta_pct"]),
        "answer_control": float(answer["control_pct"]),
        "answer_trained": float(answer["trained_pct"]),
        "answer_pct": 100.0 * float(answer["delta_pp"]) / float(answer["control_pct"]),
        "mrr_control": float(np.mean(list(control_mrr.values()))),
        "mrr_trained": float(score["mean_mrr"]),
    }


def paired(base: Sequence[float], variant: Sequence[float]) -> dict[str, Any]:
    """Summarise a per-seed statistic of both models and test the pairing.

    Args:
        base: The base model's value under each seed.
        variant: The variant's value under each seed, in the same order.

    Returns:
        The summary of the paired comparison.
    """
    b, v = np.asarray(base), np.asarray(variant)
    lower, higher = int((v < b).sum()), int((v > b).sum())
    untied = lower + higher
    return {
        "base_mean": float(b.mean()),
        "base_sd": float(b.std(ddof=1)),
        "variant_mean": float(v.mean()),
        "variant_sd": float(v.std(ddof=1)),
        "variant_lower": lower,
        "untied": untied,
        "sign_p": float(binomtest(lower, untied).pvalue) if untied else float("nan"),
    }


def spread(mean: float, sd: float, digits: int) -> str:
    """Return a mean and SD as one cell."""
    return f"{mean:+.{digits}f} ± {sd:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, help="The pooler variant's slug")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(h2r.TRAINING_SEEDS))
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Compare the variant with its base model over the seeds."""
    args = build_parser().parse_args()
    variant = resolve_model(args.model)
    if not variant.weight_from:
        raise SystemExit(f"{variant.slug} reuses no base model's weight.")
    base = resolve_model(variant.weight_from)
    rows: dict[str, Any] = {}
    for name, subset in SUBSETS.items():
        rows[name] = {
            slug: [
                seed_row(args.data_root, slug, subset, s, args.n_boot)
                for s in args.seeds
            ]
            for slug in (base.slug, variant.slug)
        }
    keys = ("score_pct", "answer_pct", "mrr_trained")
    summary = {
        name: {
            key: paired(
                [r[key] for r in by_model[base.slug]],
                [r[key] for r in by_model[variant.slug]],
            )
            for key in keys
        }
        for name, by_model in rows.items()
    }
    heading(f"{variant.label} against {base.label}")
    table(
        COLUMNS,
        [
            (
                f"{name} {key}",
                spread(s["base_mean"], s["base_sd"], DIGITS[key]),
                spread(s["variant_mean"], s["variant_sd"], DIGITS[key]),
                f"{s['variant_lower']}/{s['untied']}",
                f"{s['sign_p']:.3f}",
            )
            for name, stats in summary.items()
            for key, s in stats.items()
        ],
    )
    out = h2_dir(args.data_root) / OUT_SUBDIR / f"{variant.slug}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": variant.slug,
        "base": base.slug,
        "weight": h2r.winner_weight(h2r.load_selection(args.data_root, variant.slug)),
        "seeds": args.seeds,
        "per_seed": rows,
        "summary": summary,
    }
    out.write_text(json.dumps(payload, indent=2))
    report_saved(out)


if __name__ == "__main__":
    main()
