"""H2 pooler study: each variant that also trains the pooler against its base."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open
from scipy.stats import binomtest

from fsr.formats import FORMAT_NAMES
from fsr.h2_layout import adapter_dir, arm
from fsr.h2_results import (
    CONTROL_WEIGHT,
    TRAINING_SEEDS,
    baseline_pool,
    fold_max_d,
    fold_mean_mrr,
    inconsistent_matrix,
    load_selection,
    pooled_inconsistency,
    winner_weight,
)
from fsr.models.registry import MODELS, by_slug

ADAPTER_FILE = "adapter_model.safetensors"
WEIGHTS_FILE = "model.safetensors"
PEFT_PREFIX = "base_model.model."
LORA_MARK = "lora_"
CONTROL = "control"
LORA = "lora"
DENSE = "dense"
OUTPUT = "output"
CHOSEN = "chosen"
PAIRS = tuple((m.weight_from, m.slug) for m in MODELS if m.head_modules)


def seed_levels(
    data_root: Path,
    slug: str,
    subset: str,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Return both axes and ranking quality of one model under each seed.

    Args:
        data_root: The corpus directory.
        slug: The registry slug.
        subset: The comparison subset.
        training_seeds: The training seeds.

    Returns:
        A DataFrame indexed by seed, containing the score, answer, and MRR
        as well as the deltas and percentage changes from the control.
    """
    weight = winner_weight(load_selection(data_root, slug))
    pool = baseline_pool(data_root, slug)
    rows = {}
    for seed in training_seeds:
        row = {}
        for name, value in ((CONTROL, CONTROL_WEIGHT), (CHOSEN, weight)):
            max_d = fold_max_d(data_root, slug, value, subset, seed)
            matrix = inconsistent_matrix(data_root, slug, value, subset, seed)
            mrr = fold_mean_mrr(data_root, slug, value, seed)
            row[f"score_{name}"] = float(np.mean(list(max_d.values())))
            row[f"answer_{name}"] = pooled_inconsistency(matrix, pool)
            row[f"mrr_{name}"] = float(np.mean(list(mrr.values())))
        for axis in ("score", "answer"):
            control = row[f"{axis}_{CONTROL}"]
            change = row[f"{axis}_{CHOSEN}"] - control
            row[f"{axis}_delta"] = change
            row[f"{axis}_pct"] = 100.0 * change / control if control else np.nan
        rows[seed] = row
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("seed")


def sign_test(base: Sequence[float], variant: Sequence[float]) -> dict[str, float]:
    """Pair two models by seed and test which is lower.

    Args:
        base: The base model's value under each seed.
        variant: The variant's value under each seed, in the same order.

    Returns:
        The seeds where the variant is lower, the untied seeds, and the
        two-sided sign test p-value over the untied seeds.
    """
    b, v = np.asarray(base, dtype=float), np.asarray(variant, dtype=float)
    lower, untied = int((v < b).sum()), int((v != b).sum())
    p = float(binomtest(lower, untied).pvalue) if untied else float("nan")
    return {"variant_lower": lower, "untied": untied, "sign_p": p}


def paired_table(
    data_root: Path,
    column: str,
    subset: str,
    pairs: Sequence[tuple[str, str]] = PAIRS,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Compare one per-seed statistic between each base model and its variant.

    Args:
        data_root: The corpus directory.
        column: A column of `seed_levels`.
        subset: The comparison subset.
        pairs: The base and variant slugs.
        training_seeds: The training seeds.

    Returns:
        One row per base model label: the mean and sample SD of each model, the
        seeds where the variant is lower and the sign test.
    """
    rows = {}
    for base, variant in pairs:
        values = {
            slug: seed_levels(data_root, slug, subset, training_seeds)[column]
            for slug in (base, variant)
        }
        rows[by_slug(base).label] = {
            "base_mean": float(values[base].mean()),
            "base_sd": float(values[base].std(ddof=1)),
            "variant_mean": float(values[variant].mean()),
            "variant_sd": float(values[variant].std(ddof=1)),
            **sign_test(values[base], values[variant]),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def seed_pairs(
    data_root: Path,
    column: str,
    subset: str,
    pairs: Sequence[tuple[str, str]] = PAIRS,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Return one per-seed statistic of each base model and its variant.

    Args:
        data_root: The corpus directory.
        column: A column of `seed_levels`.
        subset: The comparison subset.
        pairs: The base and variant slugs.
        training_seeds: The training seeds.

    Returns:
        One row per base model label and seed, with the base and the variant
        value side by side and the variant less the base.
    """
    frames = {}
    for base, variant in pairs:
        frames[by_slug(base).label] = pd.DataFrame(
            {
                "base": seed_levels(data_root, base, subset, training_seeds)[column],
                "variant": seed_levels(data_root, variant, subset, training_seeds)[
                    column
                ],
            }
        )
    frame = pd.concat(frames, names=["model"])
    frame["difference"] = frame["variant"] - frame["base"]
    return frame


def adapter_tensors(path: Path) -> dict[str, np.ndarray]:
    """Read every tensor of an adapter, with the PEFT prefix removed."""
    with safe_open(path, "pt") as handle:
        names = handle.keys()
        return {
            key.removeprefix(PEFT_PREFIX): handle.get_tensor(key).float().numpy()
            for key in names
        }


def head_role(module: str) -> str:
    """Return whether a fully trained head module is the dense layer or the output."""
    return DENSE if module.endswith(DENSE) else OUTPUT


def trainable_parameters(data_root: Path, slugs: Sequence[str]) -> pd.DataFrame:
    """Count the parameters each model trains, from one of its adapters.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs.

    Returns:
        One row per model label: the LoRA parameters, the dense layer and the
        output layer of the head, and the total.
    """
    rows = {}
    for slug in slugs:
        tensors = adapter_tensors(
            adapter_dir(data_root, slug, arm(FORMAT_NAMES[0], CONTROL_WEIGHT))
            / ADAPTER_FILE
        )
        row = {LORA: 0, DENSE: 0, OUTPUT: 0}
        for key, value in tensors.items():
            role = LORA if LORA_MARK in key else head_role(key.rsplit(".", 1)[0])
            row[role] += value.size
        row["total"] = sum(row.values())
        rows[by_slug(slug).label] = row
    return pd.DataFrame.from_dict(rows, orient="index")


def relative_change(trained: np.ndarray, pretrained: np.ndarray) -> float:
    """Return the norm of the change as a share of the pretrained norm."""
    return float(np.linalg.norm(trained - pretrained) / np.linalg.norm(pretrained))


def cached_weights(slug: str) -> Path:
    """Return the pretrained weights of one model from the local cache."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(by_slug(slug).model_id, WEIGHTS_FILE, local_files_only=True)
    )


def head_weight_change(
    data_root: Path,
    slugs: Sequence[str],
    weights_of: Callable[[str], Path] = cached_weights,
    training_seeds: Sequence[int] = TRAINING_SEEDS,
) -> pd.DataFrame:
    """Measure how far training moved each fully trained head weight matrix.

    Args:
        data_root: The corpus directory.
        slugs: The registry slugs.
        weights_of: Returns the pretrained weights file of a slug.
        training_seeds: The training seeds.

    Returns:
        A DataFrame indexed by model label and weight matrix, showing how
        much each weight matrix changed under the control and chosen weights.
    """
    rows = {}
    for slug in slugs:
        chosen = winner_weight(load_selection(data_root, slug))
        with safe_open(weights_of(slug), "pt") as handle:
            changes: dict[tuple[str, str], list[float]] = {}
            for name, value in ((CONTROL, CONTROL_WEIGHT), (CHOSEN, chosen)):
                for seed in training_seeds:
                    for fold in FORMAT_NAMES:
                        path = adapter_dir(data_root, slug, arm(fold, value, seed=seed))
                        tensors = adapter_tensors(path / ADAPTER_FILE)
                        for key, trained in tensors.items():
                            if LORA_MARK in key or not key.endswith("weight"):
                                continue
                            before = handle.get_tensor(key).float().numpy()
                            changes.setdefault((key, name), []).append(
                                relative_change(trained, before)
                            )
        label = by_slug(slug).label
        for (key, name), values in changes.items():
            rows.setdefault((label, key.removesuffix(".weight")), {})[name] = float(
                np.mean(values)
            )
    frame = pd.DataFrame.from_dict(rows, orient="index")
    frame.index.names = ["model", "module"]
    frame.insert(0, "role", [head_role(m) for m in frame.index.get_level_values(1)])
    return frame[["role", CONTROL, CHOSEN]]
