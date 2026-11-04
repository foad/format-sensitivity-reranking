"""A stratified grid of knowledge-base states, wordings and format combinations.

The cross-encoder scores each question and passage pair on its own, so a
passage's score under one wording and format does not depend on the other
passages in the pool. Every pair is therefore scored once, and the ranking of
any knowledge-base state is assembled offline: retrieve with BM25 over that
state, look up the retrieved passages' scores, and rank them under every
assignment of formats to the source systems.

A state gives each passage one of its versions, or leaves an optional passage
out. The passages that compete near the cut are crossed in full. Each of
those states is paired with the reference setting of every other passage, the
one the case uses, and with one random draw of them, so the grid also shows
whether the passages far from the cut matter.
"""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

from fsr.demo.results import all_assignments, deployed, rank_of

ABSENT = -1
REFERENCE = "reference"
DRAWN = "drawn"
ORIGINAL = "original"


def grid_dir(demo_root: Path) -> Path:
    """Return the directory holding the grid's catalogue, scores and results."""
    return demo_root / "grid"


def catalogue_path(demo_root: Path) -> Path:
    """Return the file listing every passage version and wording."""
    return grid_dir(demo_root) / "catalogue.json"


def grid_scores_path(demo_root: Path, arm: str) -> Path:
    """Return the file holding one arm's scores, shaped (wordings, texts, formats)."""
    return grid_dir(demo_root) / f"scores_{arm}.npy"


def results_path(demo_root: Path) -> Path:
    """Return the file holding each variant's share of format combinations."""
    return grid_dir(demo_root) / "variants.parquet"


@dataclass(frozen=True)
class Catalogue:
    """Every version of every passage, the wordings, and the case's place.

    Attributes:
        passages: Each passage's id, source, whether it may be absent, and its
            versions, the original first.
        wordings: Each wording's text and whether it carries the stale premise.
        key: The id of the key passage.
        k: The number of passages passed on.
        near_cut: The passages crossed in full.
        case_wording: The position of the case's wording.
        case_state: The case's version of each passage, ABSENT if left out.
    """

    passages: list[dict[str, Any]]
    wordings: list[dict[str, Any]]
    key: str
    k: int
    near_cut: list[str]
    case_wording: int
    case_state: dict[str, int]

    @property
    def ids(self) -> list[str]:
        """Return the passage ids, in knowledge-base order."""
        return [p["id"] for p in self.passages]

    def texts(self) -> list[tuple[str, int]]:
        """Return every (passage, version), in the order the scores hold them."""
        return [(p["id"], v) for p in self.passages for v in range(len(p["versions"]))]

    def levels(self, passage_id: str) -> list[int]:
        """Return the settings one passage can take: its versions, and ABSENT."""
        passage = self.passages[self.ids.index(passage_id)]
        levels = list(range(len(passage["versions"])))
        return [*levels, ABSENT] if passage["optional"] else levels


def load_catalogue(demo_root: Path) -> Catalogue:
    """Read the catalogue."""
    raw = json.loads(catalogue_path(demo_root).read_text())
    return Catalogue(**raw)


def tokens(text: str) -> list[str]:
    """Return the lowercase word tokens BM25 indexes."""
    return re.findall(r"[a-z0-9]+", text.lower())


def neutral(pairs: Sequence[Sequence[str]], body: str) -> str:
    """Return the format-free text the first stage indexes: values, then body."""
    return " ".join(v for _, v in pairs) + " " + body


def present(catalogue: Catalogue, state: Mapping[str, int]) -> list[tuple[str, int]]:
    """Return the (passage, version) pairs a state holds, in knowledge-base order."""
    return [(pid, state[pid]) for pid in catalogue.ids if state[pid] != ABSENT]


def retrieve(
    catalogue: Catalogue, state: Mapping[str, int]
) -> tuple[BM25Okapi, list[tuple[str, int]]]:
    """Index one state with BM25.

    Returns:
        The index and the (passage, version) behind each of its documents.
    """
    docs = present(catalogue, state)
    by_id = {p["id"]: p for p in catalogue.passages}
    index = BM25Okapi([tokens(neutral(**by_id[pid]["versions"][v])) for pid, v in docs])
    return index, docs


def top(index: BM25Okapi, docs: Sequence[tuple[str, int]], wording: str, n: int = 15):
    """Return the n documents BM25 ranks highest for a wording, best first."""
    order = np.argsort(-index.get_scores(tokens(wording)), kind="stable")[:n]
    return [docs[i] for i in order]


def states(catalogue: Catalogue, seed: int = 0) -> Iterator[tuple[str, dict[str, int]]]:
    """Yield every state of the grid, labelled by how the far passages are set.

    The passages near the cut are crossed in full. Each such state is yielded
    with the far passages at the case's setting, and again with one random
    draw of them. The original draft, every passage at its first version and
    every optional passage left out, is yielded last.
    """
    rng = np.random.default_rng(seed)
    far = [pid for pid in catalogue.ids if pid not in catalogue.near_cut]
    for combo in itertools.product(*(catalogue.levels(p) for p in catalogue.near_cut)):
        near = dict(zip(catalogue.near_cut, combo, strict=True))
        yield REFERENCE, {**{p: catalogue.case_state[p] for p in far}, **near}
        drawn = {p: int(rng.choice(catalogue.levels(p))) for p in far}
        yield DRAWN, {**drawn, **near}
    yield (
        ORIGINAL,
        {p["id"]: ABSENT if p["optional"] else 0 for p in catalogue.passages},
    )


def keep_rates(
    scores: np.ndarray, sources: Sequence[str], key: int, k: int
) -> np.ndarray:
    """Return the share of format combinations with the key in the top k.

    Args:
        scores: The pool's scores per variant, shaped (variants, passages, formats).
        sources: The source system of each passage, per variant.
        key: The position of the key passage, per variant.
        k: The number of passages passed on.

    Returns:
        One share per variant.
    """
    every = all_assignments()
    out = np.empty(len(scores))
    for i, (s, src, kk) in enumerate(zip(scores, sources, key, strict=True)):
        out[i] = (rank_of(deployed(s, src, every), kk) <= k).mean()
    return out


def text_index(catalogue: Catalogue) -> dict[tuple[str, int], int]:
    """Return the position of each (passage, version) in the scores."""
    return {t: i for i, t in enumerate(catalogue.texts())}


def source_of(catalogue: Catalogue) -> dict[str, str]:
    """Return each passage's source system."""
    return {p["id"]: p["source"] for p in catalogue.passages}


def load_variants(demo_root: Path) -> pd.DataFrame:
    """Read the grid's variants."""
    return pd.read_parquet(results_path(demo_root))


def crossed(table: pd.DataFrame, values: pd.Series) -> np.ndarray:
    """Return one value per variant laid out as (wordings, states)."""
    frame = pd.DataFrame(
        {"wording": table["wording"], "state": table["state"], "value": values}
    )
    grid = frame.pivot(index="wording", columns="state", values="value")
    if grid.isna().to_numpy().any():
        raise ValueError("the variants do not cross every wording with every state")
    return grid.to_numpy(dtype=float)


def crossed_bootstrap(
    cells: np.ndarray, n: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Return the mean over a crossed grid, with a two-way cluster bootstrap interval.

    Wordings and states are resampled independently, with replacement, and the
    mean is taken over every cell their draws cross.

    Args:
        cells: One value per (wording, state).
        n: The number of resamples.
        seed: The seed of the resamples.
        level: The coverage of the interval.

    Returns:
        The mean, and the interval's lower and upper bounds.
    """
    rng = np.random.default_rng(seed)
    rows, cols = cells.shape
    means = np.empty(n)
    for i in range(n):
        a = np.bincount(rng.integers(rows, size=rows), minlength=rows)
        b = np.bincount(rng.integers(cols, size=cols), minlength=cols)
        means[i] = a @ cells @ b / (rows * cols)
    tail = (1 - level) / 2
    low, high = np.quantile(means, [tail, 1 - tail])
    return float(cells.mean()), float(low), float(high)


def summary(
    table: pd.DataFrame,
    arms: Sequence[str],
    control: str,
    winner: str,
    n: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Return each arm's mean share and the paired comparison, with intervals.

    Args:
        table: The variants to summarise, crossing wordings with states.
        arms: The arms to report the mean share of.
        control: The arm fine-tuned without the invariance term.
        winner: The arm fine-tuned with it.
        n: The number of bootstrap resamples.
        seed: The seed of the resamples.

    Returns:
        One row per statistic: its mean and interval bounds.
    """
    stats = {f"mean share, {arm}": table[arm] for arm in arms}
    stats[f"{winner} minus {control}"] = table[winner] - table[control]
    stats[f"share of variants with {winner} above {control}"] = (
        table[winner] > table[control]
    ).astype(float)
    stats[f"share of variants with {winner} below {control}"] = (
        table[winner] < table[control]
    ).astype(float)
    rows = {
        name: crossed_bootstrap(crossed(table, values), n=n, seed=seed)
        for name, values in stats.items()
    }
    return pd.DataFrame(rows, index=["mean", "low", "high"]).T


def per_wording(
    table: pd.DataFrame, catalogue: Catalogue, arms: Sequence[str]
) -> pd.DataFrame:
    """Return each wording's mean share over its states, per arm.

    Args:
        table: The variants to average.
        catalogue: The catalogue, for the wordings' text.
        arms: The arms to report.

    Returns:
        One row per wording, indexed by its text, with whether it carries the
        stale premise and one mean share per arm.
    """
    means = table.groupby("wording")[list(arms)].mean()
    means.insert(
        0,
        "stale_premise",
        [catalogue.wordings[w]["stale_premise"] for w in means.index],
    )
    means.index = pd.Index(
        [catalogue.wordings[w]["text"] for w in means.index], name="wording"
    )
    return means


def versions_table(catalogue: Catalogue, passage_id: str) -> pd.DataFrame:
    """Return one passage's versions, showing only the fields that differ.

    Where two versions hold the same fields in a different order, the first
    field is shown too. Version 1 is the original.
    """
    versions = catalogue.passages[catalogue.ids.index(passage_id)]["versions"]
    table = pd.DataFrame([dict(v["pairs"]) for v in versions]).fillna("")
    out = table[[c for c in table.columns if table[c].nunique() > 1]].copy()
    firsts = [v["pairs"][0][0] for v in versions]
    if len(set(firsts)) > 1:
        out.insert(0, "first field", firsts)
    out.index = pd.RangeIndex(1, len(out) + 1, name="version")
    return out


def live_wordings(
    table: pd.DataFrame, arms: Sequence[str], threshold: float = 0.5
) -> list[int]:
    """Return the wordings for which some model is in contention often enough.

    A variant is live when at least one arm places the key passage in the top k
    under at least one format combination. A wording is kept when more than the
    threshold share of its states are live.

    Args:
        table: The variants, crossing wordings with states.
        arms: The arms whose shares decide whether a variant is live.
        threshold: The share of live states a wording must exceed.

    Returns:
        The kept wordings, in order.
    """
    live = (table[list(arms)].max(axis=1) > 0).groupby(table["wording"]).mean()
    return [int(w) for w in live.index if live[w] > threshold]


def sample_variants(
    table: pd.DataFrame, per_wording: int, seed: int = 0
) -> pd.DataFrame:
    """Return a random sample of variants, the same number from each wording."""
    return table.groupby("wording", group_keys=False).sample(
        n=per_wording, random_state=seed
    )
