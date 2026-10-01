"""Consistency checks over a built corpus.

Each check returns the problems it found. An empty list means the check passed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations
from pathlib import Path

from fsr.corpus.files import digest_file
from fsr.corpus.manifest import Manifest

PARTITIONED = ("train", "dev", "test")
SAMPLE_SIZE = 3


def _sample(values: Sequence[str]) -> str:
    """Return the first few values as a readable list."""
    shown = ", ".join(sorted(values)[:SAMPLE_SIZE])
    extra = len(values) - SAMPLE_SIZE
    return f"{shown}, and {extra} more" if extra > 0 else shown


def check_quality_gate(name: str, records: Sequence[Mapping]) -> list[str]:
    """Report records that carry a quality flag.

    Args:
        name: The split name, for the message.
        records: The records of that split.

    Returns:
        The problems found.
    """
    flagged = [r["id"] for r in records if r["quality_flags"]]
    if not flagged:
        return []
    return [f"{name}: {len(flagged)} records carry a quality flag ({_sample(flagged)})"]


def check_titles_disjoint(
    splits: Mapping[str, Sequence[Mapping]],
    names: Sequence[str] = PARTITIONED,
) -> list[str]:
    """Report article titles that two splits share.

    Args:
        splits: The records of each split.
        names: The splits that must not share a title.

    Returns:
        The problems found.
    """
    titles = {name: {r["title"] for r in splits[name]} for name in names}
    problems = []
    for left, right in combinations(names, 2):
        shared = titles[left] & titles[right]
        if shared:
            problems.append(
                f"{left} and {right} share {len(shared)} titles ({_sample(shared)})"
            )
    return problems


def check_ids_unique(splits: Mapping[str, Sequence[Mapping]]) -> list[str]:
    """Report record ids that appear in more than one split.

    Args:
        splits: The records of each split.

    Returns:
        The problems found.
    """
    seen: dict[str, str] = {}
    repeated: list[str] = []
    for name, records in splits.items():
        for record in records:
            if record["id"] in seen:
                repeated.append(f"{record['id']} ({seen[record['id']]} and {name})")
            else:
                seen[record["id"]] = name
    if not repeated:
        return []
    return [f"{len(repeated)} record ids appear twice ({_sample(repeated)})"]


def check_meta(meta: Mapping, splits: Mapping[str, Sequence[Mapping]]) -> list[str]:
    """Report split counts that disagree with the meta file.

    Args:
        meta: The contents of the meta file.
        splits: The records of each split.

    Returns:
        The problems found.
    """
    expected = {name: f"n_records_{name}" for name in PARTITIONED}
    expected["nq_val"] = "nq_val_n_records"
    problems = []
    for name, key in expected.items():
        if key in meta and meta[key] != len(splits[name]):
            problems.append(
                f"meta {key} is {meta[key]}, but {name} holds {len(splits[name])}"
            )
    return problems


def check_negatives(
    payload: Mapping,
    queries: Sequence[Mapping],
    corpus_records: Sequence[Mapping],
) -> list[str]:
    """Report negatives that are missing, repeated, unknown, or self-matching.

    A negative must be a corpus record, and must appear once per query. It must
    come from an article other than the query's own.

    Args:
        payload: The contents of the negatives file.
        queries: The records the negatives were mined for.
        corpus_records: The records the negatives were mined from.

    Returns:
        The problems found.
    """
    negatives = payload["negatives"]
    cache_k = payload["cache_k"]
    title_of = {r["id"]: r["title"] for r in corpus_records}

    missing = [q["id"] for q in queries if q["id"] not in negatives]
    short = [q for q, picked in negatives.items() if len(picked) != cache_k]
    repeated = [q for q, picked in negatives.items() if len(set(picked)) != len(picked)]
    unknown = [
        q for q, picked in negatives.items() if any(p not in title_of for p in picked)
    ]
    same_title = [
        q["id"]
        for q in queries
        if any(title_of.get(p) == q["title"] for p in negatives.get(q["id"], []))
    ]

    problems = []
    for found, message in (
        (missing, "queries have no negatives"),
        (short, f"queries do not hold {cache_k} negatives"),
        (repeated, "queries repeat a negative"),
        (unknown, "queries cite a negative outside the corpus"),
        (same_title, "queries take a negative from their own article"),
    ):
        if found:
            problems.append(f"{len(found)} {message} ({_sample(found)})")
    return problems


def check_manifest_digests(manifest: Manifest, data_root: Path) -> list[str]:
    """Report manifest outputs that are missing or changed since the build.

    Args:
        manifest: The build manifest.
        data_root: The directory the output paths are relative to.

    Returns:
        The problems found.
    """
    problems = []
    for stage in manifest.stages:
        for output in stage.outputs:
            path = data_root / output.path
            if not path.exists():
                problems.append(f"{stage.name}: {output.path} is missing")
            elif digest_file(path) != output.sha256:
                problems.append(f"{stage.name}: {output.path} changed after the build")
    return problems
