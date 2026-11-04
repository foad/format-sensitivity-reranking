"""The iteration loop: score batches of knowledge-base variants as they arrive.

The variants are written by hand between rounds and pushed to an inbox. A
worker on a GPU scores each job file under every arm and format, and writes
the scores to an outbox, where they are pulled and swept offline.

The worker runs in a container that writes as root, and its files are handed
back only when it exits. So it never creates anything in the inbox, which is
pushed to between rounds. A job is read only once its ready marker exists,
since a round arrives as several files. A claim is made by creating a file in
the outbox exclusively, so several workers never score the same job. Scores
are cached per arm by the question and the rendered passage, so an edit
rescores only the pairs it changes, and a restarted worker repeats no work.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fsr.formats import FORMAT_NAMES, FORMATS

JOB_SUFFIX = ".json"
READY_SUFFIX = ".ready"
CLAIM_SUFFIX = ".claim"
DONE_SUFFIX = ".done"
FAILED_SUFFIX = ".failed"
SCORES_SUFFIX = ".npz"
CACHE_SUBDIR = "cache"
DEFAULT_POLL = 5.0
DEFAULT_IDLE = 600.0
DEFAULT_WALL = 6 * 3600.0

Scorer = Callable[[str, Sequence[tuple[str, str]]], list[float]]


def ready_jobs(inbox: Path) -> list[str]:
    """Return the jobs whose file and ready marker are both present, in order.

    Args:
        inbox: The directory the jobs are pushed to.

    Returns:
        The job names.
    """
    if not inbox.is_dir():
        return []
    return sorted(
        path.stem
        for path in inbox.glob("*" + READY_SUFFIX)
        if (inbox / (path.stem + JOB_SUFFIX)).is_file()
    )


def finished(outbox: Path, name: str) -> bool:
    """Return whether a job was completed or failed."""
    return any(
        (outbox / (name + suffix)).exists() for suffix in (DONE_SUFFIX, FAILED_SUFFIX)
    )


def claim_owner(outbox: Path, name: str) -> str | None:
    """Return the worker that claimed a job, or None when it is unclaimed."""
    path = outbox / (name + CLAIM_SUFFIX)
    if not path.exists():
        return None
    return json.loads(path.read_text())["worker"]


def claim(outbox: Path, name: str, worker: str) -> bool:
    """Claim a job for one worker, unless another holds it.

    Args:
        outbox: The directory the worker writes to.
        name: The job.
        worker: The claiming worker.

    Returns:
        Whether the claim was made.
    """
    try:
        fd = os.open(
            outbox / (name + CLAIM_SUFFIX), os.O_CREAT | os.O_EXCL | os.O_WRONLY
        )
    except FileExistsError:
        return False
    with os.fdopen(fd, "w") as handle:
        json.dump({"worker": worker, "claimed": time.time()}, handle)
    return True


def next_job(inbox: Path, outbox: Path, worker: str) -> str | None:
    """Return the job a worker should score next, claiming it if need be.

    A worker first resumes a job it claimed and did not finish, which is the
    case after the runner restarts it. Otherwise it claims the first open job.

    Args:
        inbox: The directory the jobs are pushed to.
        outbox: The directory the worker writes to.
        worker: The worker.

    Returns:
        The job name, or None when there is nothing to score.
    """
    pending = [name for name in ready_jobs(inbox) if not finished(outbox, name)]
    for name in pending:
        if claim_owner(outbox, name) == worker:
            return name
    for name in pending:
        if claim(outbox, name, worker):
            return name
    return None


def pair_key(question: str, text: str) -> str:
    """Return the cache key of one query and rendered passage pair."""
    return hashlib.sha1(f"{question}\x1f{text}".encode()).hexdigest()


class ScoreCache:
    """The scores of every pair scored so far, per arm.

    Each worker appends to its own file per arm, so no two processes write to
    one file. Every worker reads all of them.
    """

    def __init__(self, outbox: Path, worker: str) -> None:
        """Open the cache in an outbox.

        Args:
            outbox: The directory the worker writes to.
            worker: The worker, which names the files it appends to.
        """
        self.root = outbox / CACHE_SUBDIR
        self.worker = worker
        self.scores: dict[str, dict[str, float]] = {}

    def load(self, arm: str) -> dict[str, float]:
        """Read every worker's scores for one arm, and return them."""
        found: dict[str, float] = {}
        for path in sorted(self.root.glob(f"{arm}.*.jsonl")):
            for line in path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    found[row["k"]] = row["s"]
        self.scores[arm] = found
        return found

    def add(self, arm: str, scored: Mapping[str, float]) -> None:
        """Record new scores for one arm, in memory and in this worker's file."""
        if not scored:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        with (self.root / f"{arm}.{self.worker}.jsonl").open("a") as handle:
            for key, score in scored.items():
                handle.write(json.dumps({"k": key, "s": score}) + "\n")
        self.scores.setdefault(arm, {}).update(scored)


def check_pools(pools: Any) -> None:
    """Raise ValueError unless a job holds pools in the shape the scorer reads."""
    if not isinstance(pools, list) or not pools:
        raise ValueError("the job holds no pools")
    for pool in pools:
        for field in ("id", "question", "candidates"):
            if field not in pool:
                raise ValueError(f"a pool lacks {field!r}")
        if not pool["candidates"]:
            raise ValueError(f"pool {pool['id']} has no candidates")
        for candidate in pool["candidates"]:
            if "pairs" not in candidate or "truncated_body" not in candidate:
                raise ValueError(f"a candidate of pool {pool['id']} lacks text")


def render_pairs(
    pools: Sequence[Mapping[str, Any]],
) -> tuple[list[tuple[str, str]], np.ndarray]:
    """Render every candidate of every pool under every format.

    Args:
        pools: The pools of one job.

    Returns:
        The distinct query and passage pairs, and the index of the pair behind
        each (pool, candidate, format), shaped by the largest pool and -1
        where a smaller pool has no candidate.
    """
    width = max(len(pool["candidates"]) for pool in pools)
    index = np.full((len(pools), width, len(FORMAT_NAMES)), -1, dtype=np.int64)
    pairs: list[tuple[str, str]] = []
    seen: dict[tuple[str, str], int] = {}
    for q, pool in enumerate(pools):
        for c, candidate in enumerate(pool["candidates"]):
            for f, name in enumerate(FORMAT_NAMES):
                pair = (
                    pool["question"],
                    FORMATS[name](candidate["pairs"], candidate["truncated_body"]),
                )
                if pair not in seen:
                    seen[pair] = len(pairs)
                    pairs.append(pair)
                index[q, c, f] = seen[pair]
    return pairs, index


def score_pools(
    pools: Sequence[Mapping[str, Any]],
    arms: Sequence[str],
    scorer: Scorer,
    cache: ScoreCache,
    on_arm: Callable[[str, int], None] | None = None,
) -> tuple[dict[str, np.ndarray], int]:
    """Score one job's pools under every arm, scoring only pairs not cached.

    Args:
        pools: The pools of one job.
        arms: The arms to score with.
        scorer: Scores pairs under one arm, given its name.
        cache: The scores so far, updated with the new ones.
        on_arm: Called with each arm and the pairs it scored, once it is done.

    Returns:
        The scores per arm, shaped (pools, candidates, formats) with NaN where
        a pool has no candidate, and the number of pairs newly scored.
    """
    pairs, index = render_pairs(pools)
    keys = [pair_key(*pair) for pair in pairs]
    out: dict[str, np.ndarray] = {}
    scored = 0
    for arm in arms:
        known = cache.load(arm)
        missing = [i for i, key in enumerate(keys) if key not in known]
        if missing:
            new = scorer(arm, [pairs[i] for i in missing])
            cache.add(
                arm, {keys[i]: float(s) for i, s in zip(missing, new, strict=True)}
            )
            known = cache.scores[arm]
        flat = np.array([known[key] for key in keys], dtype=np.float32)
        scores = np.full(index.shape, np.nan, dtype=np.float32)
        scores[index >= 0] = flat[index[index >= 0]]
        out[arm] = scores
        scored += len(missing)
        if on_arm is not None:
            on_arm(arm, len(missing))
    return out, scored


def write_atomic(path: Path, write: Callable[[Any], None], mode: str = "w") -> None:
    """Write a file under a hidden name and rename it, so no reader sees part."""
    tmp = path.with_name(f".{path.name}.tmp")
    with tmp.open(mode) as handle:
        write(handle)
    os.replace(tmp, path)


def write_result(
    outbox: Path,
    name: str,
    pools: Sequence[Mapping[str, Any]],
    scores: Mapping[str, np.ndarray],
    meta: Mapping[str, Any],
) -> None:
    """Write a job's scores and description, then its done marker.

    Args:
        outbox: The directory the worker writes to.
        name: The job.
        pools: The pools of the job, for their identifiers.
        scores: The scores per arm.
        meta: More fields for the description.
    """
    arrays = {f"scores_{arm}": value for arm, value in scores.items()}
    write_atomic(
        outbox / (name + SCORES_SUFFIX),
        lambda handle: np.savez(handle, **arrays),
        mode="wb",
    )
    description = {
        "job": name,
        "pool_ids": [str(pool["id"]) for pool in pools],
        "arms": list(scores),
        "formats": list(FORMAT_NAMES),
        **meta,
    }
    write_atomic(
        outbox / (name + JOB_SUFFIX),
        lambda handle: json.dump(description, handle, indent=1),
    )
    (outbox / (name + DONE_SUFFIX)).write_text(f"{time.time()}\n")


def write_failure(outbox: Path, name: str, worker: str, error: str) -> None:
    """Mark a job as failed, with the reason, so no worker takes it again."""
    (outbox / (name + FAILED_SUFFIX)).write_text(
        json.dumps({"worker": worker, "error": error, "at": time.time()}) + "\n"
    )


def run_job(
    inbox: Path,
    outbox: Path,
    name: str,
    worker: str,
    arms: Sequence[str],
    scorer: Scorer,
    cache: ScoreCache,
    too_long: Callable[[str, str], bool] | None = None,
    on_arm: Callable[[str, int], None] | None = None,
    on_start: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Score one claimed job and write its result, or mark it failed.

    A job that cannot be read or scored is marked failed rather than stopping
    the worker, since the next job may be sound.

    Args:
        inbox: The directory the jobs are pushed to.
        outbox: The directory the worker writes to.
        name: The job.
        worker: The worker.
        arms: The arms to score with.
        scorer: Scores pairs under one arm, given its name.
        cache: The scores so far.
        too_long: Whether the scorer would truncate a pair. None checks nothing.
        on_arm: Called with each arm and the pairs it scored.
        on_start: Called with the job's name before it is read.

    Returns:
        A summary: the job, whether it failed, and what was scored.
    """
    started = time.time()
    if on_start is not None:
        on_start(name)
    try:
        pools = json.loads((inbox / (name + JOB_SUFFIX)).read_text())["pools"]
        check_pools(pools)
        pairs, _ = render_pairs(pools)
        truncated = (
            sum(too_long(*pair) for pair in pairs) if too_long is not None else 0
        )
        scores, scored = score_pools(pools, arms, scorer, cache, on_arm)
    # Recorded rather than raised, so the worker goes on to the next job.
    except Exception as error:
        write_failure(outbox, name, worker, f"{type(error).__name__}: {error}")
        return {"job": name, "failed": True, "error": str(error)}
    meta = {
        "worker": worker,
        "pairs": len(pairs),
        "scored": scored,
        "truncated_pairs": truncated,
        "seconds": round(time.time() - started, 2),
    }
    write_result(outbox, name, pools, scores, meta)
    return {"job": name, "failed": False, **meta}


def serve(
    inbox: Path,
    outbox: Path,
    worker: str,
    arms: Sequence[str],
    scorer: Scorer,
    too_long: Callable[[str, str], bool] | None = None,
    poll: float = DEFAULT_POLL,
    idle: float = DEFAULT_IDLE,
    wall: float = DEFAULT_WALL,
    on_job: Callable[[dict[str, Any]], None] | None = None,
    on_arm: Callable[[str, int], None] | None = None,
    on_start: Callable[[str], None] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Score jobs as they arrive, until idle for too long or out of time.

    Args:
        inbox: The directory the jobs are pushed to.
        outbox: The directory the worker writes to.
        worker: The worker.
        arms: The arms to score with.
        scorer: Scores pairs under one arm, given its name.
        too_long: Whether the scorer would truncate a pair.
        poll: Seconds between looks at the inbox.
        idle: Seconds without a job after which the worker stops.
        wall: Seconds after which the worker stops between jobs.
        on_job: Called with each job's summary.
        on_arm: Called with each arm and the pairs it scored.
        on_start: Called with each job's name as it is taken.
        clock: The time source, in seconds.
        sleep: Waits the given seconds.

    Returns:
        Why the worker stopped: "idle" or "wall".
    """
    outbox.mkdir(parents=True, exist_ok=True)
    cache = ScoreCache(outbox, worker)
    started = last_job = clock()
    while True:
        if clock() - started >= wall:
            return "wall"
        name = next_job(inbox, outbox, worker)
        if name is None:
            if clock() - last_job >= idle:
                return "idle"
            sleep(poll)
            continue
        summary = run_job(
            inbox, outbox, name, worker, arms, scorer, cache, too_long, on_arm, on_start
        )
        if on_job is not None:
            on_job(summary)
        last_job = clock()
