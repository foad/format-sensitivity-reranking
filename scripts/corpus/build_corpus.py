"""Build the corpus end to end and then check it."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from fsr.cli import add_data_root_arg
from fsr.corpus.config import DEFAULT
from fsr.corpus.run_record import RUN_NAME, RunRecord, now

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Stage:
    """One stage of the build.

    Attributes:
        name: The stage name, used in the progress and failure messages.
        script: The file name of the script that runs it.
        args: The command-line arguments the stage is given.
    """

    name: str
    script: str
    args: list[str] = field(default_factory=list)


def plan(
    data_root: Path,
    limit: int = 0,
    force: bool = False,
    fetch: bool = False,
    cache_k: int = DEFAULT.cache_k,
    prose: bool = False,
) -> list[Stage]:
    """Return the stages to run, in build order.

    Args:
        data_root: The directory holding the corpus files and the manifest.
        limit: The cap on examples scanned. 0 scans every example.
        force: Whether each derived stage rebuilds outputs that are already present.
        fetch: Whether to write the raw cache first and parse from it.
        cache_k: How many negatives to mine for each query.
        prose: Whether to collect the prose-answered subset as well.

    Returns:
        The stages, starting with the raw cache when it is asked for.
    """
    root = ["--data-root", str(data_root)]
    limited = [*root, *(["--limit", str(limit)] if limit else [])]
    rebuild = ["--force"] if force else []

    stages = []
    if fetch:
        stages.append(Stage("fetch", "fetch.py", limited))
        stages.append(
            Stage("parse", "parse.py", [*root, "--source", "cache", *rebuild])
        )
    else:
        stages.append(Stage("parse", "parse.py", [*limited, *rebuild]))
    stages.append(Stage("split", "split.py", [*root, *rebuild]))
    stages.append(
        Stage("negatives", "negatives.py", [*root, "--cache-k", str(cache_k), *rebuild])
    )
    if prose:
        stages.append(Stage("prose", "prose.py", [*limited, *rebuild]))
    stages.append(Stage("validate", "validate.py", root))
    return stages


def require_clean(git: dict | None) -> None:
    """Stop the build unless the code comes from a clean working tree.

    Args:
        git: The repository state, or None when it could not be read.

    Raises:
        SystemExit: If the tree holds uncommitted changes, or if the state is
            unknown.
    """
    if git is None:
        raise SystemExit("--require-clean: the code revision could not be read")
    if git["dirty"]:
        raise SystemExit(f"--require-clean: {git['revision']} has uncommitted changes")


def run_stage(stage: Stage) -> int:
    """Run one stage in its own process and return its exit code.

    Args:
        stage: The stage to run.

    Returns:
        The exit code of the stage.
    """
    print(f"\n{'=' * 60}\n=== {stage.name}\n{'=' * 60}", flush=True)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / stage.script), *stage.args], check=False
    )
    return completed.returncode


def main() -> None:
    """Run every stage and stop at the first failure.

    Raises:
        SystemExit: If a stage exits non-zero, or if --require-clean is given
            and the working tree is not clean.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    add_data_root_arg(ap)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Cap on examples scanned, for a small build (0 = no cap)",
    )
    ap.add_argument(
        "--force", action="store_true", help="Rebuild outputs that already exist"
    )
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="Write the raw cache first, and parse from it instead of streaming",
    )
    ap.add_argument(
        "--prose",
        action="store_true",
        help="Collect the prose-answered subset, which the capability check reads",
    )
    ap.add_argument("--cache-k", type=int, default=DEFAULT.cache_k)
    ap.add_argument(
        "--require-clean",
        action="store_true",
        help="Refuse to build unless the working tree holds no uncommitted changes",
    )
    args = ap.parse_args()

    stages = plan(
        args.data_root, args.limit, args.force, args.fetch, args.cache_k, args.prose
    )
    record = RunRecord.begin(sys.argv, SCRIPT_DIR)
    if args.require_clean:
        require_clean(record.git)
    run_path = args.data_root / RUN_NAME

    for stage in stages:
        started, clock = now(), time.monotonic()
        code = run_stage(stage)
        record.add(stage.name, code, started, time.monotonic() - clock)
        record.save(run_path)
        if code != 0:
            raise SystemExit(f"{stage.name} failed with exit code {code}")

    record.complete()
    record.save(run_path)
    print(f"\nCorpus build complete: {', '.join(s.name for s in stages)}")
    print(f"Wrote {run_path}")


if __name__ == "__main__":
    main()
