"""Build the corpus end to end and then check it."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from fsr.corpus.cli import add_data_root_arg
from fsr.corpus.config import DEFAULT

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
) -> list[Stage]:
    """Return the stages to run, in build order.

    Args:
        data_root: The directory holding the corpus files and the manifest.
        limit: The cap on examples scanned. 0 scans every example.
        force: Whether each stage rebuilds outputs that are already present.
        fetch: Whether to write the raw cache first and parse from it.
        cache_k: How many negatives to mine for each query.

    Returns:
        The stages, starting with the raw cache when it is asked for.
    """
    root = ["--data-root", str(data_root)]
    limited = [*root, *(["--limit", str(limit)] if limit else [])]
    rebuild = ["--force"] if force else []

    stages = []
    if fetch:
        stages.append(Stage("fetch", "fetch.py", [*limited, *rebuild]))
        stages.append(
            Stage("parse", "parse.py", [*root, "--source", "cache", *rebuild])
        )
    else:
        stages.append(Stage("parse", "parse.py", [*limited, *rebuild]))
    stages.append(Stage("split", "split.py", [*root, *rebuild]))
    stages.append(
        Stage("negatives", "negatives.py", [*root, "--cache-k", str(cache_k), *rebuild])
    )
    stages.append(Stage("validate", "validate.py", root))
    return stages


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
        SystemExit: If a stage exits non-zero.
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
    ap.add_argument("--cache-k", type=int, default=DEFAULT.cache_k)
    args = ap.parse_args()

    stages = plan(args.data_root, args.limit, args.force, args.fetch, args.cache_k)
    for stage in stages:
        code = run_stage(stage)
        if code != 0:
            raise SystemExit(f"{stage.name} failed with exit code {code}")
    print(f"\nCorpus build complete: {', '.join(s.name for s in stages)}")


if __name__ == "__main__":
    main()
