"""A record of one corpus build: what ran, when, and from which revision."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fsr.corpus.files import write_atomic
from fsr.corpus.manifest import package_version

RUN_NAME = "run.json"
GIT_TIMEOUT_SECONDS = 10
SECONDS_PLACES = 3


def now() -> str:
    """Return the current time as an ISO 8601 string in UTC."""
    return datetime.now(UTC).isoformat()


def _git(start: Path, *args: str) -> str | None:
    """Run one git command in a directory and return its output.

    Args:
        start: The directory to run in.
        *args: The git arguments.

    Returns:
        The standard output, or None when git fails or is unavailable.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), *args],
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout if completed.returncode == 0 else None


def git_state(start: Path) -> dict[str, Any] | None:
    """Return the revision of the repository that holds a directory.

    Args:
        start: A directory inside the repository.

    Returns:
        The revision and whether the working tree holds uncommitted changes.
        None when the directory is outside a repository, or git is absent.
    """
    revision = _git(start, "rev-parse", "HEAD")
    status = _git(start, "status", "--porcelain")
    if revision is None or status is None:
        return None
    return {"revision": revision.strip(), "dirty": bool(status.strip())}


@dataclass
class StageRun:
    """One stage that ran, and what it did.

    Attributes:
        name: The stage name.
        exit_code: The code the stage exited with. 0 means success.
        started: When the stage started.
        finished: When the stage finished.
        seconds: How long the stage took.
    """

    name: str
    exit_code: int
    started: str
    finished: str
    seconds: float


@dataclass
class RunRecord:
    """What one build of the corpus ran, and the code it ran from.

    Attributes:
        command: The command line that started the build.
        started: When the build started.
        fsr_version: The installed package version.
        git: The revision the build ran from, when git can report one.
        stages: The stages that ran, in build order.
        finished: When the build finished, or None while it runs.
    """

    command: list[str]
    started: str
    fsr_version: str = field(default_factory=package_version)
    git: dict[str, Any] | None = None
    stages: list[StageRun] = field(default_factory=list)
    finished: str | None = None

    @classmethod
    def begin(cls, command: list[str], repo: Path) -> RunRecord:
        """Start a record for a build that is about to run.

        Args:
            command: The command line that started the build.
            repo: A directory inside the repository holding the build code.

        Returns:
            The record.
        """
        return cls(command=list(command), started=now(), git=git_state(repo))

    def add(self, name: str, exit_code: int, started: str, seconds: float) -> None:
        """Add a stage that has finished.

        Args:
            name: The stage name.
            exit_code: The code the stage exited with.
            started: When the stage started.
            seconds: How long the stage took.
        """
        self.stages.append(
            StageRun(
                name=name,
                exit_code=exit_code,
                started=started,
                finished=now(),
                seconds=round(seconds, SECONDS_PLACES),
            )
        )

    def complete(self) -> None:
        """Mark the build as finished."""
        self.finished = now()

    def to_dict(self) -> dict[str, Any]:
        """Return the record as a JSON-ready mapping."""
        return {
            "command": self.command,
            "started": self.started,
            "finished": self.finished,
            "fsr_version": self.fsr_version,
            "git": self.git,
            "stages": [
                {
                    "name": s.name,
                    "exit_code": s.exit_code,
                    "started": s.started,
                    "finished": s.finished,
                    "seconds": s.seconds,
                }
                for s in self.stages
            ],
        }

    def save(self, path: Path) -> None:
        """Write the record as indented JSON."""
        write_atomic(path, json.dumps(self.to_dict(), indent=2) + "\n")
