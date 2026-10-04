"""Step-by-step record of a training run."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from types import TracebackType
from typing import Any

from fsr.reporting import eta_seconds, format_duration

LOG_NAME = "train.jsonl"

RUN = "run"
STEP = "step"
EVAL = "eval"
END = "end"


class TrainLog:
    """An append-only JSON Lines record of one training run."""

    def __init__(self, path: Path) -> None:
        """Open the log for appending, creating the directory if needed.

        Args:
            path: The log file.
        """
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one record and flush it.

        Args:
            event: The record kind, one of RUN, STEP, EVAL or END.
            **fields: The record contents.

        Returns:
            The record as written.
        """
        record = {"event": event, **fields}
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()
        return record

    def close(self) -> None:
        """Close the log."""
        self._handle.close()

    def __enter__(self) -> TrainLog:
        """Return the open log."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the log."""
        self.close()


def read_log(path: Path) -> list[dict[str, Any]]:
    """Return every record of a log, in the order written.

    Args:
        path: The log file.

    Returns:
        The records or an empty list when the file is absent.
    """
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return records


def events(records: Iterable[dict[str, Any]], event: str) -> list[dict[str, Any]]:
    """Return the records of one kind.

    Args:
        records: The records of a log.
        event: The record kind to keep.

    Returns:
        The matching records, in the order given.
    """
    return [r for r in records if r.get("event") == event]


def last_step(records: Iterable[dict[str, Any]]) -> int:
    """Return the highest step any record reached.

    Args:
        records: The records of a log.

    Returns:
        The step, or 0 when no record carries one.
    """
    steps = [r["step"] for r in records if "step" in r]
    return max(steps, default=0)


def rewind(path: Path, step: int) -> int:
    """Drop the records of a log that run past a step.

    Args:
        path: The log file.
        step: The last step to keep.

    Returns:
        The number of records dropped.
    """
    records = read_log(path)
    kept = [r for r in records if r.get("step", 0) <= step]
    dropped = len(records) - len(kept)
    if dropped:
        text = "".join(json.dumps(r, default=str) + "\n" for r in kept)
        path.write_text(text, encoding="utf-8")
    return dropped


def step_line(
    step: int,
    max_steps: int,
    loss: float,
    l_rank: float,
    l_inv: float,
    lr: float,
    elapsed: float,
) -> str:
    """Return the progress line for one step.

    Args:
        step: The steps completed.
        max_steps: The steps the run will take.
        loss: The mean total loss over the accumulated micro-batches.
        l_rank: The mean ranking term.
        l_inv: The mean invariance term.
        lr: The learning rate the scheduler last gave.
        elapsed: The seconds spent so far.

    Returns:
        The line, with the rate and the time left.
    """
    rate = elapsed / step if step > 0 else 0.0
    return (
        f"step {step:>5}/{max_steps}  "
        f"loss={loss:.4f}  rank={l_rank:.4f}  inv={l_inv:.4f}  "
        f"lr={lr:.2e}  {rate:.2f}s/step  "
        f"eta {format_duration(eta_seconds(step, max_steps, elapsed))}"
    )
