"""Progress and summary output shared by the measurement scripts."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

PROGRESS_TAG = "[progress]"
NAME_WIDTH = 48
WIDE_RULE = 100


def stamp(when: float | None = None) -> str:
    """Return a wall-clock time a reader can match against other logs.

    Args:
        when: Seconds since the epoch. The default is the current time.

    Returns:
        The local time as `YYYY-MM-DD HH:MM:SS`.
    """
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when))


def format_duration(seconds: float) -> str:
    """Return a duration as hours, minutes and seconds.

    Args:
        seconds: The duration.

    Returns:
        The duration as `h:mm:ss`.
    """
    whole = int(seconds)
    return f"{whole // 3600}:{whole // 60 % 60:02d}:{whole % 60:02d}"


def eta_seconds(done: int, total: int, elapsed: float) -> float:
    """Return the seconds left at the rate reached so far.

    Args:
        done: The units completed.
        total: The units the run will take.
        elapsed: The seconds spent so far.

    Returns:
        The estimate. The result is 0.0 before the first unit completes.
    """
    if done <= 0:
        return 0.0
    return max(total - done, 0) * elapsed / done


def shorten(name: str, width: int = NAME_WIDTH) -> str:
    """Return a name cut to a column width, keeping the end.

    Args:
        name: The name to fit.
        width: The column width.

    Returns:
        The name, marked with a leading `...` when it was cut.
    """
    return name if len(name) <= width else "..." + name[-(width - 3) :]


def heading(title: str, width: int = WIDE_RULE) -> None:
    """Print a titled rule, with the time it was reached.

    Args:
        title: The line between the rules.
        width: The rule width.
    """
    rule = "=" * width
    print(f"\n{rule}")
    print(f"{title}   {stamp()}")
    print(rule, flush=True)


def model_heading(name: str, width: int = 70) -> None:
    """Print the banner that opens one model's output.

    Args:
        name: The model identifier.
        width: The rule width.
    """
    rule = "-" * width
    print(f"\n{rule}\n{name}   {stamp()}\n{rule}", flush=True)


def progress(done: int, total: int, label: str, elapsed: float) -> str:
    """Print and return one machine-readable progress line.

    Args:
        done: The units completed.
        total: The units the job will take.
        label: What the job just finished.
        elapsed: The seconds spent so far.

    Returns:
        The line, as printed.
    """
    line = (
        f"{PROGRESS_TAG} done={done} total={total} "
        f"elapsed={elapsed:.1f} eta={eta_seconds(done, total, elapsed):.1f} "
        f"label={label}"
    )
    print(line, flush=True)
    return line


class ProgressCounter:
    """Counts the units of one job as its phases complete."""

    def __init__(self, total: int, started: float | None = None) -> None:
        """Start a counter.

        Args:
            total: The units the job will complete.
            started: The time the job began. The default is the current time.
        """
        self.total = total
        self.done = 0
        self.started = time.time() if started is None else started

    def step(self, label: str) -> str:
        """Count one unit and report it.

        Args:
            label: What the job just finished.

        Returns:
            The progress line, as printed.
        """
        self.done += 1
        return progress(self.done, self.total, label, time.time() - self.started)

    @property
    def elapsed(self) -> float:
        """Return the seconds since the job began."""
        return time.time() - self.started


def parse_progress(line: str) -> dict[str, str] | None:
    """Return the fields of a progress line, or None for any other line.

    Args:
        line: A line of a job log.

    Returns:
        The `key=value` fields, or None.
    """
    if not line.startswith(PROGRESS_TAG):
        return None
    fields = {}
    for token in line[len(PROGRESS_TAG) :].strip().split(" "):
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    return fields


def table(columns: Sequence[tuple[str, int]], rows: Iterable[Sequence[str]]) -> None:
    """Print a header, a rule, and one line per row.

    Args:
        columns: The label and the width of each column.
        rows: The already formatted cells of each row.
    """

    def render(cells: Sequence[str]) -> str:
        out = []
        for index, (cell, (_, width)) in enumerate(zip(cells, columns, strict=True)):
            if width == 0:
                out.append(cell)
            elif index == 0:
                out.append(f"{cell:<{width}}")
            else:
                out.append(f"{cell:>{width}}")
        return " ".join(out).rstrip()

    header = render([label for label, _ in columns])
    print(header)
    print("-" * max(len(header), WIDE_RULE))
    for row in rows:
        print(render(row))


def failures_block(failures: Sequence[Mapping[str, str]]) -> None:
    """Print the models that did not complete, if any.

    Args:
        failures: One mapping per failure, with `model`, `stage` and `error`.
    """
    if not failures:
        return
    print(f"\nFAILURES ({len(failures)})")
    for failure in failures:
        print(
            f"  {failure['model']:<{NAME_WIDTH}} "
            f"[{failure['stage']}] {failure['error']}"
        )


def report_saved(path: Path, label: str = "") -> None:
    """Print the file written and its size.

    Args:
        path: The file that was written.
        label: What the file holds, for a script that writes several.
    """
    what = f"{label} results" if label else "results"
    print(f"\nSaved {what} -> {path}  ({path.stat().st_size / 1e6:.1f} MB)", flush=True)


def report_elapsed(what: str, started: float) -> float:
    """Print how long a phase took, and return it.

    Args:
        what: The phase that finished.
        started: The time the phase began, from `time.time`.

    Returns:
        The seconds the phase took.
    """
    elapsed = time.time() - started
    print(f"  {what} in {format_duration(elapsed)}", flush=True)
    return elapsed
