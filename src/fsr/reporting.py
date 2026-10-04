"""Progress and summary output shared by the measurement scripts."""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

PROGRESS_SUFFIX = ".progress"
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


def progress_line(
    done: int, total: int, label: str, elapsed: float, failed: bool = False
) -> str:
    """Return the state of a job as one machine-readable line.

    A watcher reads this, so the fields are `key=value` pairs.

    Args:
        done: The units completed.
        total: The units the job will take.
        label: What the job finished last.
        elapsed: The seconds spent so far.
        failed: Whether the job has reported a failure.

    Returns:
        The line, with no trailing newline.
    """
    return (
        f"done={done} total={total} "
        f"elapsed={elapsed:.1f} eta={eta_seconds(done, total, elapsed):.1f} "
        f"failed={int(failed)} label={label}"
    )


class ProgressCounter:
    """Counts the units of one job across every phase of that job.

    A counter given a path keeps the current state there, in one line that is
    replaced at each unit. The job log holds no part of it.
    """

    def __init__(
        self,
        total: int,
        path: Path | None = None,
        started: float | None = None,
        done: int = 0,
    ) -> None:
        """Start a counter and write its opening state.

        Args:
            total: The units the job will complete.
            path: The file to keep the state in. None keeps no file.
            started: The time the job began. The default is the current time.
            done: The units a resumed job has already completed.
        """
        self.total = total
        self.path = path
        self.done = done
        self.failed = False
        self.label = ""
        self.started = time.time() if started is None else started
        self.write()

    def write(self) -> str:
        """Replace the state file, and return the line written.

        Returns:
            The line, whether or not a path was given.
        """
        line = progress_line(
            self.done, self.total, self.label, self.elapsed, self.failed
        )
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(line + "\n", encoding="utf-8")
        return line

    def step(self, label: str) -> str:
        """Count one unit and write the new state.

        Args:
            label: What the job just finished.

        Returns:
            The line written.
        """
        self.done += 1
        self.label = label
        return self.write()

    def fail(self) -> str:
        """Mark the job as failed and write the new state.

        Returns:
            The line written.
        """
        self.failed = True
        return self.write()

    @property
    def elapsed(self) -> float:
        """Return the seconds since the job began."""
        return time.time() - self.started


def parse_progress(line: str) -> dict[str, str] | None:
    """Return the fields of a progress line, or None when there are none.

    Args:
        line: The contents of a progress file.

    Returns:
        The `key=value` fields, or None when the line carries no field.
    """
    fields = {}
    for token in line.strip().split(" "):
        key, separator, value = token.partition("=")
        if separator:
            fields[key] = value
    return fields or None


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
