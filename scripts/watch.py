"""Live progress of the measurement jobs of one run.

Usage:
    uv run python scripts/watch.py
    uv run python scripts/watch.py --pattern 'test_cross_*.progress'
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from fsr.cli import DEFAULT_DATA_ROOT
from fsr.reporting import PROGRESS_SUFFIX, format_duration, parse_progress

RESULTS_SUBDIR = "h1"
DEFAULT_PATTERN = "*" + PROGRESS_SUFFIX
BAR_WIDTH = 24
DEFAULT_INTERVAL = 5.0
CURSOR_UP = "\033[{n}A"
INTERRUPTED = 130


@dataclass(frozen=True)
class JobState:
    """What one job's log says about its progress.

    Attributes:
        name: The job name, from the progress file name.
        done: The units the job has completed.
        total: The units the job will complete. 0 before the first unit.
        label: What the job finished last.
        elapsed: The seconds the job has run.
        eta: The seconds the job has left.
        failed: Whether the log reports a failure.
    """

    name: str
    done: int
    total: int
    label: str
    elapsed: float
    eta: float
    failed: bool

    @property
    def finished(self) -> bool:
        """Report whether the job completed every unit."""
        return self.total > 0 and self.done >= self.total


def read_state(path: Path) -> JobState:
    """Return what a progress file says about its job.

    Args:
        path: The progress file.

    Returns:
        The state. A file that is absent, empty or part written reports zero
        units, since a job replaces the whole line at each one.
    """
    try:
        fields = parse_progress(path.read_text(errors="replace"))
    except OSError:
        fields = None
    if fields is None:
        return JobState(path.stem, 0, 0, "", 0.0, 0.0, False)
    try:
        return JobState(
            name=path.stem,
            done=int(fields.get("done", 0)),
            total=int(fields.get("total", 0)),
            label=fields.get("label", ""),
            elapsed=float(fields.get("elapsed", 0.0)),
            eta=float(fields.get("eta", 0.0)),
            failed=fields.get("failed") == "1",
        )
    except ValueError:
        return JobState(path.stem, 0, 0, "", 0.0, 0.0, False)


def render_bar(done: int, total: int, width: int = BAR_WIDTH) -> str:
    """Return a fixed-width bar.

    Args:
        done: The units completed.
        total: The units in total. 0 draws an empty bar.
        width: The bar width, between the brackets.

    Returns:
        The bar, using `#` for the part completed.
    """
    filled = round(width * done / total) if total else 0
    filled = min(max(filled, 0), width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(states: list[JobState]) -> list[str]:
    """Return one line per job, and a line for the run.

    Args:
        states: The state of each job.

    Returns:
        The lines to print.
    """
    if not states:
        return ["no job progress found"]
    width = max(len(s.name) for s in states)
    lines = []
    for state in states:
        if state.failed:
            note = "FAILED"
        elif state.finished:
            note = f"done in {format_duration(state.elapsed)}"
        elif not state.total:
            note = "waiting"
        elif not state.done:
            note = f"starting  {format_duration(state.elapsed)}"
        else:
            note = (
                f"{state.label}  {format_duration(state.elapsed)}"
                f"  eta {format_duration(state.eta)}"
            )
        lines.append(
            f"  {state.name:<{width}}  {render_bar(state.done, state.total)}"
            f" {state.done:>3}/{state.total:<3}  {note}"
        )
    done = sum(s.done for s in states)
    total = sum(s.total for s in states)
    running = [s for s in states if s.total and not s.finished and not s.failed]
    eta = max((s.eta for s in running), default=0.0)
    lines.append(
        f"  {'':<{width}}  {len(states)} jobs, {done}/{total} units"
        f"{f', eta {format_duration(eta)}' if running else ''}"
    )
    return lines


def draw(lines: list[str], out: TextIO, redraw: int) -> None:
    """Print the lines, over the previous block when the output allows it.

    Args:
        lines: The lines to print.
        out: Where to write.
        redraw: The lines of the previous block, or 0 for the first block.
    """
    if redraw:
        out.write(CURSOR_UP.format(n=redraw))
    for line in lines:
        out.write(f"\033[2K{line}\n" if redraw else f"{line}\n")
    out.flush()


def watch(
    directory: Path,
    pattern: str,
    interval: float,
    once: bool,
    out: TextIO,
) -> int:
    """Draw the progress of every matching log until the jobs finish.

    Args:
        directory: The directory holding the progress files.
        pattern: The glob the progress file names must match.
        interval: The seconds between reads.
        once: Whether to draw a single block and stop.
        out: Where to write.

    Returns:
        The exit code. 1 when any job reports a failure.
    """
    redraw = 0
    animate = out.isatty() and not once
    while True:
        states = [read_state(path) for path in sorted(directory.glob(pattern))]
        lines = render(states)
        draw(lines, out, redraw if animate else 0)
        redraw = len(lines)
        settled = bool(states) and all(s.finished or s.failed for s in states)
        if once or settled:
            return 1 if any(s.failed for s in states) else 0
        time.sleep(interval)


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument(
        "--results-subdir",
        default=RESULTS_SUBDIR,
        help="Directory under the corpus root that holds the jobs",
    )
    ap.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help="Glob the progress file names must match",
    )
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument("--once", action="store_true", help="Draw one block and stop")
    return ap


def main() -> int:
    """Watch the logs the arguments name.

    Returns:
        The exit code.
    """
    args = build_parser().parse_args()
    try:
        return watch(
            args.data_root / args.results_subdir,
            args.pattern,
            args.interval,
            args.once,
            sys.stdout,
        )
    except KeyboardInterrupt:
        return INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
