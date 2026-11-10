"""Live progress of the measurement jobs of one run.

Usage:
    uv run python scripts/watch.py
    uv run python scripts/watch.py --pattern 'test_cross_*.progress'
    uv run python scripts/watch.py --pattern '*_jina_v2_*' --pattern '*_mxbai_v1_*'
    uv run python scripts/watch.py --pattern '*' --max-age-hours 12

Only progress files are read, whatever the pattern names, so a pattern may
match a job's logs as well.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from fsr.cli import DEFAULT_DATA_ROOT
from fsr.models.registry import MODELS
from fsr.reporting import PROGRESS_SUFFIX, format_duration, parse_progress

RESULTS_SUBDIR = "h1"
DEFAULT_PATTERN = "*" + PROGRESS_SUFFIX
BAR_WIDTH = 24
BAR_WIDTHS = (BAR_WIDTH, 16, 8)
NO_BAR = 0
MIN_NAME = 8
ELLIPSIS = "..."
GROUP_INDENT = "    "
FLAT_INDENT = "  "
CLEAR_BELOW = "\033[J"
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
        elapsed: The seconds the job had run when it wrote this.
        eta: The seconds the job had left when it wrote this.
        failed: Whether the job reports a failure.
        age: The seconds since the job wrote this.
    """

    name: str
    done: int
    total: int
    label: str
    elapsed: float
    eta: float
    failed: bool
    age: float = 0.0

    @property
    def finished(self) -> bool:
        """Report whether the job completed every unit."""
        return self.total > 0 and self.done >= self.total

    @property
    def live_elapsed(self) -> float:
        """Return the seconds the job has run, counting the time since."""
        return self.elapsed + self.age

    @property
    def live_eta(self) -> float:
        """Return the seconds the job has left, counting the time since."""
        return max(self.eta - self.age, 0.0)


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
        age = max(time.time() - path.stat().st_mtime, 0.0)
    except OSError:
        fields = None
        age = 0.0
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
            age=age,
        )
    except ValueError:
        return JobState(path.stem, 0, 0, "", 0.0, 0.0, False)


def by_start(state: JobState) -> tuple[float, str]:
    """Return the key that orders jobs by when they began.

    Args:
        state: The state of one job.

    Returns:
        The key, breaking a tie by name.
    """
    return (-state.live_elapsed, state.name)


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


def split_name(name: str, slugs: Sequence[str] = ()) -> tuple[str | None, str]:
    """Return the model a job name holds, and the name without it.

    Args:
        name: The job name, such as `train_bge_v2_m3_only_yaml_lam0_s1`.
        slugs: The model slugs to look for. The default is the registry.

    Returns:
        The slug, or None when the name holds none, and the rest of the name,
        such as `train only_yaml_lam0_s1`.
    """
    for slug in sorted(slugs or [m.slug for m in MODELS], key=len, reverse=True):
        token = f"_{slug}"
        at = name.find(token + "_")
        if at < 0 and name.endswith(token):
            at = len(name) - len(token)
        if at < 0:
            continue
        before = name[:at]
        after = name[at + len(token) :].lstrip("_")
        return slug, " ".join(part for part in (before, after) if part)
    return None, name


def elide(text: str, width: int) -> str:
    """Return the text cut to the width, keeping most of its start and its end.

    The start of a job name says what the job is, so it keeps two thirds.
    """
    if len(text) <= width:
        return text
    if width <= len(ELLIPSIS):
        return text[:width]
    keep = width - len(ELLIPSIS)
    head = -(-2 * keep // 3)
    return text[:head] + ELLIPSIS + text[len(text) - (keep - head) :]


def notes(state: JobState, short: bool = False) -> str:
    """Return what a job is doing, in full or with only the time left.

    Args:
        state: The state of one job.
        short: Whether to leave out the label and every time but the time
            left.

    Returns:
        The note.
    """
    if state.failed:
        return "FAILED"
    if state.finished:
        return "done" if short else f"done in {format_duration(state.elapsed)}"
    if not state.total:
        return "waiting"
    if not state.done:
        return (
            "starting" if short else f"starting  {format_duration(state.live_elapsed)}"
        )
    left = f"eta {format_duration(state.live_eta)}" if state.live_eta else "eta overdue"
    if short:
        return left
    return f"{state.label}  {format_duration(state.live_elapsed)}  {left}"


def group(
    states: list[JobState],
) -> list[tuple[str | None, list[tuple[str, JobState]]]]:
    """Gather the jobs by the model each one names.

    Args:
        states: The states, in the order they began.

    Returns:
        The jobs naming no model first, then one group per model, ordered by
        its earliest job. Each job carries the name to show for it.
    """
    groups: dict[str | None, list[tuple[str, JobState]]] = {None: []}
    for state in states:
        slug, rest = split_name(state.name)
        shown = state.name if slug is None else rest
        groups.setdefault(slug, []).append((shown, state))
    return [(slug, jobs) for slug, jobs in groups.items() if jobs]


def _rows(
    entries: list[tuple[str | None, list[tuple[str, JobState]]]],
    bar: int,
    short: bool,
    name_width: int | None,
) -> list[str]:
    """Return the lines of a block, at one bar width, note form and name width."""
    names = [shown for _, jobs in entries for shown, _ in jobs]
    width = max(len(n) for n in names) if name_width is None else name_width
    lines = []
    for slug, jobs in entries:
        indent = FLAT_INDENT
        if slug is not None:
            finished = sum(1 for _, s in jobs if s.finished)
            lines.append(f"{FLAT_INDENT}{slug}  {finished}/{len(jobs)} done")
            indent = GROUP_INDENT
        for shown, state in jobs:
            drawn = f"  {render_bar(state.done, state.total, bar)}" if bar else ""
            lines.append(
                f"{indent}{elide(shown, width):<{width}}{drawn}"
                f" {state.done:>3}/{state.total:<3}  {notes(state, short)}"
            )
    return lines


def render(states: list[JobState], width: int | None = None) -> list[str]:
    """Return the lines for the jobs, grouped by model, and a line for the run.

    With a width, the block gives up detail until every line fits: first the
    bar narrows, then the notes keep only the time left, then the bar goes,
    then the names lose part of their middle, and last each line is cut at
    the edge, so no line wraps.

    Args:
        states: The state of each job, in the order they began.
        width: The columns a line may take, or None for no limit.

    Returns:
        The lines to print.
    """
    if not states:
        return ["no job progress found"[: width or None]]
    entries = group(states)

    def fits(lines: list[str]) -> bool:
        return width is None or all(len(line) <= width for line in lines)

    steps = [
        *((bar, False) for bar in BAR_WIDTHS),
        *((bar, True) for bar in (*BAR_WIDTHS, NO_BAR)),
    ]
    lines = next(
        (
            candidate
            for bar, short in steps
            if fits(candidate := _rows(entries, bar, short, None))
        ),
        None,
    )
    if lines is None:
        widest = _rows(entries, NO_BAR, True, None)
        longest = max(len(shown) for _, jobs in entries for shown, _ in jobs)
        spare = max(len(line) for line in widest) - width
        lines = _rows(entries, NO_BAR, True, max(longest - spare, MIN_NAME))

    done = sum(s.done for s in states)
    total = sum(s.total for s in states)
    running = [s for s in states if s.total and not s.finished and not s.failed]
    eta = max((s.live_eta for s in running), default=0.0)
    lines.append(
        f"{FLAT_INDENT}{len(states)} jobs, {done}/{total} units"
        f"{f', eta {format_duration(eta)}' if running else ''}"
    )
    if width is not None:
        lines = [line[:width] for line in lines]
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
    if redraw:
        # A shorter block, after a job ages out, leaves the old lines below.
        out.write(CLEAR_BELOW)
    out.flush()


def find_states(
    directory: Path,
    patterns: str | Sequence[str],
    max_age: float | None = None,
) -> list[JobState]:
    """Return the state of every job a pattern names, in the order they began.

    Args:
        directory: The directory holding the progress files.
        patterns: The globs, one of which a progress file name must match.
        max_age: The seconds since a job last wrote, past which it is left
            out, or None to keep every job.

    Returns:
        The states. A file two patterns match appears once.
    """
    if isinstance(patterns, str):
        patterns = [patterns]
    paths = {
        path
        for pattern in patterns
        for path in directory.glob(pattern)
        if path.name.endswith(PROGRESS_SUFFIX)
    }
    states = (read_state(path) for path in paths)
    return sorted(
        (s for s in states if max_age is None or s.age <= max_age), key=by_start
    )


def watch(
    directory: Path,
    patterns: str | Sequence[str],
    interval: float,
    once: bool,
    out: TextIO,
    max_age: float | None = None,
) -> int:
    """Draw the progress of every matching job until the viewer stops.

    Args:
        directory: The directory holding the progress files.
        patterns: The globs, one of which a progress file name must match.
        interval: The seconds between reads.
        once: Whether to draw a single block and stop.
        out: Where to write.
        max_age: The seconds since a job last wrote, past which it is left
            out, or None to keep every job.

    Returns:
        The exit code. 1 when a single block holds a failure.
    """
    redraw = 0
    animate = out.isatty() and not once
    while True:
        states = find_states(directory, patterns, max_age)
        width = shutil.get_terminal_size().columns - 1 if animate else None
        lines = render(states, width)
        draw(lines, out, redraw if animate else 0)
        redraw = len(lines)
        if once:
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
        dest="patterns",
        action="append",
        default=None,
        help="Glob the progress file names must match. Give it more than once "
        "to watch every job any of them names",
    )
    ap.add_argument(
        "--max-age-hours",
        type=float,
        default=None,
        help="Leave out a job that has not written for longer than this",
    )
    ap.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    ap.add_argument(
        "--once",
        action="store_true",
        help="Draw one block and stop, exiting non-zero if a job failed",
    )
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
            args.patterns or [DEFAULT_PATTERN],
            args.interval,
            args.once,
            sys.stdout,
            None if args.max_age_hours is None else args.max_age_hours * 3600,
        )
    except KeyboardInterrupt:
        return INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(main())
