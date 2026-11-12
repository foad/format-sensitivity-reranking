"""Hold a block of GPU memory without computing, until stopped or timed out.

A chain of jobs leaves its GPU looking unused between jobs and during light
ones. Holding a fixed block for the chain's lifetime keeps the GPU visibly
claimed to anyone who checks before starting a job of their own.
"""

from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import torch

GIB = 2**30


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gb", type=float, default=3.0, help="Memory to hold, in GiB")
    ap.add_argument(
        "--minutes",
        type=float,
        default=0.0,
        help="Stop after this long. 0 holds until the process is stopped",
    )
    ap.add_argument(
        "--until-gone",
        type=Path,
        default=None,
        help="Stop once this file no longer exists, so a launcher can release it",
    )
    return ap


def should_stop(
    elapsed: float, minutes: float, until_gone: Path | None, stopping: bool
) -> bool:
    """Return whether the hold is over.

    Args:
        elapsed: The seconds since the block was allocated.
        minutes: The time limit, or 0 for none.
        until_gone: A file whose removal ends the hold, or None.
        stopping: Whether a signal asked the process to stop.

    Returns:
        True once any of the three conditions is met.
    """
    if stopping:
        return True
    if minutes and elapsed >= minutes * 60:
        return True
    return until_gone is not None and not until_gone.exists()


def main() -> None:
    """Allocate the block, report it, and sleep until stopped."""
    args = build_parser().parse_args()
    stopping = False

    def stop(_signum, _frame) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    block = torch.empty(int(args.gb * GIB), dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    held = torch.cuda.memory_allocated() / GIB
    print(f"holding {held:.2f} GiB on {torch.cuda.get_device_name()}", flush=True)

    started = time.monotonic()
    while not should_stop(
        time.monotonic() - started, args.minutes, args.until_gone, stopping
    ):
        time.sleep(1)

    del block
    torch.cuda.empty_cache()
    print(f"released after {(time.monotonic() - started) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
