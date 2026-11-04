"""Draw the synthetic pools of the pipeline demonstration.

Writes `scenarios.json` to the demonstration directory. Every later stage
reads it, so the pools and their mixed assignment are fixed once.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from fsr.cli import skip_existing
from fsr.demo.layout import DEFAULT_DEMO_ROOT, scenarios_path
from fsr.demo.scenarios import N_CANDIDATES, build_scenarios
from fsr.reporting import heading, report_saved

DEFAULT_N = 2000
DEFAULT_SEED = 0


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="Pools to draw")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--force", action="store_true", help="Draw again when present")
    return ap


def main() -> None:
    """Draw the pools and write them."""
    args = build_parser().parse_args()
    out_path = scenarios_path(args.demo_root)
    if skip_existing("scenario build", [out_path], args.force):
        return

    pools = build_scenarios(args.n, args.seed)
    heading("DEMO SCENARIOS", width=70)
    print(f"  pools:      {len(pools):,}")
    print(f"  candidates: {N_CANDIDATES} each")
    print(f"  seed:       {args.seed}")
    for (kind, target), n in sorted(
        Counter((p["kind"], p["target"]) for p in pools).items()
    ):
        print(f"    {kind:<10} {target:<16} {n:,}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"seed": args.seed, "n": len(pools), "scenarios": pools}, indent=1)
    )
    report_saved(out_path, "scenario")


if __name__ == "__main__":
    main()
