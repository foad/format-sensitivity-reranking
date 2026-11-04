"""Find the demonstration pools that lose their answer to the format.

Reads the pools and the score tensors of the untrained model, the rank-only
control and the invariance arm. Writes `selection_{model}.json` with the
funnel and every matching pool, and `topk_{model}_{id}.md` with the top-k
text each arm passes on for the chosen pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fsr.cli import add_data_root_arg, resolve_model
from fsr.demo.export import pipeline_views, topk_markdown
from fsr.demo.layout import DEFAULT_DEMO_ROOT, scenarios_path, selection_path, topk_path
from fsr.demo.selection import CONDITIONS, K, select
from fsr.demo.store import load_tensor
from fsr.reporting import heading, report_saved
from fsr.standardisation.results import WINNER, arm_names

DEFAULT_MODEL = "bge_base"


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--model", default=DEFAULT_MODEL, help="Registry slug or identifier"
    )
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument(
        "--k", type=int, default=K, help="Candidates the reranker passes on"
    )
    ap.add_argument(
        "--scenario",
        default=None,
        help="The pool to export. The default is the most preferred match",
    )
    add_data_root_arg(ap)
    return ap


def main() -> None:
    """Select the pools and export the chosen one."""
    args = build_parser().parse_args()
    entry = resolve_model(args.model)
    names = arm_names(args.data_root, entry.slug)
    if WINNER not in names:
        raise SystemExit(
            f"the sweep chose a weight of zero for {entry.slug}, so there is no "
            "invariance arm to contrast with the control."
        )

    pools = json.loads(scenarios_path(args.demo_root).read_text())["scenarios"]
    tensors = {c: load_tensor(args.demo_root, entry.slug, names[c]) for c in CONDITIONS}
    result = select(tensors, pools, args.k)

    heading(f"DEMO SELECTION  {entry.label}  k={args.k}", width=70)
    for condition in CONDITIONS:
        print(
            f"  {condition:<12} {result['arms'][condition]:<14} "
            f"standard {result['standards'][condition]}"
        )
    print()
    for step in result["funnel"]:
        print(f"  {step['n']:>6,}  {step['step']}")
    print(
        f"  {result['n_wrong_in_topk']:>6,}  of which the wrong answer survives the cut"
    )

    out_path = selection_path(args.demo_root, entry.slug)
    out_path.write_text(json.dumps(result, indent=1))
    report_saved(out_path, "selection")

    chosen = args.scenario
    if chosen is None:
        if not result["matches"]:
            print("\nNo pool matched, so there is nothing to export.")
            return
        chosen = result["matches"][0]["id"]
    pool = next((p for p in pools if p["id"] == chosen), None)
    if pool is None:
        raise SystemExit(f"no pool {chosen!r} in {scenarios_path(args.demo_root)}")
    views = pipeline_views(tensors, pools, result["standards"], chosen)
    md_path = topk_path(args.demo_root, entry.slug, chosen)
    md_path.write_text(topk_markdown(pool, views, args.k))
    report_saved(md_path, f"{chosen} top-k")


if __name__ == "__main__":
    main()
