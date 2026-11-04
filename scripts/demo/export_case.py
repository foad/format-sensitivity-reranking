"""Export the demonstration case from the grid.

Writes `data/demo/case/`: the case's pool as BM25 retrieves it for one state of
the knowledge base and one wording, what each passage says, the formats of its
source systems and every arm's scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from fsr.demo.grid import (
    grid_scores_path,
    load_catalogue,
    retrieve,
    source_of,
    states,
    text_index,
    top,
)
from fsr.demo.layout import DEFAULT_DEMO_ROOT
from fsr.demo.results import SOURCES, case_dir, scores_path
from fsr.reporting import report_saved

ARMS = ("base", "5fmt_lam0", "5fmt_lam1", "json_lam0", "json_lam1")
KB_QUESTIONS = Path("data/demo/kb/questions.json")
NATIVE = "yaml,json,markdown,inline_kv"
# What each passage of the upload-limit question says the single-request limit is.
CLAIMS = {
    "cl-3-4-0": "200 MB",
    "wiki-large-files": "50 MB",
    "tk-4471": "50 MB",
    "tk-4519": "50 MB",
    "ref-v3-uploads": "50 MB",
    "ref-v2-uploads": "25 MB (v2)",
}


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--state", type=int, required=True, help="The grid state")
    ap.add_argument("--wording", required=True, help="The wording's text")
    ap.add_argument("--seed", type=int, default=0, help="Seed the grid was built with")
    ap.add_argument(
        "--assignment",
        default=NATIVE,
        help="Format per source, in the order " + ",".join(SOURCES),
    )
    return ap


def main() -> None:
    """Write the case."""
    args = build_parser().parse_args()
    catalogue = load_catalogue(args.demo_root)
    setting, state = list(states(catalogue, args.seed))[args.state]
    w = [x["text"] for x in catalogue.wordings].index(args.wording)
    got = top(*retrieve(catalogue, state), args.wording)
    by_id = {p["id"]: p for p in catalogue.passages}
    source = source_of(catalogue)
    where = text_index(catalogue)
    question = next(
        x
        for x in json.loads(KB_QUESTIONS.read_text())["questions"]
        if x["key"] == catalogue.key
    )

    case = {
        "question": args.wording,
        "k": catalogue.k,
        "key": catalogue.key,
        "expected": question["expected_answer"],
        "stale": question["stale_answer"],
        "assignment": dict(zip(SOURCES, args.assignment.split(","), strict=True)),
        "claims": CLAIMS,
        "arms": list(ARMS),
        "grid": {"state": args.state, "setting": setting, "wording": w},
        "candidates": [
            {
                "passage_id": pid,
                "source": source[pid],
                "pairs": by_id[pid]["versions"][v]["pairs"],
                "truncated_body": by_id[pid]["versions"][v]["body"],
            }
            for pid, v in got
        ],
    }
    out = case_dir(args.demo_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "case.json").write_text(json.dumps(case, indent=1))
    for arm in ARMS:
        scores = np.load(grid_scores_path(args.demo_root, arm))[w]
        np.save(
            scores_path(args.demo_root, "case", arm), scores[[where[t] for t in got]]
        )
    for pid in catalogue.near_cut:
        v = state[pid]
        print(f"  {pid}: " + ("absent" if v < 0 else f"version {v + 1}"))
    report_saved(out / "case.json", "case")


if __name__ == "__main__":
    main()
