"""Collect every passage version and wording the demonstration has, for the grid.

Reads the knowledge base and the variant files under `data/demo/loop/rounds/`,
and writes `data/demo/grid/catalogue.json`: each passage with all of its
versions (the original first), whether it may be left out, every wording of
the case's question, and the setting of each passage in the case.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from fsr.demo.grid import ABSENT, catalogue_path
from fsr.demo.layout import DEFAULT_DEMO_ROOT, inbox_dir
from fsr.reporting import report_saved

KB = Path("data/demo/kb")
NEAR_CUT = (
    "cl-3-4-0",
    "tk-4471",
    "tk-4519",
    "wiki-large-files",
    "ref-v3-uploads",
    "ref-v2-uploads",
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument("--rounds", type=Path, default=Path("data/demo/loop/rounds"))
    ap.add_argument("--case-job", default="round6", help="The job holding the case")
    ap.add_argument("--case-pool", required=True, help="The case's pool id")
    ap.add_argument("--k", type=int, default=3)
    return ap


def apply(passages: list[dict], patch: dict) -> list[dict]:
    """Return the passages with one patch's replacements and additions."""
    by_id = {p["id"]: p for p in passages}
    for pid, new in patch.get("replace", {}).items():
        by_id[pid] = {**by_id[pid], **new}
    return list(by_id.values()) + [copy.deepcopy(p) for p in patch.get("add", [])]


def text(passage: dict) -> str:
    """Return a key identifying one version of a passage."""
    return json.dumps([passage["pairs"], passage["body"]])


def main() -> None:
    """Write the catalogue."""
    args = build_parser().parse_args()
    base = json.loads((KB / "passages.json").read_text())["passages"]
    question = json.loads((KB / "questions.json").read_text())["questions"][0]
    order = [p["id"] for p in base]
    optional: list[str] = []
    versions = {p["id"]: [text(p)] for p in base}
    sources = {p["id"]: p["source"] for p in base}
    known = {p["id"]: p for p in base}

    def record(passages):
        for p in passages:
            if p["id"] not in versions:
                order.append(p["id"])
                optional.append(p["id"])
                versions[p["id"]] = []
                sources[p["id"]] = p["source"]
            if text(p) not in versions[p["id"]]:
                versions[p["id"]].append(text(p))

    specs = sorted(args.rounds.glob("round*.json"), key=lambda p: int(p.stem[5:]))
    for path in specs:
        spec = json.loads(path.read_text())
        start = apply(copy.deepcopy(base), spec.get("global", {}))
        record(start)
        for q in spec["questions"].values():
            for factor in q["factors"].values():
                for patch in factor.values():
                    # Apply to the round's knowledge base so a replacement of an
                    # added passage finds it.
                    record(apply(copy.deepcopy(start), patch))
        known.update({p["id"]: p for p in start})

    wordings: list[str] = []
    case_pool = None
    for path in sorted(inbox_dir(args.demo_root).glob("round*.json")):
        for pool in json.loads(path.read_text())["pools"]:
            if pool["qid"] != question["id"]:
                continue
            if pool["question"] not in wordings:
                wordings.append(pool["question"])
            if path.stem == args.case_job and pool["id"] == args.case_pool:
                case_pool = pool
    if case_pool is None:
        raise SystemExit(f"no pool {args.case_pool} in {args.case_job}")

    # The case's knowledge base: its job's shared setting, with its factor levels.
    spec = json.loads((args.rounds / f"{args.case_job}.json").read_text())
    kb = apply(copy.deepcopy(base), spec.get("global", {}))
    factors = spec["questions"][question["id"]]["factors"]
    levels = case_pool["id"].split("|")[1:-1]
    for (_, factor), level in zip(factors.items(), levels, strict=True):
        kb = apply(kb, factor[level])
    in_case = {p["id"]: versions[p["id"]].index(text(p)) for p in kb}
    case_state = {pid: in_case.get(pid, ABSENT) for pid in order}

    stale = question["stale_answer"].lower()
    catalogue = {
        "passages": [
            {
                "id": pid,
                "source": sources[pid],
                "optional": pid in optional,
                "versions": [
                    dict(zip(("pairs", "body"), json.loads(v), strict=True))
                    for v in versions[pid]
                ],
            }
            for pid in order
        ],
        "wordings": [
            {"text": w, "stale_premise": stale in w.lower() and "still" in w.lower()}
            for w in wordings
        ],
        "key": question["key"],
        "k": args.k,
        "near_cut": list(NEAR_CUT),
        "case_wording": wordings.index(case_pool["question"]),
        "case_state": case_state,
    }
    out = catalogue_path(args.demo_root)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(catalogue, indent=1))
    n_versions = {pid: len(v) for pid, v in versions.items() if len(v) > 1}
    print(f"passages: {len(order)} ({len(optional)} optional); versions: {n_versions}")
    print(f"wordings: {len(wordings)}")
    report_saved(out, "catalogue")


if __name__ == "__main__":
    main()
