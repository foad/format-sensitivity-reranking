"""Assemble the grid: every state and wording, ranked under every format combination.

Reads the catalogue and its scores, retrieves with BM25 for each state and
wording, and writes `data/demo/grid/variants.parquet`: one row per variant with
its wording, its state, whether BM25 retrieved the key passage, and each arm's
share of the 625 format combinations that place the key passage in the top k.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fsr.demo.grid import (
    grid_scores_path,
    keep_rates,
    load_catalogue,
    results_path,
    retrieve,
    source_of,
    states,
    text_index,
    top,
)
from fsr.demo.layout import DEFAULT_DEMO_ROOT
from fsr.reporting import report_saved

ARMS = ("base", "5fmt_lam0", "5fmt_lam1", "json_lam0", "json_lam1")


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    ap.add_argument(
        "--seed", type=int, default=0, help="Seed of the far passages' draw"
    )
    ap.add_argument(
        "--n", type=int, default=15, help="Passages the first stage retrieves"
    )
    return ap


def main() -> None:
    """Write the grid's variants."""
    args = build_parser().parse_args()
    catalogue = load_catalogue(args.demo_root)
    scores = {arm: np.load(grid_scores_path(args.demo_root, arm)) for arm in ARMS}
    where = text_index(catalogue)
    source = source_of(catalogue)
    started = time.time()

    rows, pools, sources, keys = [], [], [], []
    for state_id, (setting, state) in enumerate(states(catalogue, args.seed)):
        index, docs = retrieve(catalogue, state)
        for w, wording in enumerate(catalogue.wordings):
            got = top(index, docs, wording["text"], args.n)
            ids = [pid for pid, _ in got]
            row = {
                "wording": w,
                "stale_premise": wording["stale_premise"],
                "state": state_id,
                "setting": setting,
                "retrieved": catalogue.key in ids,
                **{f"v_{pid}": state[pid] for pid in catalogue.near_cut},
            }
            rows.append(row)
            if row["retrieved"]:
                pools.append((w, [where[t] for t in got]))
                sources.append([source[pid] for pid in ids])
                keys.append(ids.index(catalogue.key))
    table = pd.DataFrame(rows)
    print(
        f"{table['state'].nunique():,} states x {len(catalogue.wordings)} wordings "
        f"= {len(table):,} variants; key retrieved in {table['retrieved'].mean():.1%}"
    )

    for arm in ARMS:
        stack = np.stack([scores[arm][w][idx] for w, idx in pools])
        rates = np.zeros(len(table))
        rates[table["retrieved"].to_numpy()] = keep_rates(
            stack, sources, keys, catalogue.k
        )
        table[arm] = rates
        print(f"  {arm}: done at {time.time() - started:.0f}s", flush=True)

    out = results_path(args.demo_root)
    table.to_parquet(out, index=False)
    report_saved(out, "grid variants")


if __name__ == "__main__":
    main()
