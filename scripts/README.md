# scripts/

Pipeline entry points, run via `uv run`

See [`../docs/reproduction.md`](../docs/reproduction.md) for exact commands and outputs.

## `h1/`
 -  `build_nq_cache.py`: stream Natural Questions, keep infobox records, cache raw HTML.
 -  `parse_nq_cache.py`: extract `(k,v)` pairs and body text, apply the quality gate.
 -  `nq_h1_measurement.py --split train`: score 6 models across 2 modes (with_body, metadata_only).

## `h2/`
 - `build_splits.py`: article-level 80/10/10 splits.
 - `build_negatives.py`: BM25 hard-negative mining (K=15).
 - `sweep_{minilm,mxbai,jina}.sh`: Phase-1 lambda selection, then Phase-2 five-fold hold-one-out.
 - `train.py`, `eval.py`, `compare.py`, `select_lambda.py`: per-run building blocks.
 - `tanh_ablation.py`, mxbai tanh symmetric test.
 - `*_nonbox*`, capability preservation ablation test.
