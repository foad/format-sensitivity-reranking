# scripts/

Pipeline entry points, run via `uv run`

See [`../docs/reproduction.md`](../docs/reproduction.md) for exact commands and outputs.

## `corpus/`
 - `fetch.py`: stream Natural Questions and cache the raw infobox HTML. Optional.
 - `parse.py`: extract `(k, v)` pairs and body text, apply the quality gate. Reads the stream directly by default.
 - `split.py`: article-level 80/10/10 partition.
 - `negatives.py`: BM25 hard-negative mining.

## `h1/`
 - `nq_h1_measurement.py --split train`: score six models across two modes (with_body, metadata_only).
 - `nq_h1_within_query.py`: within-query ranking disturbance over the BM25 candidate lists.
 - `h1_within_query_report.py`: the reported within-query figures.

## `h2/`
 - `sweep.sh`: Phase-1 lambda selection, then Phase-2 five-fold hold-one-out, parameterised by model.
 - `train.py`, `eval.py`, `compare.py`, `select_lambda.py`: per-run building blocks.
 - `tanh_ablation.py`: the tanh classifier head test.
 - `*_nonbox*`: capability preservation ablation.
