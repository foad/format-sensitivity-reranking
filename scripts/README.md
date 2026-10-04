# scripts/

Pipeline entry points, run via `uv run`

See [`../docs/reproduction.md`](../docs/reproduction.md) for exact commands and outputs.

## `corpus/`
 - `build_corpus.py`: run every stage below in order, then validate. Stops at the first stage that fails.
 - `fetch.py`: stream Natural Questions and cache the raw infobox HTML. Optional.
 - `parse.py`: extract `(k, v)` pairs and body text, apply the quality gate. Reads the stream directly by default.
 - `split.py`: article-level 80/10/10 partition.
 - `negatives.py`: BM25 hard-negative mining.
 - `validate.py`: check the built corpus for issues.

## `runners/`

`local.sh` launches a measurement locally on the device named by `GPU`, falling back to `CUDA_VISIBLE_DEVICES`. Setting `FSR_RUNNER` uses a different runner.

`dispatch.sh` spreads per-model jobs across `GPUS` in waves, skipping a model whose output is present. With several GPUs the screen shows one line per model and each model's detail goes to its own log.

## `h1/`
 - `run.sh`: the full H1 pass.
 - `cross_query.sh`: cross-query sensitivity for one split, one model per GPU.
 - `within_query.sh`: builds the shared candidate cache once, then scores one model per GPU.
 - `cross_query.py`, `within_query.py`: the measurements themselves.

## `h2/`
 - `sweep.sh`: Phase-1 lambda selection, then Phase-2 five-fold hold-one-out, parameterised by model.
 - `train.py`, `eval.py`, `compare.py`, `select_lambda.py`: per-run building blocks.
 - `tanh_ablation.py`: the tanh classifier head test.
 - `*_nonbox*`: capability preservation ablation.
