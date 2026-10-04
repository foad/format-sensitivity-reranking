# scripts/

Pipeline entry points, run via `uv run`

See [`../docs/reproduction.md`](../docs/reproduction.md) for exact commands and outputs.

## Watching a run

`watch.py` draws one progress bar per job from the progress log lines.

```sh
uv run python scripts/watch.py
uv run python scripts/watch.py --pattern 'all_cross_*.log'
```

## `corpus/`
 - `build_corpus.py`: run every stage below in order, then validate. Stops at the first stage that fails.
 - `fetch.py`: stream Natural Questions and cache the raw infobox HTML. Optional.
 - `parse.py`: extract `(k, v)` pairs and body text, apply the quality gate. Reads the stream directly by default.
 - `split.py`: article-level 80/10/10 partition.
 - `negatives.py`: BM25 hard-negative mining.
 - `validate.py`: check the built corpus for issues.

## `runners/`

`local.sh` launches a measurement locally on the device named by `GPU`, falling back to `CUDA_VISIBLE_DEVICES`. Setting `FSR_RUNNER` uses a different runner.

`dispatch.sh` spreads jobs across `GPUS` in waves, skipping a job whose output is present. A job is named by a key the caller chooses: a model slug for the H1 passes, a trained arm for the H2 waves. With several GPUs the screen shows one line per job and each job's detail goes to its own log.

## Shared measurements

`within_query.py` measures the answer axis: how far the ranking inside one query's candidate list moves with the metadata format. H1 runs it on the untrained roster and H2 runs it on each trained adapter.

## `h1/`
 - `run.sh`: the full H1 pass.
 - `cross_query.sh`: the score axis for one split, one model per GPU.
 - `within_query.sh`: the answer axis for one split. Builds the shared candidate cache once, then scores one model per GPU.
 - `cross_query.py`: the score axis measurement.

## `h2/`
 - `train.py`: train one adapter under the ranking and invariance objective.
 - `eval.py`: the score axis and the ranking guardrail for one arm.
 - `select.py`: choose the invariance weight, or the adapter rank.
 - `compare.py`: one trained fold against the untrained baseline.
 - `eval_prose.py`, `compare_prose.py`: the capability check on ordinary prose.
