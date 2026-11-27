# Reproduction

Commands for each pipeline, run from the repository root.

```bash
uv sync
```

Build the image:

```bash
docker build -t format-sensitivity-reranking .
docker run --rm --gpus device=0 -v "$PWD:/work" format-sensitivity-reranking \
    python scripts/corpus/build_corpus.py
```

Built on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`.

## Corpus

Every measurement reads the corpus, so build it first, or download it
from the release and unpack into `data/`.

To build it, check the pipeline first over a small scan:

```bash
uv run python scripts/corpus/build_corpus.py \
    --data-root data/smoke --limit 2000 --cache-k 10
```

Then build the corpus into `data/nq/`:

```bash
uv run python scripts/corpus/build_corpus.py
```

The dataset revision is pinned, so a rebuild reproduces the corpus. Options:
`--fetch` keeps the raw HTML cache, `--force` rebuilds, `--require-clean`
refuses to build from an uncommitted working tree, and `--revision` builds from
a different revision.

### Checking the result

The build prints `Corpus build complete` once `validate` has passed.

Validation can also be run on its own:

```bash
uv run python scripts/corpus/validate.py
```

`data/nq/run.json` holds the exit code of each stage. `data/nq/manifest.json`
holds the build parameters and a digest of every file written.

## H1

```bash
GPUS=0,1,2 bash scripts/h1/run.sh
```

Each model writes its own log. To watch every job at once, from a second
shell:

```bash
uv run python scripts/watch.py
```

| Stage | Axis | Records | Default |
|---|---|---|---|
| 1, 2 | cross-query, within-query | `test`, the reported result | on |
| 3, 4 | cross-query, within-query | `nq_val`, an independent check | on |
| 5, 6 | cross-query, within-query | `all`, the whole corpus | off |

Stages 5 and 6 are descriptive only.

Results land in `data/nq/h1/`, named `{split}_{axis}_{mode}_{model}.json`.

### Truncation check

Find the passages that exceed a model's context after budgeting. This needs
tokenizers only, no GPU:

```bash
uv run python scripts/h1/overflow.py --split test
```

It writes `data/nq/h1/test_overflow.json`. Then score those passages again with
a body that fits every tokenizer, one job per model:

```bash
GPUS=0 bash scripts/h1/rescore_overflow.sh test
```

Each job writes `data/nq/h1/test_rescore_{model}.json`, read by
[`../notebooks/h1/h1_truncation.ipynb`](../notebooks/h1/h1_truncation.ipynb).

### Split-draw sweep

The reported result uses the split drawn with seed 42 in `data/nq/`. Each
further draw gets its own corpus directory, `data/nq_s{seed}/`, which links to
the parsed corpus and the mined negatives in `data/nq/` so that only the split
changes:

```bash
SEED=1
mkdir -p data/nq_s$SEED/splits
ln -s ../nq/parsed_train.json ../nq/parsed_validation.json data/nq_s$SEED/
ln -s ../../nq/splits/bm25_negatives.json data/nq_s$SEED/splits/
uv run python scripts/corpus/split.py --data-root data/nq_s$SEED --seed $SEED
```

Then score the test split of that draw:

```bash
DATA_ROOT=data/nq_s1 STAGES=1,2 GPUS=0,1,2 bash scripts/h1/run.sh
```

Repeat for seeds 1 to 4. Results land in `data/nq_s{seed}/h1/`, read by
[`../notebooks/h1/h1_seed_sweep.ipynb`](../notebooks/h1/h1_seed_sweep.ipynb),
which uses every draw whose six models are all measured.

## H2

```bash
MODEL=bge_base GPUS=0,1,2,3,4 bash scripts/h2/run.sh
```

| Stage | Step | Default |
|---|---|---|
| 0 | baseline eval, `dev` and `test` | on |
| 1, 2, 3 | Phase 1: train one arm per weight, eval on `dev`, choose the weight | on |
| 4, 5 | Phase 2: train the five held-out-format folds at weight 0 and at the chosen weight | on |
| 6, 7 | Phase 2: eval on `test`, score axis then answer axis | on |
| 8 | compare each arm against the baseline | on |
| 9, 10 | prose ranking and comparison | off |

Set `STAGES` to run a subset. The weight is chosen under seed 0, and the
choice lands in `data/nq/h2/selection/{model}_lambda.json`. Adapters land in
`data/nq/h2/train/` and comparisons in `data/nq/h2/comparison/`.

### Training seeds

Repeat Phase 2 under seeds 1 to 4, one model on one GPU, at the weight chosen
under seed 0:

```bash
MODEL=bge_base GPU=0 bash scripts/h2/seed_sweep.sh
```

Every arm it writes ends in `_s<seed>`. Phase 2 and the seeds together are
read by [`../notebooks/h2/h2_analysis.ipynb`](../notebooks/h2/h2_analysis.ipynb).

### Prose check

Rank the prose corpus with every Phase 2 arm and compare each with its
untrained model:

```bash
SEEDS="0 1 2 3 4" GPUS=0 bash scripts/h2/prose.sh
```

Read by [`../notebooks/h2/h2_ablation_nonbox.ipynb`](../notebooks/h2/h2_ablation_nonbox.ipynb).

### Pooler study

The pooler variants (`minilm_l6_pooler`, `minilm_l12_pooler`,
`mxbai_v1_pooler`) also train the pooler, and reuse the weight their base
model chose, so they skip Phase 1:

```bash
MODEL=minilm_l6_pooler STAGES=0,4,5,6,7,8 GPUS=0 bash scripts/h2/run.sh
MODEL=minilm_l6_pooler GPU=0 bash scripts/h2/seed_sweep.sh
```

Then compare each variant with its base model, and measure how far training
moved the head:

```bash
uv run python scripts/h2/pooler_comparison.py --model minilm_l6_pooler
uv run python scripts/h2/head_weight_change.py
```

These write `data/nq/h2/pooler/{variant}.json` and
`data/nq/h2/head_weight_change.json`, read by
[`../notebooks/h2/h2_pooler.ipynb`](../notebooks/h2/h2_pooler.ipynb).

### Representation study

Capture the final-layer representation on `dev`, for the untrained models and
then for the trained arms. The trained captures are large, so they go to a
directory outside the repository:

```bash
SPLITS=dev GPUS=0 bash scripts/h2/representation_capture.sh
FEATURE_DIR=/path/to/captures SPLITS=dev WEIGHTS="0 winner" SEEDS="0 1 2 3 4" GPUS=0 \
    bash scripts/h2/representation_capture.sh
```

Then measure the separation between formats, and check it under other
whitening settings. Neither needs a GPU:

```bash
uv run python scripts/h2/representation_separation.py \
    --models minilm_l6 minilm_l12 bge_base bge_v2_m3 mxbai_v1 jina_v2 \
    --feature-dir /path/to/captures
uv run python scripts/h2/representation_robustness.py --feature-dir /path/to/captures
```

These write `data/nq/h2/format_geometry/`, read by
[`../notebooks/h2/h2_representation.ipynb`](../notebooks/h2/h2_representation.ipynb).

### Standardisation study

Train the single-format arms, one model at a time, alongside the five-format
arms retrained under each seed. An arm whose adapter is present is skipped:

```bash
MODELS=bge_base SEEDS="0 1 2 3 4" GPUS=0 bash scripts/h2/standardisation_train.sh
```

Then score every candidate under every format, for the untrained model and
each of those arms. `LIST=1` prints the arm names without training:

```bash
ARMS="base,$(MODELS=bge_base SEEDS="0 1 2 3 4" LIST=1 bash scripts/h2/standardisation_train.sh)" \
    MODELS=bge_base GPUS=0 bash scripts/h2/standardisation.sh
```

Each capture writes a score tensor to `data/nq/h2/standardisation/`. Every
measurement after that is offline, in
[`../notebooks/h2/h2_standardisation.ipynb`](../notebooks/h2/h2_standardisation.ipynb).

See [`../data/README.md`](../data/README.md).
