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

| Stage | Axis | Records | Default |
|---|---|---|---|
| 1, 2 | cross-query, within-query | `test`, the reported result | on |
| 3, 4 | cross-query, within-query | `nq_val`, an independent check | on |
| 5, 6 | cross-query, within-query | `all`, the whole corpus | off |

Stages 5 and 6 are descriptive only.

Results land in `data/nq/h1/`, named `{split}_{axis}_{mode}_{model}.json`.

See [`../data/README.md`](../data/README.md).
