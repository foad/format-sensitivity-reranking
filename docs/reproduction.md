# Reproduction

Commands for each pipeline, run from the repository root.

```bash
uv sync
```

## Corpus

Check the pipeline first, over a small scan:

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

The build prints `Corpus build complete` once passed. It reports split leakage, malformed negatives, and any file changed since the stage that wrote it.
Validation can also be ran independently:

```bash
uv run python scripts/corpus/validate.py
```

`data/nq/run.json` holds the exit code of each stage. `data/nq/manifest.json`
holds the build parameters and a digest of every file written.
