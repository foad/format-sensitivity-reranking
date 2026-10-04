# data/

The built corpus and the result artefacts.

Licensed CC BY-SA 3.0 (see [`LICENSE`](LICENSE))

The result JSONs and their run logs are in the repository. The corpus is not:
download it from the release and unpack into `data/`, or build it manually.

| Path | Contents | Built by |
|---|---|---|
| `nq/parsed_{train,validation}.json` | Infobox `(k,v)` pairs, body text, and quality flags | `scripts/corpus/parse.py` |
| `nq/splits/` | Article-level 80/10/10 splits, and the BM25 hard negatives | `scripts/corpus/split.py`, `negatives.py` |
| `nq/h1/*.json` | Per-model H1 results, both axes and both modes. Tracked | `scripts/h1/run.sh` |
| `nq/h1/*_candidates*.json` | Candidate lists, rebuilt by the measurement | `scripts/within_query.py` |
| `nq/h2_compare/`, `h2_selection/` | H2 result JSONs | H2 sweep scripts |
| `nq/matched_{train,validation}.json` | The raw Wikipedia HTML cache (2.9 GB). Only needed to re-parse without re-streaming. | `scripts/corpus/fetch.py` |

## Result file names

```
nq/h1/{split}_{axis}_{mode}_{model}.json
```

| Part | Values |
|---|---|
| `split` | `test` for the reported result, `nq_val` for the independent check, `all` for the pooled descriptive run |
| `axis` | `cross` for score-axis comparisons between queries, `within` for the answer axis inside one query's candidate list |
| `mode` | `with_body` for metadata and body text, `metadata_only` for the ablation |
| `model` | the registry slug, such as `bge_base` |

Each file also records its split and mode, so the name is an index rather than
the only description. A `.log` beside it holds the run output.
A `.progress` file holds the live state of a running job and is not kept.

`all` (pooled) is `train`, `dev` and `test` from `nq_train` together.
It overlaps the H2 training data, so is only used to describe the effect at scale as part of the robustness checks.

`nq_val` is a separate article set from `nq_train` and is used for independent robustness checks.

## Building manually

The dataset revision is pinned so a build reproduces the same corpus. See
[`../docs/reproduction.md`](../docs/reproduction.md).

## Checking a download

`nq/manifest.json` records the build parameters and the size, record count
and SHA-256 of every corpus file, so a downloaded corpus can be checked
against the one the results came from:

```bash
uv run python scripts/corpus/validate.py
```
