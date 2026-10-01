# data/

The built corpus and the result artefacts.

Licensed CC BY-SA 3.0 (see [`LICENSE`](LICENSE))

Download them from the release and unpack into `data/`, or build them manually.

| Path | Contents | Built by |
|---|---|---|
| `nq/parsed_{train,validation}.json` | Infobox `(k,v)` pairs, body text, and quality flags | `scripts/corpus/parse.py` |
| `nq/splits/` | Article-level 80/10/10 splits, and the BM25 hard negatives | `scripts/corpus/split.py`, `negatives.py` |
| `nq/h1_measurement/*.json` | Per-model per-format scores and summary statistics | `scripts/h1/nq_h1_measurement.py` |
| `nq/h2_compare/`, `h2_selection/`, `h2_tanh_ablation/` | H2 result JSONs | H2 sweep scripts |
| `nq/matched_{train,validation}.json` | The raw Wikipedia HTML cache (2.9 GB). Only needed to re-parse without re-streaming. | `scripts/corpus/fetch.py` |

## Building manually

The dataset revision is pinned so a build reproduces the same corpus. See
[`../docs/reproduction.md`](../docs/reproduction.md).

## Checking a download

`nq/manifest.json` records the build parameters and the size, record count
and SHA-256 of every file, so a downloaded corpus can be checked against
the one the results came from:

```bash
uv run python scripts/corpus/validate.py
```
