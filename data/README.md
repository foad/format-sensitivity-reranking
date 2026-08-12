# data/

Derived corpora and result artefacts.

Licensed CC BY-SA 3.0 (see [`LICENSE`](LICENSE))

## Provided data

| Path | Contents | Rebuilt by |
|---|---|---|
| `nq/parsed_{train,validation}.json` | Parsed infobox `(k,v)` pairs + extracted body + quality flags | `scripts/h1/parse_nq_cache.py` |
| `nq/h1_measurement/*.json` | Per-model per-format raw scores + summary stats (H1) | `scripts/h1/nq_h1_measurement.py` |
| `nq/h2_splits/` | Article-level 80/10/10 splits | `scripts/h2/build_splits.py` |
| `nq/h2_compare/`, `h2_selection/`, `h2_tanh_ablation/` | H2 result JSONs | H2 sweep scripts |

## Optional data

Not needed to reproduce the main results or to use an adapter.

- `nq/matched_{train,validation}.json`: the raw NQ and Wikipedia HTML cache (~2.9 GB), the input to parsing. Only to rebuild the corpora from raw, with `scripts/h1/build_nq_cache.py`.
- Hidden-state `.npz` dumps: only for the penultimate-hidden-state ablation test, regenerated with `scripts/h2/eval_hidden.py`.
 - Seed: `0`
