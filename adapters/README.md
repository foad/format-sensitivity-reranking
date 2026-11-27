# adapters/

Trained LoRA adapters from H2. Each is a rank-16 delta on a frozen base reranker, trained with `L_total = L_rank + ( lambda * L_inv )` for a fixed 500-step budget.

The adapters are release assets rather than tracked files. Each `adapters-<model>.tar.gz` on the release holds every arm of one model, and unpacks from the repository root into `data/nq/h2/train/`, where the scripts and notebooks read them:

```bash
tar -xzf adapters-bge_base.tar.gz
```

`SHA256SUMS` on the release checks the downloads.

## Licensing

Each adapter inherits its base model's licence (see [`../SOURCES.md`](../SOURCES.md)). Each archive carries it as `data/nq/h2/train/LICENSE-adapters-<model>.txt`:

| Adapter | Base model | Licence |
|---|---|---|
| `minilm_l6_*`, `minilm_l6_pooler_*` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0-compatible |
| `minilm_l12_*`, `minilm_l12_pooler_*` | `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0-compatible |
| `bge_base_*` | `BAAI/bge-reranker-base` | MIT |
| `bge_v2_m3_*` | `BAAI/bge-reranker-v2-m3` | Apache-2.0-compatible |
| `mxbai_v1_*`, `mxbai_v1_pooler_*` | `mixedbread-ai/mxbai-rerank-base-v1` | Apache-2.0-compatible |
| `jina_v2_*` | `jinaai/jina-reranker-v2-base-multilingual` | CC BY-NC 4.0, non-commercial |

## Layout

```
data/nq/h2/train/<model>_<formats>_lam<weight>[_s<seed>]/adapter/
```

| Part | Values |
|---|---|
| `model` | the registry slug, such as `bge_base`, or a pooler variant such as `mxbai_v1_pooler` |
| `formats` | `5fmt` for all five formats, a format name for the fold that holds that format out, or `only_<format>` for one format alone (the standardisation study) |
| `weight` | the invariance weight, `0` for the rank-only control |
| `seed` | the training seed; no suffix is seed 0 |

Each `adapter/` holds `adapter_model.safetensors` and `adapter_config.json`. The `train.jsonl` beside it is the training log, tracked in the repository.

To run a fine-tuned model, use the arm trained on all five formats at the weight its model chose, such as `bge_base_5fmt_lam1`. The chosen weight is in `data/nq/h2/selection/<model>_lambda.json`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
model = PeftModel.from_pretrained(base, "data/nq/h2/train/bge_base_5fmt_lam1/adapter")
```

The pooler variants also train the pooler, which the adapter saves alongside its LoRA weights, so they load the same way.
