# adapters/

Trained LoRA adapters from H2. Each is a rank-16 delta on a frozen base reranker, trained with `L_total = L_rank + ( lambda * L_inv )` for a fixed 500-step budget.

## Licensing

Each adapter inherits its base model's licence (see [`../SOURCES.md`](../SOURCES.md)):

| Adapter | Base model | Licence |
|---|---|---|
| `minilm_l6_*` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0-compatible |
| `minilm_l12_*` | `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0-compatible |
| `bge_*` | `BAAI/bge-reranker-base` | MIT |
| `mxbai_*` | `mixedbread-ai/mxbai-rerank-base-v1` | Apache-2.0-compatible |
| `mxbai_tanh_*` | `mixedbread-ai/mxbai-rerank-base-v1` | Apache-2.0-compatible |
| `jina_*` | `jinaai/jina-reranker-v2-base-multilingual` | CC BY-NC 4.0, non-commercial |

## Layout

```
adapters/
  # trained on all five formats (use this to run a fine-tuned model)
  minilm_l6/  minilm_l12/  bge/  mxbai/  mxbai_tanh/  jina/

  # five-fold hold-one-out adapters, for reproducing the held-out-format results
  <model>_holdout-<format>/
```

Each directory holds `adapter_model.safetensors` and `adapter_config.json`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base = AutoModelForSequenceClassification.from_pretrained(
    "mixedbread-ai/mxbai-rerank-base-v1"
)
model = PeftModel.from_pretrained(base, "adapters/mxbai")
```

The `mxbai_tanh` adapters were trained on a replaced classifier head. Apply the
patch to the base model before attaching one:

```python
from fsr.models.loading import load_model

model = load_model(
    "mixedbread-ai/mxbai-rerank-base-v1",
    device="cuda",
    lora_adapter_path="adapters/mxbai_tanh",
    tanh_head=True,
)
```
