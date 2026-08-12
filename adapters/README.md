# adapters/

Trained LoRA adapters from H2. Each is a rank-16 delta on a frozen base reranker, trained with `L_total = L_rank + ( lambda * L_inv )`.

## Licensing

Each adapter inherits its base model's licence (see [`../SOURCES.md`](../SOURCES.md)):

| Adapter | Base model | Licence |
|---|---|---|
| `minilm_*` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0-compatible |
| `mxbai_*` | `mixedbread-ai/mxbai-rerank-base-v1` | Apache-2.0-compatible |
| `jina_*` | `jinaai/jina-reranker-v2-base-multilingual` | CC BY-NC 4.0, non-commercial |

## Layout

```
adapters/
  # trained on all five formats (use this to run a fine-tuned model)
  minilm/
  mxbai/
  jina/

  # five-fold hold-one-out adapters, for reproducing OOD results
  minilm_holdout-<format>/
  mxbai_holdout-<format>/
  jina_holdout-<format>/
```

Each directory holds `adapter_model.safetensors` and `adapter_config.json`.

## Loading

```python
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

base = AutoModelForSequenceClassification.from_pretrained("mixedbread-ai/mxbai-rerank-base-v1")
model = PeftModel.from_pretrained(base, "adapters/mxbai")
```
