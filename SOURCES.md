# Sources, attribution, and licensing

## Licence map

| Component | Path | Licence | Notes |
|---|---|---|---|
| Source code | `src/`, `scripts/`, `notebooks/` | MIT | This repository's own code. |
| Derived corpora | `data/` | CC BY-SA 3.0 | Derived from Natural Questions / Wikipedia. |
| MiniLM adapters | `adapters/minilm_*` | Apache-2.0 (inherited) | Base: `cross-encoder/ms-marco-MiniLM-L6-v2`. |
| mxbai adapters | `adapters/mxbai_*` | Apache-2.0 (inherited) | Base: `mixedbread-ai/mxbai-rerank-base-v1`. |
| jina adapters | `adapters/jina_*` | CC BY-NC 4.0 (inherited) | Base: `jinaai/jina-reranker-v2-base-multilingual`. |

## Dataset attribution

**Natural Questions** (Kwiatkowski, T., Palomaki, J., Redfield, O., et al., 2019. *Natural Questions: A Benchmark for Question Answering Research.* Transactions of the Association for Computational Linguistics, 7, 452-466). Distributed under CC BY-SA 3.0 at https://ai.google.com/research/NaturalQuestions. Its documents are Wikipedia articles under CC BY-SA 3.0, (c) their authors.

The `data/` corpora transform Natural Questions (infobox extraction, body truncation, format rendering) and contain Wikipedia-derived text and so inherit CC BY-SA 3.0.

## Base model licences

| Model | Licence | Role |
|---|---|---|
| `cross-encoder/ms-marco-MiniLM-L6-v2` | Apache-2.0 | H1 + H2 |
| `cross-encoder/ms-marco-MiniLM-L12-v2` | Apache-2.0 | H1 only |
| `BAAI/bge-reranker-base` | MIT | H1 only |
| `BAAI/bge-reranker-v2-m3` | Apache-2.0 | H1 only |
| `mixedbread-ai/mxbai-rerank-base-v1` | Apache-2.0 | H1 + H2 |
| `jinaai/jina-reranker-v2-base-multilingual` | CC BY-NC 4.0 | H1 + H2 |
