# Format Sensitivity in Metadata-Enriched Cross-Encoder Reranking

Companion code and artefacts for the MSc dissertation *Evaluating and Mitigating Format Sensitivity in Metadata-Enriched Cross-Encoder Reranking* (Dan Foad, University of Bath, 2026).

The study runs in two phases:

- **H1 (characterisation).** Does a pointwise cross-encoder reranker score the same passage differently when its metadata is serialised as YAML, JSON, TOML, inline key-value, or Markdown? Measured across six models on a refined Natural Questions corpus.
- **H2 (mitigation).** Can a LoRA adapter trained with a composite ranking-plus-invariance objective (`L_total = L_rank + lambda*L_inv`) reduce that sensitivity without harming ranking quality? Evaluated on three format-sensitive models with a five-fold hold-one-out design over the five formats.

## Key results

- Format choice shifts scores by up to **Cohen's |d| = ~0.94** on three of six models (MiniLM-L6, mxbai, jina). The two BGE models stay near-invariant (|d| < 0.5). Full results in [`docs/h1-results.md`](docs/h1-results.md).
- A per-model composite-objective LoRA cuts in-distribution max |d| while preserving MRR. Out-of-distribution transfer to unseen formats is architecture-conditional. Full results in [`docs/h2-results.md`](docs/h2-results.md).

## Repository layout

```
src/fsr/         # library code: corpus building, scoring, LoRA
  h1/            #   H1 (characterisation)
  h2/            #   H2 (mitigation)
scripts/         # runnable pipeline entry points
notebooks/       # analysis notebooks
data/            # refined corpus
adapters/        # trained LoRA adapter weights
docs/            # results write-ups, methodology notes
```

## Prerequisites

Requires Python 3.13 and [`uv`](https://docs.astral.sh/uv/).

A CUDA GPU is recommended, results were gathered on RTX 3090 and RTX A5000 GPUs.

*This research made use of Hex, the GPU Cloud in the Department of Computer Science at the University of Bath*

## Reproducing Results

See [`docs/reproduction.md`](docs/reproduction.md) for the full H1 and H2 pipelines and instructions.

## Models

H1 scores six models: 
 - [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
 - [`cross-encoder/ms-marco-MiniLM-L12-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L12-v2)
 - [`BAAI/bge-reranker-base`](https://huggingface.co/BAAI/bge-reranker-base)
 - [`BAAI/bge-reranker-v2-m3`](https://huggingface.co/BAAI/bge-reranker-v2-m3)
 - [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1)
 - [`jinaai/jina-reranker-v2-base-multilingual`](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual)

H2 fine-tunes the three format-sensitive models: `MiniLM-L6`, `mxbai`, `jina`.

## Licensing

Multi-licensed (full breakdown in [`SOURCES.md`](SOURCES.md)):

- Code (`src/`, `scripts/`, `notebooks/`): MIT.
- Derived corpora (`data/`): CC BY-SA 3.0, inherited from Natural Questions and Wikipedia.
- LoRA adapters (`adapters/`): each inherits its base model's licence. The jina adapter is CC BY-NC 4.0. The others are Apache-2.0-compatible.

## Citation

See [`CITATION.cff`](CITATION.cff).
