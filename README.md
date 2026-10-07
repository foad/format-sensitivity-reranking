# Format Sensitivity in Metadata-Enriched Cross-Encoder Reranking

Companion code and artefacts for the MSc dissertation *Evaluating and Mitigating Format Sensitivity in Metadata-Enriched Cross-Encoder Reranking* (Dan Foad, University of Bath, 2026).

The study runs in two phases:

- **H1 (characterisation).** Does a pointwise cross-encoder reranker score the same passage differently when its metadata is serialised as YAML, JSON, TOML, inline key-value, or Markdown? Measured across six models on a refined Natural Questions corpus.
- **H2 (mitigation).** Can a LoRA adapter trained with a composite ranking-plus-invariance objective (`L_total = L_rank + lambda*L_inv`) reduce that sensitivity without harming ranking quality? Evaluated on six model variants with a five-fold hold-one-out design over the five formats, under a fixed training budget.

## Results

Sensitivity is measured on two axes: the shift in a passage's absolute score, as the maximum pairwise Cohen's |d| across format pairs, and the disturbance to ranking (which I call conditional inconsistency), the share of answerable queries whose gold passage ranks first under some formats but not all.

### H1: characterisation

- Format choice shifts a passage's score by up to **Cohen's |d| = 0.94**, above the medium-effect threshold on four of the six models (MiniLM-L6, MiniLM-L12, mxbai-v1, jina-v2). The two BGE models fall below it (|d| < 0.5).
- Ranking disturbance does not track that ordering. Conditional inconsistency peaks at **17.7%** on `bge-reranker-base` (the model least sensitive by |d|), and the three most inconsistent models separate from the other three on non-overlapping intervals. The two axes are therefore near-independent across the six models and capture distinct failure modes.

**Supporting studies**

- Metadata-only ablation: whether the sensitivity comes from the metadata block or from its interaction with the surrounding prose.

Figures and intervals are in [`notebooks/h1/h1_analysis.ipynb`](notebooks/h1/h1_analysis.ipynb) and [`notebooks/h1/h1_ablation_metadata_only.ipynb`](notebooks/h1/h1_ablation_metadata_only.ipynb).

### H2: mitigation

- The invariance term reduces out-of-distribution sensitivity on all six models, by up to **0.218** in maximum pairwise |d| on the held-out format, with every interval excluding zero. The reduction is measured against a rank-only control, so it is the contribution of the invariance term rather than of LoRA fine-tuning.
- The answer axis responds on **three of the six**. MiniLM-L6, bge-base and MiniLM-L12 reduce conditional inconsistency with intervals excluding zero; bge-v2-m3, mxbai-v1 and jina-v2 do not move. The two axes disagree about which models respond, so the near-independence H1 found in the untrained models survives the intervention.
- Ranking quality is preserved on every arm, well inside the pre-registered non-inferiority margin. Capability transfers to the unseen format in full, while invariance transfers only partly and unevenly across models, which is the clearest limit on the intervention.

Figures and intervals are in [`notebooks/h2/h2_analysis.ipynb`](notebooks/h2/h2_analysis.ipynb).

## Repository layout

```
src/fsr/           # library code
  formats.py       #   the five metadata renderers under study
  passages.py      #   tokenizer contract, body budgeting, truncation
  scoring.py       #   cross-encoder scoring primitives
  metrics.py       #   score-axis statistics and bootstrap intervals
  within_query.py  #   answer-axis statistics over one query's candidates
  selection.py     #   the sweep rule that picks the invariance weight
  comparison.py    #   a trained arm against the untrained baseline, both axes
  h2_layout.py     #   artefact paths for the mitigation phase
  h1_results.py    #   notebook readers and table builders
  h2_results.py    #
  corpus/          #   infobox location and text extraction
  models/          #   model loading and vendor-specific patches
  training/        #   batch construction and the composite objective
scripts/           # runnable pipeline entry points
notebooks/         # analysis notebooks
data/              # refined corpus
adapters/          # trained LoRA adapter weights
docs/              # reproduction guide
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

H2 fine-tunes every H1 model.

## Licensing

Multi-licensed (full breakdown in [`SOURCES.md`](SOURCES.md)):

- Code (`src/`, `scripts/`, `notebooks/`): MIT.
- Derived corpora (`data/`): CC BY-SA 3.0, inherited from Natural Questions and Wikipedia.
- LoRA adapters (`adapters/`): each inherits its base model's licence. The jina adapter is CC BY-NC 4.0. The others are Apache-2.0-compatible.

## Citation

See [`CITATION.cff`](CITATION.cff).
