# src/fsr/

Format Sensitivity Reranking python package.

- `fsr/h1/`: characterisation. Corpus construction, format renderers, model scoring, statistical measures (paired Cohen's *d*, Spearman rho, rank-flip rate).
- `fsr/h2/`: mitigation. Composite loss, LoRA training, jina and mxbai head handling (including the tanh patch), scoring, evaluation metrics.
