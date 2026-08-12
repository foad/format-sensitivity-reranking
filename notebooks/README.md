# notebooks/

Analysis notebooks that produce the figures and tables.

## `h1/`
- `h1_analysis.ipynb`: primary results across six models.
- `h1_ablation_metadata_only.ipynb`: metadata-only ablation.

## `h2/`
- `h2_analysis.ipynb`: primary results, cross-model synthesis.
- `h2_analysis_{minilm,mxbai,jina}.ipynb`: per-model Phase-1 and Phase-2 results.
- `h2_ablation_nonbox.ipynb`: capability preservation on non-infobox prose.
- `h2_confounds.ipynb`: confound testing (sigma-decomposition, penultimate hidden-state, negative-side stability).
- `h2_mxbai_tanh.ipynb`: tanh symmetric test (architecture-conditional invariance).
