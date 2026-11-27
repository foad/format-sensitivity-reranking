# notebooks/

Analysis notebooks that produce the figures and tables. Each writes its figures to the `figures/` directory beside it.

## `h1/`
- `h1_analysis.ipynb`: primary results on both axes, and the claims across the roster.
- `h1_ablation_metadata_only.ipynb`: both axes with the body removed, the metadata alone.
- `h1_seed_sweep.ipynb`: the same measurement over five draws of the test split, and the coverage of the reported intervals.
- `h1_truncation.ipynb`: the passages that still exceed a model's context, scored again with a body that fits.

## `h2/`
- `h2_analysis.ipynb`: Phase 1 weight selection, the Phase 2 folds against the rank-only control on both axes, ranking quality, and transfer, over five training seeds.
- `h2_ablation_nonbox.ipynb`: ranking quality on ordinary prose with no metadata.
- `h2_pooler.ipynb`: whether also training the frozen pooler changes the held-out response.
- `h2_representation.ipynb`: how far the formats separate in the final-layer representation, and whether that tracks the score axis or the mitigation.
- `h2_standardisation.ipynb`: converting every candidate to one format against training for invariance.

## `demo/`
The pipeline demonstration is on the `demo` branch, tag `demo-v1`.
