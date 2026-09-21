# src/fsr/

Library code for the format-sensitivity study.

| Module | Contents |
|---|---|
| `formats.py` | The five metadata renderers under study, and the `FORMATS` registry. |
| `passages.py` | The tokenizer contract, body-token budgeting, and semantic truncation. |
| `scoring.py` | Cross-encoder scoring primitives and the XLM-RoBERTa compatibility patch. |
| `metrics.py` | Score-axis statistics: Cohen's d across formats, reciprocal ranks, bootstrap intervals. |
| `within_query.py` | Answer-axis statistics: conditional inconsistency and within-query rank disturbance over one query's candidate list. |
| `corpus/` | Infobox location, Wikipedia text extraction, and the build parameters. |
| `models/` | Model loading, and vendor-specific architecture patches for jina and mxbai. |
| `training/` | Batch construction and the composite ranking-plus-invariance objective. |
