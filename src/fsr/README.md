# src/fsr/

Library code for the format-sensitivity study.

| Module | Contents |
|---|---|
| `candidates.py` | Candidate lists for the within-query measurement, budgeted per candidate. |
| `cli.py` | The arguments, the skip rule and the reporting shared by the pipeline scripts. |
| `formats.py` | The five metadata renderers under study, and the `FORMATS` registry. |
| `passages.py` | The tokenizer contract, body-token budgeting, and semantic truncation. |
| `scoring.py` | Cross-encoder scoring primitives and the XLM-RoBERTa compatibility patch. |
| `metrics.py` | Score-axis statistics: Cohen's d across formats, reciprocal ranks, bootstrap intervals. |
| `within_query.py` | Answer-axis statistics. |
| `answer_location.py` | Classification of a corpus record by where its short answer appears. |
| `corpus/` | Infobox location, Wikipedia text extraction, the build parameters, the consistency checks, and the build and run records. |
| `models/` | The model roster, model loading, and vendor-specific architecture patches for jina and mxbai. |
| `training/` | Batch construction and the composite ranking-plus-invariance objective. |
