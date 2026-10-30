# ClaimFusion experimental artifact recovery gate

Status: **BLOCKED — original experimental inputs not supplied** (2026-10-30).

## Inspection scope

The current mounted workspace has 43 files: 29 PNG, 4 TeX, 3 BibTeX, 4 ZIP, and 3 PDF. All four ZIP member lists were checked: only manuscript, bibliography, figures, PDF, and one BBL are present. No Python scripts, notebooks, experiment configurations, checkpoints, prediction tables, or original run logs were found. This is **only** an audit of the available workspace, not the researcher's original computer.

## Required to claim reproducibility

1. Original training and evaluation code with dependency lock/environment versions.
2. Immutable model and tokenizer IDs/revisions, checkpoint hashes, and training configurations.
3. Dataset versions, preprocessing, split IDs, label mapping, and overlap analysis.
4. Per-claim and per-evidence predictions with stable IDs and inference settings.
5. Retrieval configuration and BM25 corpus/index, top-k and sentence-selection code.
6. Paired tests and bootstrap code, resampling units, seeds, logs, and outputs.
7. Clean-environment execution instructions and passing replay checks.

Do not upload the manuscript or third-party datasets. Do not invent missing experiments or advertise independent reproduction before a replay passes. Existing README values are *reported* rather than independently recomputed.

## Fastest next step

Supply one ZIP of the original experiment directory (scripts, configs, logs, prediction CSV/JSON, requirements, checkpoint metadata; omit large datasets and weights initially). Audit hashes, build a runnable release, and commit verified artifacts only.