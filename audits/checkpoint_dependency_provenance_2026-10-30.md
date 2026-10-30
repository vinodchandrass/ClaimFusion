# Checkpoint and dependency provenance gate — 2026-10-30

## Evidence inspected
Original user-supplied recovery archive `ClaimFusion_Recovery(1).zip` (not published wholesale). Recovered paths:
- `files/DataLineage_CPU_Domain_Adaptation/run.py`
- `files/DataLineage_CPU_Domain_Adaptation/requirements.txt`
- `files/DataLineage_CPU_NLI_Comparison/requirements.txt`
- `files/DataLineage_CrossDomain_Fusion/requirements.txt`
- `files/CPU_Domain_Adaptation_Output/checkpoint_unweighted/config.json`
- `files/CPU_Domain_Adaptation_Output/checkpoint_weighted/config.json`

## Verified configuration facts
Both recovered adapted checkpoint **configuration JSONs** declare `DebertaV2ForSequenceClassification`, `model_type=deberta-v2`, 3 classes, `id2label` 0=contradiction, 1=entailment, 2=neutral, 12 layers, hidden size 768, vocabulary size 128100, and `transformers_version=4.57.6`. These are metadata, **not evidence that the model weight files are present or match a frozen hash**. The `max_position_embeddings=512` architectural field does not establish the evaluation token truncation length.

Recovered CPU domain-adaptation runner exposes defaults: epochs 1, batch size 4, gradient accumulation 4, max length 192, learning rate 1.5e-5, validation fraction 0.2, seed 20261021, threads 4. It explicitly requires `--dev-only-after-training` and excludes exact overlapping normalized claims from its internal fit/validation subsets, but acknowledges evidence reuse remains. Defaults do **not** authenticate the precise command-line arguments of the historical completed runs.

Three requirements files specify only broad ranges:
- adaptation: `torch>=2.2,<3`, `transformers>=4.40,<5`, `sentencepiece>=0.2`, `safetensors>=0.4`
- NLI comparison: `torch>=2.2`, `transformers>=4.41,<5`, `safetensors>=0.4`, `sentencepiece>=0.2`
- fusion: same broad ranges as adaptation.

## Determination
**PARTIAL PASS: recovered architecture metadata, source entry point, dependency families, and declared defaults. BLOCKED: exact dependency lock, authenticated base-model immutable revision, full weights/tokenizer bundle, historical invocation, and clean-environment model inference.**

No training, model inference, or package installation was performed for this gate. Do not claim the prior SciFact frozen 512 versus adapted 192 truncation confound is resolved by architectural max-position metadata.

## Efficient next action
Generate a local-only checkpoint inventory (names, sizes, SHA-256) from the original Windows `D:\DataLineage` checkpoint folders, and compare with the recorded model-weight hashes. Upload **only the inventory**, not checkpoint binaries or datasets, after reviewing for sensitive paths. Once model identities and full tokenizer files are confirmed, attempt a separate reproducibility job; do not rerun costly training merely to reconstruct unavailable history.
