# Checkpoint file inventory reconciliation — 2026-10-30

Input: user-supplied `ClaimFusion_checkpoint_inventory.csv`, generated from the local Windows D:\DataLineage tree. Inventory contains 19 file records covering `CPU_Domain_Adaptation_Output/checkpoint_unweighted`, `CPU_Domain_Adaptation_Output/checkpoint_weighted`, and `S7_NLI_Model`.

## Hash comparison against recovered COVID-Fact experiment manifest
- `checkpoint_weighted/model.safetensors`: **526385242bdae3043d7462d9cfb848641c96fde196dea846fb678973532cf652** — exact match to `recovered/S129_COVIDFact_External_Validation/manifest.json` models.weighted.weights_sha256.
- `S7_NLI_Model/model.safetensors`: **d8148c6d49e0a7925134294c56326c71fe0ab1dc390e37355e00c7efbb488afa** — exact match to models.frozen.weights_sha256.
- `checkpoint_unweighted/model.safetensors`: **723f82c37787a3efdd09fa09e77a576c6aebd0ef8fda9cdb09dc93231f3a37ef** — locally inventoried, but no matching historical experiment reference established in this gate.

Both adapted checkpoints list `config.json`, `model.safetensors`, `special_tokens_map.json`, `spm.model`, `tokenizer.json`, and `tokenizer_config.json`. The frozen S7 folder additionally lists `pytorch_model.bin`. Both adapted checkpoints have the same config/tokenizer hashes but distinct weight hashes.

## Interpretation
**PASS — file identity concordance between local inventory and historical COVID-Fact manifest for frozen and weighted model weights.** This is stronger provenance evidence than configuration-only recovery, but is **not** an independent reading or execution of the binary model files in this environment. Full inference reproducibility still requires a local inference replay, matching preprocessing/tokenization, exact dependency environment, and record-level output comparison. Do not upload the large checkpoint binaries or third-party datasets to GitHub.

## Next gate
Provide a small, local-only Python inference replay script with strict SHA-256 preflight and metric/output comparison. Run on the user's Windows computer and upload only its compact, non-sensitive result report. Preserve the distinction between observed local hashes and independently reproduced predictions.
