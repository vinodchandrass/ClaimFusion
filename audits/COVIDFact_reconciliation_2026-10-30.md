# ClaimFusion next gate: COVID-Fact provenance reconciliation (2026-10-30)

## Recovered evidence
The recovered `recovered/S129_COVIDFact_External_Validation/evaluation_summary.csv` reports 4,086 evaluated records. Under mean aggregation, frozen macro-F1 is **0.6859927096432392** and weighted-adapted macro-F1 is **0.6049547987767432** (difference **−0.08103791086649603**). `paired_cluster_bootstrap.csv` reports the 95% cluster-bootstrap interval **[−0.09545163531858954, −0.0667237916628051]**. These are consistent with the rounded manuscript values, **not independently recomputed**.

## Manifest integrity check
`S129_COVIDFact_External_Validation/manifest.json` includes 4,086 records, 10,389 evidence pairs, max_length=192, hashes for model weights and prediction files. The two files retained in the curated repository, `evaluation_summary.csv` and `paired_cluster_bootstrap.csv`, both **match their manifest SHA-256**.

Five manifest-listed files are **absent from the curated repository**: `evidence_counts.csv`, `frozen_evidence_probabilities.csv`, `record_predictions.csv`, `unique_claim_sensitivity.csv`, and `weighted_evidence_probabilities.csv`. They were not verified or uploaded in this gate. In particular, original claim-level predictions are required to independently recalculate metrics and the paired bootstrap. Some missing files are present in the user's original recovery ZIP, but require size, sensitivity, licensing and integrity screening before publication.

## Additional SciFact caution
Recovered S16 report states `risk_target_met: false`; its threshold fallback at 1.0 had calibration risk 0.2095, above the specified 0.10 target. It explicitly labels synthetic perturbations as sensitivity analysis rather than real-world source-dependence evidence.

## Decision
**PASS: consistency of two COVID-Fact aggregate result files with the manifest. BLOCKED: independent replay and full research reproducibility.** Next gate: check original record-level predictions, hash-match against manifest, independently recompute all metrics, then validate bootstrap implementation and clean-environment execution. No new experiments claimed.
