SciFact S3/S4 combined BASELINE evaluation package (October 2026)

This package runs a leakage-controlled gold-evidence classification baseline experiment.
It does NOT implement a scientifically novel claim-validity fusion method: no such mathematical specification has yet been prospectively frozen.

Place SciFact_S3S4_Combined_Kit.zip and SciFact_S2_Protocol_Results.zip in D:\DataLineage.
Expand SciFact_S3S4_Combined_Kit.zip into D:\DataLineage.
From Anaconda PowerShell:
  & 'D:\DataLineage\SciFact_S3S4_Combined_Kit\RUN_S3S4.ps1'

Requirements: Python, numpy, scipy, scikit-learn. Existing local SciFact files at:
  D:\DataLineage\ScientificClaim_Benchmark_SciFact\data\data

Integrity: verifies SHA256 of all source files in S2 manifest, enforces official-dev training ID exclusions, checks exact claim text nonoverlap, and never uses unannotated pairs as negatives.

Models: majority, TF-IDF claim-only, evidence-only, and concatenated claim+evidence logistic regression. Three-fold claim-grouped inner CV chooses C from {0.1,1,10} on TRAIN ONLY. Evaluates once on official development, with 10,000 claim-cluster bootstrap resamples for uncertainty. Does not touch unlabelled official test.

Important caveats: this is an oracle gold-evidence task. Annotation-selected evidence sentences are unavailable at inference without a retrieval/evidence identification system. Not end-to-end verification. The paired bootstrap accounts for repeated pairs within claim, but not every shared-document dependence. No independent held-out test evaluation. No novelty or superiority claim.

Output: D:\DataLineage\SciFact_S3S4_Results.zip
Upload the results ZIP to continue with prospective method definition, ablations, CV and robustness.
