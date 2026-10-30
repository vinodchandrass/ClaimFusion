# ClaimFusion — scientific claim verification research

Research artifacts for evaluating cross-domain adaptation and claim-level evidence aggregation in scientific claim verification.

**Status: documentation-only initial release; not an independently reproducible experiment release.** Source code, original training configuration, exact checkpoint revisions, and claim-level bootstrap inputs have not yet been authenticated and are **not** represented as available.

## Reported results and their evidentiary limits

| Evaluation | Reported observation | Qualification |
|---|---|---|
| HEALTHVER development | Frozen accuracy 54.98%, macro-F1 0.3883; unweighted adapted 77.62%, 0.7595; weighted adapted 78.46%, 0.7679 | Training/development evidence-string overlap limits generalization interpretations |
| SciFact annotated development | 188 binary-labeled claims; adapted top-document accuracy 149/188 = 79.26%; historical frozen top-document 125/188 = 66.49% | Frozen/adapted input lengths differed (512 vs 192); not a matched causal comparison |
| SciFact aggregation | Adapted five-document 139/188 = 73.94%; majority vote 130/188 = 69.15%; cross-fitted selection 145/188 = 77.13% | Original claim-level provenance remains unavailable for independent paired recomputation |
| COVID-Fact external | 4,086 records; frozen mean-aggregation macro-F1 0.6860; adapted 0.6050; difference −0.0810; reported 95% bootstrap CI [−0.0955, −0.0667] | Reported matched inference comparison; original run artifacts must be authenticated before independent replication claims |

These values are **transcribed from the existing research record**, not recomputed in this repository.

## Provenance and reproducibility gaps

- Exact base-model/checkpoint IDs, immutable revisions, and checkpoint hashes: **not verified**.
- Optimizer, learning rate, batch size, epochs, seeds, and final training configuration: **not verified**.
- Original claim-level prediction tables, split identifiers, bootstrap draws, and resampling logs: **not verified**.
- BM25 retrieval configuration and sentence-selection implementation: **not independently replayed**.
- A modern dense-retrieval baseline and calibration analysis: **not performed**.
- No original datasets or model weights are mirrored here.

## Publication and interpretation policy

This repository is not evidence of a successful end-to-end reproduction. Differences between frozen and adapted checkpoints should be described as observed under their evaluation configurations, not automatically attributed causally to adaptation. Figure labels quoting 2,000 resamples reflect a prior analysis summary and do not authenticate underlying bootstrap draws.

## Next required artifacts

Deposit authenticated source scripts, environment lock/requirements, actual experiment configurations, checkpoint revision and SHA-256 manifests, dataset acquisition/split manifest, predictions with per-claim IDs, evaluation and bootstrap code, and reproducible figure generation scripts. Review licenses and confidentiality before uploading any third-party data or checkpoints.

## Citation

Manuscript title and publication details should be confirmed against the final submitted version before assigning a formal citation or archival release.
