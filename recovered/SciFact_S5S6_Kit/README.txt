SciFact S5-S6: baseline recovery + leakage-controlled five-fold evaluation.

1. Save SciFact_S5S6_Kit.zip to D:\DataLineage.
2. In Anaconda PowerShell, with claimvalid-lid active:
   Expand-Archive -LiteralPath 'D:\DataLineage\SciFact_S5S6_Kit.zip' -DestinationPath 'D:\DataLineage' -Force
   & 'D:\DataLineage\SciFact_S5S6_Kit\RUN_S5S6.ps1'
3. Upload D:\DataLineage\SciFact_S5S6_Results.zip.

Protocol: Uses only gold annotated evidence, excludes mixed-label claim-document pairs, excludes exact-text-overlapping training claims, tunes C and contradiction threshold with three-fold stratified grouped training CV, evaluates six lexical classifiers on official dev and five official CV folds, bootstraps by claim group (2,000 resamples per configuration).

IMPORTANT: No neural NLI model, no proposed fusion method, no retrieval experiment, and no superiority claims. This is a baseline-recovery and validation package. Source hashes are pinned. Output is not a test-set result. Five folds are not independent external datasets.
