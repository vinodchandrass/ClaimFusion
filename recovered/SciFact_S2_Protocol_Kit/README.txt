S2 PRE-EXPERIMENT PROTOCOL PACKAGE

This stage does not train or evaluate models. It produces a reproducible leakage exclusion manifest, a claim/document evidence candidate registry, source SHA256 hashes, and a draft preregistered experiment plan.

1. Unzip into D:\DataLineage\SciFact_S2_Protocol_Kit
2. In Anaconda PowerShell:
   & 'D:\DataLineage\SciFact_S2_Protocol_Kit\RUN_S2.ps1'
3. Upload D:\DataLineage\ScientificClaim_Benchmark_SciFact\SciFact_S2_Protocol_Results.zip

Important: gold evidence groups are a curated subset; the registry does not yet define a realistic retrieval benchmark. UNKNOWN cited pairs must not be treated as NEI. Official test lacks gold labels. The experiment specification contains pending decisions requiring a separate freeze before actual training.
