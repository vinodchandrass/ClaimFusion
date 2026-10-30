# COVID-Fact paired cluster bootstrap replay — 2026-10-30

**Gate: PASS for archived-prediction bootstrap replay; model inference remains unverified.**

Source: user-supplied ClaimFusion_Recovery(1).zip, files/S129_COVIDFact_External_Validation/record_predictions.csv and paired_cluster_bootstrap.csv, plus S129_COVIDFact_External_Validation.py.

The original implementation clusters by normalized_claim, uses numpy.random.default_rng(20261024), resamples all unique normalized claims with replacement, concatenates their row indices, computes adapted minus frozen binary macro-F1, and takes numpy quantiles at 0.025 and 0.975. There are 4,086 rows and 4,080 unique claim clusters. It performs 2,000 draws per policy, reusing the same generator sequentially in first, mean, rank_discounted order.

Independent implementation of binary macro-F1 from confusion counts and the original bootstrap sampling protocol yielded:

| Policy | CI low | CI high |
|---|---:|---:|
| first | -0.109475339622592 | -0.07916409923625128 |
| mean | -0.09545163531858954 | -0.0667237916628051 |
| rank_discounted | -0.09769256665545208 | -0.06875889369828643 |

All six CI endpoints exactly match the archived paired_cluster_bootstrap.csv values.

**Interpretation:** The negative transfer under mean aggregation is supported by an independently replayed, normalized-claim-clustered bootstrap from archived predictions. This is not evidence that model weights, tokenizer revisions, inference or training can be reconstructed. Normalized-claim clustering does not necessarily account for other dependencies, including shared evidence or source documents. Evidence-order rank_discounted is not independently authenticated as relevance-ranked.
