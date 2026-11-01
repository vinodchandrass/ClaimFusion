# COVID-Fact verification (code and summary only)

This folder publishes verification code and aggregate outcomes only. **No benchmark data, normalized claim text, per-record predictions, model checkpoints, or manuscript files are distributed.**

## Scope and provenance
Archived binary predictions for 4,086 records (4,080 distinct normalized-claim clusters; 2,790 REFUTED, 1,296 SUPPORTED) were supplied separately as research artifacts. The code recomputes Tables 6 and 8 and Table 7 point estimates from those predictions. It also runs an independent paired cluster bootstrap, sampling normalized-claim clusters with replacement. This is **not** reproduction of the original training, inference, evidence aggregation, or historical bootstrap implementation. The upstream creation of the archived predictions has not been independently authenticated.

## Run with authorized local predictions
```bash
python -m pip install -r verification/requirements.txt
python verification/verify_covid_tables.py --data /path/to/record_predictions.csv --bootstrap 3000 --seed 20261101
```

Required CSV columns: `record_id`, `gold` (0=REFUTED, 1=SUPPORTED), `normalized_claim`, and `frozen_first`, `frozen_mean`, `frozen_rank_discounted`, `weighted_first`, `weighted_mean`, `weighted_rank_discounted`. The CSV is **not** available in this repository. Obtain the underlying benchmark and any derived predictions through authorized channels and confirm licensing.

## Aggregate verification outcome
| Policy | Frozen macro-F1 | Weighted-adapted macro-F1 | Adapted minus frozen | Independent 95% CI (3,000 draws) |
|---|---:|---:|---:|---|
| First | 0.659473 | 0.565527 | -0.093946 | [-0.109425, -0.078606] |
| Mean | 0.685993 | 0.604955 | -0.081038 | [-0.095166, -0.066465] |
| Rank-discounted | 0.682404 | 0.598611 | -0.083793 | [-0.098681, -0.069247] |

All three independently calculated intervals exclude zero. Original archived confidence intervals were first [-0.109475, -0.079164], mean [-0.095452, -0.066724], rank-discounted [-0.097693, -0.068759]. Differences are compatible with Monte Carlo variation; historical seed and bootstrap algorithm are not authenticated.

For mean aggregation, frozen-only correct: 710; adapted-only correct: 321; both correct: 2,151; neither correct: 904.

A separate archived report states that inference replay matched 10,389 evidence pairs for each checkpoint within a probability tolerance of 1e-4. That report was not independently rerun by this code-only release. This repository remains a **partial verification resource**, not an end-to-end replication package.
