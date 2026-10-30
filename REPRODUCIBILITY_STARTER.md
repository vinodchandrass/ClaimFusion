# Executable evaluation starter — not a full research reproduction

This is an original, dependency-free binary **claim-level evaluator**. It does not contain the original training pipeline, checkpoint revisions, retrieval implementation, claim-level predictions, or original bootstrap logs. The historical results are **not independently reproduced** by adding this code.

## Run
```bash
python -m unittest discover -s tests -v
python src/evaluate.py --predictions authenticated_predictions.csv --output metrics.json --resamples 2000 --seed 2026
```

Input CSV: one row per claim with `claim_id,gold,frozen,adapted`; labels `SUPPORTS` or `REFUTES`. Matched IDs alone are insufficient: inference protocols must also be comparable. The 2,000-resample CLI default is **not** evidence of the original run count. Metrics include accuracy, macro-F1, recall, confusion matrices and paired claim-bootstrap percentile CI. No data or manuscript included.

**Release gate:** Authenticate original scripts, exact checkpoint/tokenizer revisions and hashes, optimizer/training settings, dataset splits, BM25 retrieval/index configuration, prediction-level tables, original statistical logs and environment. Only then run an independent end-to-end replay and claim reproducibility.
