# Clean-environment reproducibility gate

This repository contains recovered scripts and archived-results audit reports. A CI workflow runs only the standalone standard-library binary evaluation starter. Passing that workflow means the **starter compiles, its unit tests pass, and its tiny synthetic command-line example succeeds**; it does **not** prove that historical HEALTHVER, SciFact, or COVID-Fact experiments can be rerun.

## Supported now
- Python 3.10 and 3.12 standard-library-only tests for `src/evaluate.py`.
- An intentionally synthetic, three-row CLI fixture.
- Archived prediction/aggregation and bootstrap *audit reports* under `audits/`; consult each report for evidence scope.

## Explicitly not validated by CI
- PyTorch, Transformers, tokenizers, model training, retrieval, dataset loading, checkpoint inference, or third-party dataset licenses.
- Exact checkpoint/tokenizer immutable revisions or hashes and historical environment dependency lock.
- Original author-reported COVID-Fact normalized-claim clustered bootstrap from `src/evaluate.py`: the starter samples individual claim_id rows, **not** the original claim normalization clusters.
- SciFact controlled adaptation ablation: historical frozen maximum length 512 versus adapted length 192.

## Exit criteria for full reproducibility
Authenticate original dependencies/configurations, obtain permissible original input/prediction files, verify all SHA-256 digests, rerun pipeline in a clean isolated environment, compare per-record outputs and aggregate metrics against frozen expected values, and archive exact invocation and logs. Do not change the manuscript or distribute third-party data/weights without permission.

**Status: SMOKE_TEST_WORKFLOW_ADDED; workflow execution not yet observed.**
