# HEALTHVER and SciFact recovered-prediction metric replay — 2026-10-30

**Gate: PASS — archived prediction-level metric recalculation.**

The user-supplied ClaimFusion_Recovery(1).zip was inspected without uploading the underlying prediction records. Classification metrics were independently recalculated from CSV rows using per-class TP, FP and FN and unweighted macro-F1.

| Archived file | Unit | Rows | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| HealthVer_Transfer_Output/healthver_dev_pair_predictions.csv | HEALTHVER development evidence pair | 1917 | 0.5498174230568597 | 0.38825926882808875 |
| ScientificClaim_Benchmark_SciFact/S7_Results/S7_predictions.csv | SciFact development gold-evidence claim-document pair | 209 | 0.8133971291866029 | 0.8076914002595257 |

HEALTHVER class support by numeric ID: 0=391, 1=533, 2=993. SciFact S7 gold labels: SUPPORT=138 and CONTRADICT=71. Both recomputations match their corresponding recovered reports.

**Scope and limitations:** SciFact S7 uses 209 gold-evidence pairs and is not interchangeable with the manuscript's 188-claim top-document evaluation. No model checkpoints were executed; no training or retrieval was rerun. HEALTHVER development overlap and SciFact duplicate-text leakage remain caveats. The recovered CSVs are not published here pending sensitivity and licensing review.

**Next:** identify the exact archived 188-claim predictions, reconcile frozen versus adapted input-length settings, and test clean-environment execution of dependency-light scripts before claiming full reproduction.
