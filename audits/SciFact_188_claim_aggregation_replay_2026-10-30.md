# SciFact 188-claim archived-probability replay — 2026-10-30

**PASS: independent aggregation-stage accuracy replay. NOT VERIFIED: model inference or causal adaptation benefit.**

Source: user-provided ClaimFusion_Recovery(1).zip, files/CrossDomain_Fusion_Output/adapted_scifact_candidates_DEV_ONLY.csv and scifact_dev_gold.csv. The source contains 4,500 sentence candidates (300 claims, five ranked documents and three sentences per document) and 188 binary-labelled development claims.

Independent replay grouped sentence probabilities by claim and document rank. Per-document margin = mean(entailment probability minus contradiction probability). Four decision policies: top document margin, arithmetic mean across five documents, weighted margin with weight 1/log2(rank+2), and majority sign vote. Predicted SUPPORTS when policy score > 0, otherwise REFUTES. Both original frozen probability columns and adapted_ probability columns were replayed.

| Policy | Frozen correct /188 | Adapted correct /188 |
|---|---:|---:|
| top_document | 125 | 149 |
| five_doc_mean | 101 | 139 |
| rank_discounted | 100 | 139 |
| majority | 84 | 130 |

All eight counts match the original development-only report. Top-document accuracy frozen 0.6648936170, adapted 0.7925531915; difference 24/188 = 0.1276595745.

**Limitations:** This is a replay from archived inference probabilities, not a fresh checkpoint inference. The original report is marked DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION. Adapted maximum sequence length is 192, whereas historical frozen comparison used 512; architecture, checkpoint revisions and identical tokenization settings have not been authenticated. Do not interpret the difference as isolated adaptation effect. The rank-discounted policy uses document rank, not independently verified relevance ranking. Dataset split text overlap remains a separate caution.

**Next:** clean-environment testing of the aggregation code and provenance/availability checks for checkpoint and tokenizer artifacts. Do not publish third-party data or the manuscript.
