S7 ZERO-SHOT SCIENTIFIC CLAIM NLI BASELINE

Prerequisites: internet on first run; conda environment with torch transformers huggingface-hub sentencepiece numpy scikit-learn.
Install: python -m pip install torch transformers huggingface-hub sentencepiece numpy scikit-learn
Run: & "D:\DataLineage\SciFact_S7_Semantic_Kit\RUN_S7.ps1"
No inline python -c quoting. The model commit is resolved and frozen before inference. Existing lock refuses model/protocol drift.
Uses gold SciFact evidence sentences; model sees evidence as premise and claim as hypothesis. Per-sentence probabilities averaged; contradiction vs entailment decision, neutral retained as diagnostic. 10,000 paired claim-level bootstrap resamples.
No invented provenance reliability. No fusion claim, no retrieval or test set. Five-fold zero-shot evaluation omitted because no model training occurs; rerunning same examples across folds would be pseudo-replication.
Model download may require internet. Check Hugging Face model access and architecture before running.
