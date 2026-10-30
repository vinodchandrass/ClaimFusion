#!/usr/bin/env python3
"""Verify repository-held archived summaries; no original model inference."""
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "recovered" / "S129_COVIDFact_External_Validation"

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))

def macro_f1(matrix):
    scores = []
    for c in range(2):
        tp = matrix[c][c]
        fp = sum(matrix[r][c] for r in range(2) if r != c)
        fn = sum(matrix[c][r] for r in range(2) if r != c)
        scores.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0)
    return sum(scores) / 2

def verify():
    manifest = json.loads((BASE / "manifest.json").read_text(encoding="utf-8-sig"))
    verified = []
    missing = []
    for name, expected in manifest["files"].items():
        path = BASE / name
        if not path.is_file():
            missing.append(name)
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual.lower() != expected.lower():
            raise AssertionError(f"SHA256 mismatch: {name}")
        verified.append(name)
    for essential in ("evaluation_summary.csv", "paired_cluster_bootstrap.csv"):
        if essential not in verified:
            raise AssertionError(f"Required archived summary missing: {essential}")
    rows = read_csv(BASE / "evaluation_summary.csv")
    by_key = {(r["model"], r["policy"]): r for r in rows}
    assert len(rows) == 6 and len(by_key) == 6
    for row in rows:
        cm = json.loads(row["confusion_matrix_refuted_supported"])
        assert sum(map(sum, cm)) == int(row["n"]) == manifest["records"]
        assert abs(macro_f1(cm) - float(row["macro_f1"])) < 1e-12
        assert abs(sum(cm[i][i] for i in range(2)) / manifest["records"] - float(row["accuracy"])) < 1e-12
        assert abs(cm[0][0] / sum(cm[0]) - float(row["refuted_recall"])) < 1e-12
        assert abs(cm[1][1] / sum(cm[1]) - float(row["supported_recall"])) < 1e-12
    bootstrap = read_csv(BASE / "paired_cluster_bootstrap.csv")
    assert len(bootstrap) == 3
    for row in bootstrap:
        policy = row["policy"]
        delta = float(by_key[("weighted", policy)]["macro_f1"]) - float(by_key[("frozen", policy)]["macro_f1"])
        assert abs(delta - float(row["weighted_minus_frozen_macro_f1"])) < 1e-12
        assert float(row["cluster_bootstrap_ci_low"]) < delta < float(row["cluster_bootstrap_ci_high"])
    assert abs(float(by_key[("frozen", "mean")]["macro_f1"]) - 0.6859927096432392) < 1e-12
    assert abs(float(by_key[("weighted", "mean")]["macro_f1"]) - 0.6049547987767432) < 1e-12
    scifact = json.loads((ROOT / "recovered" / "ScientificClaim_Benchmark_SciFact" / "S7_Results" / "S7_results.json").read_text(encoding="utf-8-sig"))
    assert scifact["gold_pairs"] == 209 and scifact["claims"] == 188
    assert abs(macro_f1(scifact["model_metrics"]["confusion_matrix"]) - scifact["model_metrics"]["macro_f1"]) < 1e-12
    return {"status": "PASS_ARCHIVED_SUMMARY_INTEGRITY", "covid_manifest_files_verified": sorted(verified), "covid_manifest_files_not_present": sorted(missing), "covid_rows": manifest["records"], "scifact_s7_gold_pairs": 209, "scope": "archived summary and manifest checks only; no raw prediction replay or inference"}

if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
