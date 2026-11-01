#!/usr/bin/env python3
"""Verify COVID-Fact Tables 6-8 from an independently supplied archived prediction CSV.
No dataset, predictions, checkpoints, or original training are distributed here.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score

POLICIES = {"first": "first", "mean": "mean", "rank-discounted": "rank_discounted"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True,
                    help="Local record_predictions.csv; not distributed in this repository")
    ap.add_argument("--bootstrap", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    d = pd.read_csv(args.data)
    required = {"record_id", "gold", "normalized_claim"} | {
        f"{prefix}_{suffix}" for prefix in ("frozen", "weighted")
        for suffix in POLICIES.values()
    }
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    assert len(d) == 4086 and d.record_id.is_unique
    assert d.gold.isin([0, 1]).all()
    assert d.normalized_claim.notna().all()
    assert d.normalized_claim.nunique() == 4080
    assert (d.gold.value_counts().sort_index().to_numpy() == [2790, 1296]).all()
    y = d.gold.to_numpy(dtype=int)
    groups = list(d.groupby("normalized_claim", sort=True).indices.values())
    for policy, suffix in POLICIES.items():
        a = d[f"frozen_{suffix}"].to_numpy(dtype=int)
        b = d[f"weighted_{suffix}"].to_numpy(dtype=int)
        assert np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all()
        for name, p in (("frozen", a), ("weighted", b)):
            print(policy, name, "accuracy", round(accuracy_score(y, p)*100, 4),
                  "macro_f1", round(f1_score(y, p, average="macro"), 6),
                  "recall_refuted_supported",
                  np.round(recall_score(y, p, labels=[0, 1], average=None)*100, 4))
        print(policy, "adapted_minus_frozen",
              round(f1_score(y, b, average="macro") - f1_score(y, a, average="macro"), 6))
        if policy == "mean":
            print("mean paired counts: both_correct, frozen_only, adapted_only, neither",
                  int(((a == y) & (b == y)).sum()),
                  int(((a == y) & (b != y)).sum()),
                  int(((a != y) & (b == y)).sum()),
                  int(((a != y) & (b != y)).sum()))
        if args.bootstrap > 0:
            rng = np.random.default_rng(args.seed)
            differences = np.empty(args.bootstrap)
            for i in range(args.bootstrap):
                draws = rng.integers(0, len(groups), len(groups))
                ids = np.concatenate([groups[j] for j in draws])
                differences[i] = f1_score(y[ids], b[ids], average="macro") - f1_score(y[ids], a[ids], average="macro")
            print(policy, "independent_cluster_bootstrap_95ci",
                  np.quantile(differences, [0.025, 0.975]))

if __name__ == "__main__":
    main()
