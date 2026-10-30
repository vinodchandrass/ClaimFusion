import json
from pathlib import Path

root = Path("D:/DataLineage/ScientificClaim_Benchmark_SciFact/data/data")

for filename in ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]:
    path = root / filename

    with path.open(encoding="utf-8-sig") as f:
        first = json.loads(next(line for line in f if line.strip()))

    print("\n===", filename, "===")
    print("Top-level keys:", sorted(first.keys()))
    print("Claim ID:", first.get("id"))
    print("Claim text:", first.get("claim"))
    print(
        "Evidence structure:",
        json.dumps(first.get("evidence", "ABSENT"), ensure_ascii=False)[:1200]
    )
    print("Cited document IDs:", first.get("cited_doc_ids", "ABSENT"))
