import json
from pathlib import Path

root = Path("D:/DataLineage/ScientificClaim_Benchmark_SciFact/data/data")

for filename in ["claims_train.jsonl", "claims_dev.jsonl"]:
    print("\n===", filename, "===")
    count = 0

    with (root / filename).open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            evidence = row.get("evidence", {})

            if isinstance(evidence, dict) and evidence:
                print(json.dumps({
                    "id": row.get("id"),
                    "claim": row.get("claim"),
                    "evidence": evidence
                }, ensure_ascii=False)[:2500])
                count += 1
                if count == 3:
                    break

    print("Non-empty examples printed:", count)
