import json, hashlib, csv, os, sys, collections, itertools
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def load_jsonl(path):
    rows, errors = [], []
    with path.open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    errors.append({"line": line_no, "error": "JSON value is not an object"})
                else:
                    rows.append(obj)
            except Exception as e:
                errors.append({"line": line_no, "error": str(e)})
    return rows, errors

def first_value(obj, keys):
    for k in keys:
        if k in obj and obj[k] is not None:
            return obj[k]
    return None

def norm_id(v):
    return str(v) if v is not None else None

def claim_id(r):
    return norm_id(first_value(r, ["claim_id", "id"]))

def claim_text(r):
    return first_value(r, ["claim", "text", "statement"])

def labels(r):
    vals = []
    for k in ("label", "gold_label", "verdict"):
        if r.get(k) is not None:
            vals.append(str(r[k]).strip().lower())
    ev = r.get("evidence")
    if isinstance(ev, dict):
        vals.extend(str(k).strip().lower() for k in ev.keys())
    return vals

def evidence_doc_ids(obj):
    result = set()
    ev = obj.get("evidence")
    if isinstance(ev, dict):
        for label, groups in ev.items():
            if not isinstance(groups, list):
                continue
            for group in groups:
                if isinstance(group, dict):
                    did = first_value(group, ["doc_id", "document_id", "paper_id"])
                    if did is not None:
                        result.add(norm_id(did))
    elif isinstance(ev, list):
        for item in ev:
            if isinstance(item, dict):
                did = first_value(item, ["doc_id", "document_id", "paper_id"])
                if did is not None:
                    result.add(norm_id(did))
    for k in ("evidence_doc_id", "doc_id", "document_id"):
        v = obj.get(k)
        if v is not None:
            if isinstance(v, list):
                result.update(norm_id(x) for x in v)
            else:
                result.add(norm_id(v))
    return result

files = {
    "train": root / "claims_train.jsonl",
    "dev": root / "claims_dev.jsonl",
    "test": root / "claims_test.jsonl",
    "corpus": root / "corpus.jsonl",
}
for name, path in files.items():
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

report = {
    "audit_type": "SciFact structural integrity and leakage audit",
    "dataset_directory": str(root),
    "files": {},
    "split_overlap": {},
    "cross_validation": {},
    "evidence_reference_audit": {},
    "warnings": [],
}
loaded = {}

for name, path in files.items():
    rows, errors = load_jsonl(path)
    loaded[name] = rows
    ids = [claim_id(r) if name != "corpus" else norm_id(first_value(r, ["doc_id", "document_id", "id"])) for r in rows]
    nonnull = [x for x in ids if x is not None]
    id_counts = collections.Counter(nonnull)
    text_hashes = []
    for r in rows:
        t = claim_text(r) if name != "corpus" else first_value(r, ["abstract", "title", "text"])
        if t is not None:
            text_hashes.append(hashlib.sha256(str(t).strip().encode("utf-8")).hexdigest())
    labs = collections.Counter(x for r in rows for x in labels(r))
    report["files"][name] = {
        "path": str(path),
        "rows": len(rows),
        "parse_errors": len(errors),
        "missing_ids": len(rows) - len(nonnull),
        "unique_ids": len(set(nonnull)),
        "duplicate_id_values": sum(1 for c in id_counts.values() if c > 1),
        "duplicate_id_excess_rows": sum(c - 1 for c in id_counts.values() if c > 1),
        "unique_text_hashes": len(set(text_hashes)),
        "duplicate_text_hash_excess": len(text_hashes) - len(set(text_hashes)),
        "observed_label_tokens": dict(labs),
        "top_level_keys_first_5": [sorted(r.keys()) for r in rows[:5]],
    }
    if errors:
        with (out / f"{name}_parse_errors.json").open("w", encoding="utf-8") as f:
            json.dump(errors, f, indent=2)

def idset(name, corpus=False):
    return {
        norm_id(first_value(r, ["doc_id", "document_id", "id"]) if corpus else claim_id(r))
        for r in loaded[name]
        if (first_value(r, ["doc_id", "document_id", "id"]) if corpus else claim_id(r)) is not None
    }

split_names = ["train", "dev", "test"]
split_ids = {n: idset(n) for n in split_names}
for a, b in itertools.combinations(split_names, 2):
    overlap = split_ids[a] & split_ids[b]
    report["split_overlap"][f"{a}__{b}"] = {
        "shared_claim_ids": len(overlap),
        "examples_first_20": sorted(overlap)[:20],
    }

corpus_ids = idset("corpus", corpus=True)
all_claim_rows = [(split, r) for split in split_names for r in loaded[split]]
ref_summary = {}
for split, r in all_claim_rows:
    for did in evidence_doc_ids(r):
        ref_summary.setdefault(split, {"references": 0, "missing_corpus_references": 0, "missing_ids_examples": []})
        ref_summary[split]["references"] += 1
        if did not in corpus_ids:
            ref_summary[split]["missing_corpus_references"] += 1
            if len(ref_summary[split]["missing_ids_examples"]) < 20:
                ref_summary[split]["missing_ids_examples"].append(did)
report["evidence_reference_audit"] = {
    "corpus_unique_document_ids": len(corpus_ids),
    "claim_evidence_references": ref_summary,
    "note": "Reference checks are best-effort and depend on the observed JSON schema."
}

cv_root = root / "cross_validation"
for i in range(1, 6):
    train_path = cv_root / f"fold_{i}" / f"claims_train_{i}.jsonl"
    dev_path = cv_root / f"fold_{i}" / f"claims_dev_{i}.jsonl"
    if not train_path.is_file() or not dev_path.is_file():
        report["cross_validation"][str(i)] = {"status": "missing fold file"}
        continue
    tr, er1 = load_jsonl(train_path)
    dv, er2 = load_jsonl(dev_path)
    tr_ids = {claim_id(r) for r in tr if claim_id(r) is not None}
    dv_ids = {claim_id(r) for r in dv if claim_id(r) is not None}
    report["cross_validation"][str(i)] = {
        "train_rows": len(tr),
        "dev_rows": len(dv),
        "train_parse_errors": len(er1),
        "dev_parse_errors": len(er2),
        "shared_claim_ids": len(tr_ids & dv_ids),
        "train_unique_ids": len(tr_ids),
        "dev_unique_ids": len(dv_ids),
    }

if any(v["parse_errors"] for v in report["files"].values()):
    report["warnings"].append("One or more JSONL files contain parse errors.")
if any(v["shared_claim_ids"] for v in report["split_overlap"].values()):
    report["warnings"].append("Claim identifiers overlap between official train/dev/test files; investigate before using these splits.")
if any(v["missing_ids"] for v in report["files"].values()):
    report["warnings"].append("Missing record identifiers detected.")
report["warnings"].append("This is a structural audit, not proof of scientific validity or absence of semantic leakage.")
report["warnings"].append("Do not score the unlabelled official test split unless authenticated gold labels are available.")

with (out / "SciFact_Integrity_Audit.json").open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

with (out / "SciFact_File_Inventory.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "rows", "parse_errors", "missing_ids", "unique_ids", "duplicate_id_values", "duplicate_text_hash_excess"])
    for name, rec in report["files"].items():
        w.writerow([name, rec["rows"], rec["parse_errors"], rec["missing_ids"], rec["unique_ids"], rec["duplicate_id_values"], rec["duplicate_text_hash_excess"]])

print(json.dumps({
    "audit_json": str(out / "SciFact_Integrity_Audit.json"),
    "inventory_csv": str(out / "SciFact_File_Inventory.csv"),
    "files": {k: {x: v[x] for x in ["rows", "parse_errors", "missing_ids", "unique_ids"]} for k, v in report["files"].items()},
    "split_overlap": report["split_overlap"],
    "cross_validation": report["cross_validation"],
    "warnings": report["warnings"],
}, indent=2))
