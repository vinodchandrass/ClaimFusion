import json, csv, hashlib, itertools, collections, sys
from pathlib import Path

root, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)

def load(path):
    rows, errors = [], []
    with path.open(encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                x = json.loads(line)
                if not isinstance(x, dict):
                    errors.append({"line": n, "error": "not a JSON object"})
                else:
                    rows.append(x)
            except Exception as e:
                errors.append({"line": n, "error": str(e)})
    return rows, errors

def key(x):
    return str(x) if x is not None else None

def digest(s):
    return hashlib.sha256(str(s).strip().encode()).hexdigest()

required = ["claims_train.jsonl", "claims_dev.jsonl",
            "claims_test.jsonl", "corpus.jsonl"]
for name in required:
    if not (root / name).is_file():
        raise FileNotFoundError(root / name)

data, report = {}, {
    "audit": "SciFact corrected schema-aware audit",
    "files": {}, "split_overlap": {}, "cross_validation": {},
    "evidence_reference_checks": {}, "duplicate_text": {},
    "warnings": []
}

for name in required:
    rows, errors = load(root / name)
    data[name] = rows
    report["files"][name] = {
        "rows": len(rows), "parse_errors": len(errors),
        "first_record_keys": sorted(rows[0]) if rows else [],
        "first_record": rows[0] if name != "corpus.jsonl" and rows else None
    }
    if errors:
        (out / (name + ".parse_errors.json")).write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )

corpus = data["corpus.jsonl"]
corpus_ids = {
    key(x.get("doc_id", x.get("id")))
    for x in corpus
    if x.get("doc_id", x.get("id")) is not None
}
report["corpus"] = {
    "unique_document_ids": len(corpus_ids),
    "duplicate_document_ids": len(corpus) - len(corpus_ids),
    "first_record_keys": sorted(corpus[0]) if corpus else []
}

splits = {}
for name in ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl"]:
    rows = data[name]
    ids = [key(x.get("id")) for x in rows]
    texts = [digest(x["claim"]) for x in rows if x.get("claim") is not None]
    labels = collections.Counter()
    cited_total = cited_missing = evidence_total = evidence_missing = 0
    malformed_evidence = 0
    duplicate_ids = len(ids) - len(set(x for x in ids if x is not None))
    missing_ids = sum(x is None for x in ids)

    for x in rows:
        ev = x.get("evidence", None)
        if isinstance(ev, dict):
            if not ev:
                labels["NO_EVIDENCE_ANNOTATION"] += 1
            else:
                for lab, groups in ev.items():
                    labels[str(lab).upper()] += 1
                    if not isinstance(groups, list):
                        malformed_evidence += 1
                        continue
                    for group in groups:
                        if not isinstance(group, dict):
                            malformed_evidence += 1
                            continue
                        did = key(group.get("doc_id"))
                        evidence_total += 1
                        if did is None or did not in corpus_ids:
                            evidence_missing += 1
        elif ev is not None:
            malformed_evidence += 1

        cited = x.get("cited_doc_ids", [])
        if cited is not None:
            if not isinstance(cited, list):
                malformed_evidence += 1
                cited = []
            for did in cited:
                cited_total += 1
                if key(did) not in corpus_ids:
                    cited_missing += 1

    splits[name] = {
        "ids": set(x for x in ids if x is not None),
        "texts": collections.Counter(texts)
    }
    report["files"][name].update({
        "missing_ids": missing_ids,
        "duplicate_id_rows": duplicate_ids,
        "unique_ids": len(splits[name]["ids"]),
        "unique_claim_texts": len(set(texts)),
        "duplicate_text_excess": len(texts) - len(set(texts)),
        "observed_evidence_categories": dict(labels),
        "cited_document_references": cited_total,
        "cited_document_ids_missing_from_corpus": cited_missing,
        "evidence_groups": evidence_total,
        "evidence_group_doc_ids_missing_from_corpus": evidence_missing,
        "malformed_evidence_items": malformed_evidence
    })

for a, b in itertools.combinations(splits, 2):
    ia, ib = splits[a]["ids"], splits[b]["ids"]
    ta, tb = set(splits[a]["texts"]), set(splits[b]["texts"])
    report["split_overlap"][a + "__" + b] = {
        "shared_ids": len(ia & ib),
        "shared_text_hashes": len(ta & tb)
    }

cvroot = root / "cross_validation"
for i in range(1, 6):
    tp = cvroot / f"fold_{i}" / f"claims_train_{i}.jsonl"
    vp = cvroot / f"fold_{i}" / f"claims_dev_{i}.jsonl"
    if not tp.is_file() or not vp.is_file():
        report["cross_validation"][str(i)] = {"status": "missing files"}
        continue
    tr, te = load(tp)[0], load(vp)[0]
    ti = {key(x.get("id")) for x in tr if x.get("id") is not None}
    vi = {key(x.get("id")) for x in te if x.get("id") is not None}
    th = {digest(x["claim"]) for x in tr if x.get("claim") is not None}
    vh = {digest(x["claim"]) for x in te if x.get("claim") is not None}
    report["cross_validation"][str(i)] = {
        "train_rows": len(tr), "dev_rows": len(te),
        "shared_ids": len(ti & vi),
        "shared_text_hashes": len(th & vh)
    }

# Record exact local hashes for reproducibility
hashes = {}
for p in sorted(root.rglob("*.jsonl")):
    hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
report["sha256"] = hashes

report["warnings"].append(
    "An empty evidence object is not automatically treated as a gold NOT ENOUGH INFO label; inspect official annotation semantics."
)
report["warnings"].append(
    "The unlabelled test file cannot support test accuracy without authenticated gold labels."
)
report["warnings"].append(
    "No semantic or paper-level leakage guarantee is implied by matching claim IDs alone."
)

(out / "SciFact_Corrected_Audit.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
with (out / "SciFact_File_Summary.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["file", "rows", "parse_errors", "unique_ids",
                "duplicate_text_excess", "cited_refs_missing",
                "evidence_refs_missing", "evidence_categories"])
    for name in ["claims_train.jsonl", "claims_dev.jsonl", "claims_test.jsonl", "corpus.jsonl"]:
        x = report["files"][name]
        w.writerow([name, x["rows"], x["parse_errors"], x.get("unique_ids", ""),
                    x.get("duplicate_text_excess", ""),
                    x.get("cited_document_ids_missing_from_corpus", ""),
                    x.get("evidence_group_doc_ids_missing_from_corpus", ""),
                    json.dumps(x.get("observed_evidence_categories", {}))])

print(json.dumps({
    "files": report["files"],
    "corpus": report["corpus"],
    "split_overlap": report["split_overlap"],
    "cross_validation": report["cross_validation"],
    "audit_json": str(out / "SciFact_Corrected_Audit.json")
}, indent=2))
