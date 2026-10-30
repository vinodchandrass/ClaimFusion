import argparse,csv,hashlib,json,datetime,pathlib,collections

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvrows(p):
 with p.open('r',encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,obj):p.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def main():
 a=argparse.ArgumentParser(description='S59-S62 read-only historical audit; never opens claims_test.jsonl')
 a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--out');x=a.parse_args()
 base=pathlib.Path(x.base);out=pathlib.Path(x.out) if x.out else base/'S59_S62_Handoff_Output';out.mkdir(parents=True,exist_ok=True)
 inventory=base/'S55_S58_Custody_Gate_Output'/'s55_artifact_inventory.csv'
 gate=base/'S55_S58_Custody_Gate_Output'/'s58_gate_report.json'
 s20=base/'SciFact_S20_Output'/'S20_claim_overlap_and_eligibility.csv'
 s20registry=base/'SciFact_S20_Output'/'S20_conflict_claim_registry.csv'
 test=base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_test.jsonl'
 rows=[];issues=[]
 if inventory.is_file():
  for r in csvrows(inventory):
   p=pathlib.Path(r['path']);found=p.is_file();actual=sha(p) if found else ''
   ok=found and actual.lower()==r.get('sha256','').lower()
   rows.append(dict(asset=r['asset'],path=str(p),exists=found,hash_match=ok,expected_sha256=r.get('sha256',''),actual_sha256=actual))
   if not ok:issues.append('artifact mismatch: '+r['asset'])
 else:issues.append('S55 artifact inventory missing')
 with (out/'s59_frozen_hash_verification.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['asset','path','exists','hash_match','expected_sha256','actual_sha256']);w.writeheader();w.writerows(rows)
 historical={}
 for name,p in [('s20_overlap',s20),('s20_conflict_registry',s20registry)]:
  item={'path':str(p),'exists':p.is_file()}
  if p.is_file():
   rr=csvrows(p);item.update(sha256=sha(p),rows=len(rr),columns=list(rr[0]) if rr else [],column_value_counts={})
   for c in (list(rr[0]) if rr else []):
    if any(s in c.lower() for s in ['overlap','eligible','status','decision','expos','split','source']):
     v=collections.Counter(str(r.get(c,'')) for r in rr)
     if len(v)<=30:item['column_value_counts'][c]=dict(v)
  historical[name]=item
 if not s20.is_file():issues.append('S20 overlap evidence missing')
 g=json.loads(gate.read_text(encoding='utf-8-sig')) if gate.is_file() else {}
 frozen_ok=bool(rows) and all(r['hash_match'] for r in rows)
 report={'status':'HISTORICAL_EVIDENCE_AUDIT_ONLY_TEST_BLOCKED','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_artifacts_verified':frozen_ok,'frozen_artifact_count':len(rows),'historical_audits':historical,'prior_s58_gates':g.get('gates',{}),'test_file_metadata_only':{'path':str(test),'exists':test.is_file(),'content_read':False,'hash_computed':False},'limitations':['Historical overlap registry is not independent exposure clearance','A missing test-content read does not establish lack of past exposure','No semantic overlap certification or training contamination clearance','No independently attested evaluator custody','No three-class NEI rule or source-provenance validity established'],'issues':issues,'independent_evaluation_authorized':False,'new_model_inference':False}
 dump(out/'s62_handoff_report.json',report)
 text='''# S59–S62 Independent evaluator handoff (NOT authorization)\n\n## Frozen system\n- Candidate policy: `top_doc_mean` (development selected, not independently validated).\n- Historical dev benchmark: 125/188 correct, 66.49% (selection-biased).\n- Verify `s59_frozen_hash_verification.csv` and `s62_handoff_report.json` before transfer.\n\n## Required independent attestations\n1. Evaluator identity, institution, date, independent machine, no investigator access to outcomes.\n2. Source and creation dates of prospective evaluation claims, documents, and labels.\n3. Prior exposure disclosure for investigator, annotators, models, and evaluator.\n4. Exact AND semantic overlap screening against all development and previously viewed claims; independently reviewed.\n5. Training-data contamination assessment and unresolved risks.\n6. Three-class decision policy including NEI, frozen before label exposure; if absent, restrict confirmatory claims to specified annotated binary task.\n7. Source provenance, version identifiers, dependencies and licensing; a Wikipedia page ID alone is not independent provenance.\n8. Explicit dated approval to unlock evaluation and prespecified statistical analysis.\n\n## Separation protocol\n- Custodian retains all evaluation labels and outcomes in an access-controlled location.\n- Investigator transfers frozen inference code, model, policy, checksums, and allowed document corpus only.\n- Custodian performs overlap screening in isolation and logs hashes of evaluation inputs.\n- If eligibility fails, quarantine and report; do not tune against outcomes.\n- No sealed test content may be opened by this audit package.\n\n## Decision\nBLOCKED until every relevant gate is attested. An inventory or historical overlap report cannot itself authorize evaluation.\n'''
 (out/'s62_independent_handoff.md').write_text(text,encoding='utf-8')
 print(json.dumps({'status':report['status'],'frozen_ok':frozen_ok,'s20_found':s20.is_file(),'output':str(out),'issues':issues},indent=2))
if __name__=='__main__':main()
