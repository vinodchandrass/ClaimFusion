import csv,json,hashlib,zipfile,argparse,re
from collections import Counter
from pathlib import Path

def norm(s):return re.sub(r'[^a-z0-9]+',' ',str(s).casefold()).strip()
def readjsonl(z,n):return [json.loads(x) for x in z.read(n).decode('utf-8-sig').splitlines() if x.strip()]
def readcsv(z,n):return list(csv.DictReader(z.read(n).decode('utf-8-sig').splitlines()))
def writecsv(path,rows,cols):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--s18',required=True);ap.add_argument('--s19',required=True);ap.add_argument('--original',required=True);ap.add_argument('--out',required=True)
 a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.s18) as z:
  claims=readjsonl(z,'claims.jsonl');metadata=readjsonl(z,'claims_metadata.jsonl');candidates=readjsonl(z,'corpus_candidates.jsonl')
 with zipfile.ZipFile(a.s19) as z:
  s19=json.loads(z.read('S19_report.json'));overlap=readcsv(z,'S19_original_scifact_overlap.csv');conflicts=readcsv(z,'S19_conflict_claims.csv')
 original=Path(a.original)
 splitnames=['claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl']
 original_claims={}
 for name in splitnames:
  path=original/name
  if not path.is_file():raise RuntimeError('Missing original SciFact '+str(path))
  original_claims[name]=[json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
 idmap={name:{str(c['id']) for c in cs} for name,cs in original_claims.items()}
 txtmap={name:{norm(c['claim']) for c in cs} for name,cs in original_claims.items()}
 overlaps=[];keys=set();pairrows=[]
 for c in claims:
  cid=str(c['id']);txt=norm(c['claim']);key=(cid,txt)
  if key in keys:raise RuntimeError('Duplicate SciFact-Open claim id+text')
  keys.add(key)
  isplits=sorted(n for n in splitnames if cid in idmap[n]);tsplits=sorted(n for n in splitnames if txt in txtmap[n])
  overlaps.append({'claim_id':cid,'id_splits':';'.join(isplits),'text_splits':';'.join(tsplits),'original_train_text_overlap':int('claims_train.jsonl' in tsplits),'original_dev_text_overlap':int('claims_dev.jsonl' in tsplits),'original_test_text_overlap':int('claims_test.jsonl' in tsplits),'evidence_documents':len(c.get('evidence',{}))})
  for did,v in c.get('evidence',{}).items():
   pairrows.append({'claim_id':cid,'document_id':str(did),'label':v.get('label'),'provenance':v.get('provenance'),'highlight_sentence_count':len(v.get('sentences',[]))})
 if len(claims)!=s19['claims'] or len(pairrows)!=s19['evidence_pairs'] or len(conflicts)!=s19['conflict_claims']:raise RuntimeError('S18/S19 inconsistency')
 prior={(str(x['claim_id']),x['id_splits'],x['text_splits']) for x in overlap}
 now={(x['claim_id'],x['id_splits'],x['text_splits']) for x in overlaps}
 if prior!=now:
  writecsv(out/'S20_overlap_discrepancies.csv',[{'claim_id':k[0],'source':'S19' if k in prior else 'S20','id_splits':k[1],'text_splits':k[2]} for k in sorted(prior^now)],['claim_id','source','id_splits','text_splits'])
  raise RuntimeError('STOP: S19/S20 overlap disagreement; inspect S20_overlap_discrepancies.csv')
 train_text=[r for r in overlaps if r['original_train_text_overlap']]
 dev_text=[r for r in overlaps if r['original_dev_text_overlap']]
 writecsv(out/'S20_claim_overlap_and_eligibility.csv',overlaps,list(overlaps[0]))
 writecsv(out/'S20_evidence_registry.csv',pairrows,list(pairrows[0]))
 cc={str(r['claim_id']) for r in conflicts}
 writecsv(out/'S20_conflict_claim_registry.csv',[dict(x,**{'has_train_text_overlap':int(x['claim_id'] in {r['claim_id'] for r in train_text})}) for x in overlaps if x['claim_id'] in cc],list(overlaps[0])+['has_train_text_overlap'])
 # This stage cannot establish that SciFact-Open labels were unseen: only the researcher can attest to past use.
 status='PASS_STRUCTURAL_AUDIT_HUMAN_EXPOSURE_ATTESTATION_REQUIRED'
 report={'status':status,'s18_archive_sha256':hashlib.sha256(Path(a.s18).read_bytes()).hexdigest(),'s19_archive_sha256':hashlib.sha256(Path(a.s19).read_bytes()).hexdigest(),'counts':{'claims':len(claims),'evidence_pairs':len(pairrows),'opposing_label_claims':len(cc),'train_text_overlaps':len(train_text),'dev_text_overlaps':len(dev_text),'test_text_overlaps':sum(x['original_test_text_overlap'] for x in overlaps),'citation_pairs':sum(x['provenance']=='citation' for x in pairrows),'pooling_pairs':sum(x['provenance']=='pooling' for x in pairrows)},'cannot_establish':['Whether SciFact-Open labels or conflict identities were consulted during prior method selection','Whether pretrained NLI weights encountered original SciFact test claims','True document source dependence, scientific consensus or evidence source reliability','An untouched confirmatory evaluation set']}
 (out/'S20_structural_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (out/'S20_exposure_attestation_TEMPLATE.md').write_text('''# Researcher exposure attestation — complete manually before S21

1. Date when SciFact-Open labels were first opened: **[enter]**
2. Were any SciFact-Open labels, conflict IDs, or error cases used to select models, thresholds, prompts, or fusion operators? **[yes/no/details]**
3. Which prior SciFact splits and labels were used for development? **[enter]**
4. Is any claim-disjoint, never-inspected benchmark reserved for confirmatory evaluation? **[identify, or none]**
5. Intended use of SciFact-Open: **[exploratory / evaluation with disclosed exposure / confirmatory only if justified]**

Do not mark this attestation as completed until filled in and signed/dated by the researcher.
''',encoding='utf-8')
 (out/'S20_prospective_fusion_protocol_DRAFT.md').write_text('''# S21 protocol draft — NOT preregistered or executed

**Unit:** Claim with 2+ independently annotated document-level evidence items. **Observed data:** claim ID, document ID, evidence label, citation/pooling acquisition provenance, text; not source trust scores.

**Frozen comparator:** S7 DeBERTa claim–evidence predictions and a document-level aggregation comparator. Apply identical evidence retrieval/availability conditions.

**Candidate operator (to freeze before evaluation):** use calibrated per-document contradiction/entailment probabilities, de-duplicate identical document IDs and exact duplicate texts, aggregate log odds with a prespecified effective-sample-size correction for observed duplicates only; abstain when evidence is neutral or cross-document scores conflict. Do NOT use ground-truth evidence labels at inference or assume citation provenance equals reliability.

**Ablations:** no deduplication; deduplication only; dependence correction; abstention. **Controls:** matched evidence budget, missing-document and duplicated-document perturbations; synthetic perturbations explicitly flagged. **Metrics:** claim-level macro-F1 where claim-level gold is defined; coverage-risk with confidence intervals; bootstrap by claim. **Eligibility:** do not use SciFact-Open for independent confirmatory evaluation if its labels informed method selection. Obtain another never-inspected dataset or prospectively annotated external claims.

**STOP gates:** no claim-level gold for genuinely mixed evidence; no leakage-safe calibration; no untouched independent test; insufficient conflicts. Under any STOP, report descriptive findings only.
''',encoding='utf-8')
 with zipfile.ZipFile(out.parent/'SciFact_S20_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in sorted(out.iterdir()):z.write(f,f.name)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
