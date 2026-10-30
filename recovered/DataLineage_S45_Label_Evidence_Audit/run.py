import argparse,csv,json,hashlib,collections,pathlib

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def load_jsonl(p):
 with open(p,encoding='utf-8-sig') as f:
  for i,line in enumerate(f,1):
   if line.strip():
    try: yield json.loads(line)
    except Exception as e: raise ValueError(f'Invalid JSON at line {i}: {e}')
def norm_label(x):
 x=str(x).upper().strip()
 return {'SUPPORT':'SUPPORTS','CONTRADICT':'REFUTES'}.get(x,x)
def main():
 a=argparse.ArgumentParser();a.add_argument('--claims',required=True);a.add_argument('--retrieval',required=True);a.add_argument('--out',required=True);x=a.parse_args()
 out=pathlib.Path(x.out);out.mkdir(parents=True,exist_ok=True)
 claims={};duplicates=[]
 for c in load_jsonl(x.claims):
  cid=str(c['id']);
  if cid in claims:duplicates.append(cid)
  claims[cid]=c
 if duplicates:raise ValueError('Duplicate claim IDs: '+str(duplicates[:10]))
 ranked=collections.defaultdict(list)
 with open(x.retrieval,encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):ranked[str(r['claim_id'])].append(r)
 rows=[];details=[];invalid=[]
 for cid,rs in ranked.items():
  c=claims.get(cid)
  if c is None:invalid.append(cid);continue
  ev=c.get('evidence') or {}; labels=set();evdocs=set();sentences=0
  for docid,annotations in ev.items():
   if not isinstance(annotations,list):continue
   for ann in annotations:
    lab=norm_label(ann.get('label',''))
    if lab in ('SUPPORTS','REFUTES'):
     labels.add(lab);evdocs.add(str(docid));sentences+=len(ann.get('sentences') or [])
  label='UNANNOTATED' if not labels else (next(iter(labels)) if len(labels)==1 else 'MIXED_ANNOTATIONS')
  retrieved=[str(r['doc_id']) for r in sorted(rs,key=lambda r:int(r['rank']))]
  cited={str(z) for z in c.get('cited_doc_ids',[])}
  hit_ev=len(evdocs.intersection(retrieved));hit_cited=len(cited.intersection(retrieved))
  rows.append(dict(claim_id=cid,annotation_status=label,annotated_evidence_documents=len(evdocs),annotated_sentences=sentences,cited_documents=len(cited),retrieved_documents=len(retrieved),evidence_doc_any_at_k=int(bool(hit_ev)) if evdocs else '',evidence_doc_all_at_k=int(hit_ev==len(evdocs)) if evdocs else '',cited_doc_any_at_k=int(bool(hit_cited)) if cited else '',cited_doc_all_at_k=int(hit_cited==len(cited)) if cited else ''))
  details.append(dict(claim_id=cid,annotation_status=label,annotated_evidence_doc_ids=sorted(evdocs),cited_doc_ids=sorted(cited),retrieved_doc_ids=retrieved))
 def mean(field):
  vals=[int(r[field]) for r in rows if r[field]!=''];return dict(eligible=len(vals),success=sum(vals),rate=sum(vals)/len(vals) if vals else None)
 report=dict(status='DEVELOPMENT_ONLY',claim_count=len(rows),retrieval_claim_count=len(ranked),missing_claim_ids=invalid[:30],annotation_status_counts=dict(collections.Counter(r['annotation_status'] for r in rows)),evidence_doc_any_at_k=mean('evidence_doc_any_at_k'),evidence_doc_all_at_k=mean('evidence_doc_all_at_k'),cited_doc_any_at_k=mean('cited_doc_any_at_k'),cited_doc_all_at_k=mean('cited_doc_all_at_k'),sha256={'claims_dev':sha(x.claims),'retrieval_csv':sha(x.retrieval)},caveats=['UNANNOTATED is not independently verified NOT_ENOUGH_INFO','MIXED_ANNOTATIONS is not forced into a binary claim label','Evidence annotations are used only after retrieval, for development evaluation','No test file read; no model inference performed','Document retrieval does not establish sentence-level evidence or claim classification'])
 with open(out/'s45_claim_audit.csv','w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else ['claim_id']);w.writeheader();w.writerows(rows)
 (out/'s45_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (out/'s45_retrieval_diagnostics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in details),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
