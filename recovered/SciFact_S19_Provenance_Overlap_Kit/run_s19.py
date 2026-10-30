from pathlib import Path
import zipfile, json, csv, hashlib, argparse, collections, re

def rows(z,name):return [json.loads(x) for x in z.read(name).splitlines() if x.strip()]
def norm(x):return re.sub(r'[^a-z0-9]+',' ',x.casefold()).strip()
def writecsv(p,records,fields):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--s18',required=True);ap.add_argument('--original-data',default='');ap.add_argument('--out',required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(a.s18) as z:
  claims=rows(z,'claims.jsonl');meta=rows(z,'claims_metadata.jsonl');ret=rows(z,'retrievals.jsonl');candidates=rows(z,'corpus_candidates.jsonl')
 meta={str(x['id']):x for x in meta};retr={str(x['claim_id']):x for x in ret};docs={str(x['doc_id']):x for x in candidates}
 conflict=[];pairs=[];issue=[];provenance=collections.Counter();labelprov=collections.Counter();overlap=[]
 for c in claims:
  cid=str(c['id']); evidence=c.get('evidence',{});labs={v.get('label') for v in evidence.values()};m=meta.get(cid,{});cited={str(i) for i in m.get('cited_doc_ids',[])}
  if len(labs)>1:
   conflict.append({'claim_id':cid,'claim_text':c['claim'],'documents':len(evidence),'support':sum(v.get('label')=='SUPPORT' for v in evidence.values()),'contradict':sum(v.get('label')=='CONTRADICT' for v in evidence.values()),'citation':sum(v.get('provenance')=='citation' for v in evidence.values()),'pooling':sum(v.get('provenance')=='pooling' for v in evidence.values()),'source_doc_id':m.get('source_doc_id','')})
  for did,v in evidence.items():
   prov=v.get('provenance');label=v.get('label');provenance[prov]+=1;labelprov[(prov,label)]+=1
   if prov not in ('citation','pooling') or label not in ('SUPPORT','CONTRADICT'):issue.append({'claim_id':cid,'doc_id':did,'issue':'unknown_provenance_or_label'})
   if did not in docs:issue.append({'claim_id':cid,'doc_id':did,'issue':'missing_candidate_doc'})
   pairs.append({'claim_id':cid,'doc_id':did,'label':label,'provenance':prov,'in_cited_doc_ids':int(did in cited),'in_top50_retrieval':int(did in {str(i) for i in retr.get(cid,{}).get('doc_ids',[])})})
 orig=Path(a.original_data) if a.original_data else None
 files=['claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl']
 available=[]; original=[]
 if orig:
  for fn in files:
   p=orig/fn
   if p.is_file():
    available.append(fn);original.extend((fn,x) for x in map(json.loads,p.read_text(encoding='utf-8-sig').splitlines()) if x)
 oldid=collections.defaultdict(list);oldtext=collections.defaultdict(list)
 for fn,x in original:
  oldid[str(x.get('id'))].append(fn);oldtext[norm(x.get('claim',''))].append(fn)
 for c in claims:
  cid=str(c['id']);t=norm(c['claim']);overlap.append({'claim_id':cid,'exact_id_in_original':int(cid in oldid),'id_splits':';'.join(sorted(set(oldid[cid]))),'exact_text_in_original':int(t in oldtext),'text_splits':';'.join(sorted(set(oldtext[t])))})
 writecsv(out/'S19_conflict_claims.csv',conflict,['claim_id','claim_text','documents','support','contradict','citation','pooling','source_doc_id'])
 writecsv(out/'S19_evidence_provenance.csv',pairs,['claim_id','doc_id','label','provenance','in_cited_doc_ids','in_top50_retrieval'])
 writecsv(out/'S19_original_scifact_overlap.csv',overlap,['claim_id','exact_id_in_original','id_splits','exact_text_in_original','text_splits'])
 writecsv(out/'S19_integrity_issues.csv',issue,['claim_id','doc_id','issue'])
 report={'status':'PASS_DESCRIPTIVE_AUDIT' if not issue else 'REVIEW_INTEGRITY_ISSUES','claims':len(claims),'evidence_pairs':len(pairs),'conflict_claims':len(conflict),'provenance':dict(provenance),'provenance_label_counts':{str(k):v for k,v in labelprov.items()},'conflict_provenance_mixes':dict(collections.Counter(('citation' if r['citation'] else '')+'+'+('pooling' if r['pooling'] else '') for r in conflict)),'original_scifact_files_present':available,'exact_text_overlap_count':sum(r['exact_text_in_original'] for r in overlap) if available else None,'exact_id_overlap_count':sum(r['exact_id_in_original'] for r in overlap) if available else None,'integrity_issues':len(issue),'limitations':['Source provenance is evidence acquisition method, not source reliability','Conflicting labels are document-level annotations, not independently verified scientific consensus','Original SciFact overlap cannot be ruled out unless original dataset files were present','No confirmatory evaluation is performed']}
 (out/'S19_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with zipfile.ZipFile(out.parent/'SciFact_S19_Results.zip','w',zipfile.ZIP_DEFLATED) as z:
  for f in out.iterdir():z.write(f,f.name)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
