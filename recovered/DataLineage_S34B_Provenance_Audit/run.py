import argparse,csv,json,hashlib,collections,math,random
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def decide(s,r):return 'SUPPORTS' if s>=.5 and s>r else 'REFUTES' if r>=.5 and r>s else 'NOT ENOUGH INFO'
def metric(rows,key):
 n=len(rows);return {'n':n,'accuracy':sum(x[key]==x['gold'] for x in rows)/n if n else None,'abstention_rate':sum(x[key]=='NOT ENOUGH INFO' for x in rows)/n if n else None}
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default=r'D:\DataLineage');p.add_argument('--bootstrap',type=int,default=3000);a=p.parse_args()
 root=Path(a.base);source=root/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv';cache=root/'S33_S35_Validation_Output'/'s34_s35_dev'/'pair_probabilities.csv';out=root/'S33_S35_Validation_Output'/'s34b_provenance_audit';out.mkdir(parents=True,exist_ok=True)
 scores={}
 with cache.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   k=(r['claim_id'],r['page_id'],r['sentence_index']);assert k not in scores,k;scores[k]=(float(r['p_entailment']),float(r['p_contradiction']))
 claims={};seen=collections.Counter();unresolved=collections.Counter()
 with source.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['split']!='dev':continue
   cid=r['claim_id'];c=claims.setdefault(cid,{'gold':r['gold_label'],'groups':collections.defaultdict(list)})
   assert c['gold']==r['gold_label'];key=(cid,r['page_id'],r['sentence_index']);g=r['group_index'];seen[(cid,g)]+=1
   if r['resolved']=='1' and r['evidence_text'].strip():
    if key not in scores:raise ValueError('Missing score: '+str(key))
    c['groups'][g].append(key)
   else:unresolved[(cid,g)]+=1
 records=[];incomplete=0
 for cid,c in sorted(claims.items()):
  groups=c['groups'];complete=[list(dict.fromkeys(keys)) for g,keys in groups.items() if keys and unresolved[(cid,g)]==0]
  partial=[list(dict.fromkeys(keys)) for g,keys in groups.items() if keys]
  if not partial:continue
  # Structural metadata from FEVER annotations, NOT independent source trust.
  unique=list(dict.fromkeys(k for group in partial for k in group));pmax=(max(scores[k][0] for k in unique),max(scores[k][1] for k in unique))
  # Conservative completeness gate: only groups without unresolved rows; does not infer missing original references.
  eligible=[k for group in complete for k in group]
  if not complete:incomplete+=1
  cg=(max((min(scores[k][0] for k in group) for group in complete),default=0),max((max(scores[k][1] for k in group) for group in complete),default=0))
  rec={'claim_id':cid,'gold':c['gold'],'max_pair':decide(*pmax),'complete_group_gate':decide(*cg),'n_unique_pairs':len(unique),'n_groups_with_resolved_rows':len(partial),'n_fully_resolved_groups':len(complete),'has_incomplete_group':int(any(unresolved[(cid,g)] for g in set(groups)|{k[1] for k in unresolved if k[0]==cid}))}
  records.append(rec)
 if len(records)!=6616:raise ValueError('Eligibility mismatch, expected 6616, got '+str(len(records)))
 report={'status':'EXPLORATORY_STRUCTURAL_PROVENANCE_PROXY_ONLY','source_sha256':sha(source),'pair_cache_sha256':sha(cache),'claims':len(records),'no_complete_groups':incomplete,'metrics':{k:metric(records,k) for k in ['max_pair','complete_group_gate']},'limitations':['FEVER gold group membership is oracle annotation metadata, not independently observed provenance','Completeness is defined by rows in resolved CSV; missing original group references cannot be recovered here','The complete-group gate is a diagnostic sensitivity analysis, not a validated provenance-aware method','No retrieval, source independence, NEI gold classification, or SciFact independent confirmation','No thresholds selected from test data']}
 rng=random.Random(20261019);n=len(records);delta=[]
 for _ in range(a.bootstrap):
  sample=[records[rng.randrange(n)] for __ in range(n)];delta.append(sum((r['complete_group_gate']==r['gold'])-(r['max_pair']==r['gold']) for r in sample)/n)
 delta.sort();report['paired_delta_complete_group_minus_max_pair']={'point':report['metrics']['complete_group_gate']['accuracy']-report['metrics']['max_pair']['accuracy'],'bootstrap_95ci':[delta[int(.025*len(delta))],delta[min(len(delta)-1,int(.975*len(delta)))]],'resamples':a.bootstrap}
 with (out/'s34b_claim_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (out/'s34b_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
