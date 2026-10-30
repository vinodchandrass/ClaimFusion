import argparse,csv,json,collections,hashlib
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def decision(e,c):
 return 'SUPPORTS' if e>=.5 and e>c else 'REFUTES' if c>=.5 and c>e else 'NOT ENOUGH INFO'
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');x=a.parse_args();root=Path(x.base)
 source=root/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv';cache=root/'S33_S35_Validation_Output'/'s34_s35_dev'/'pair_probabilities.csv';diag=root/'S33_S35_Validation_Output'/'s34b_provenance_audit'/'s34b_claim_diagnostics.csv';out=root/'S33_S35_Validation_Output'/'s34c_audit';out.mkdir(parents=True,exist_ok=True)
 scores={}
 with cache.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   k=(r['claim_id'],r['page_id'],r['sentence_index']);assert k not in scores;scores[k]=(float(r['p_entailment']),float(r['p_contradiction']))
 groups=collections.defaultdict(lambda:collections.defaultdict(list));missing=collections.Counter();gold={}
 with source.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['split']!='dev':continue
   cid=r['claim_id'];g=r['group_index'];gold[cid]=r['gold_label'];key=(cid,r['page_id'],r['sentence_index'])
   if r['resolved']=='1' and r['evidence_text'].strip():
    assert key in scores;groups[cid][g].append(key)
   else:missing[(cid,g)]+=1
 previous={}
 with diag.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):previous[r['claim_id']]=r
 rows=[]
 for cid,gs in groups.items():
  allkeys=list(dict.fromkeys(k for v in gs.values() for k in v));
  if not allkeys:continue
  full=[list(dict.fromkeys(v)) for g,v in gs.items() if v and not missing[(cid,g)]]
  fullkeys=list(dict.fromkeys(k for v in full for k in v))
  def mx(keys):return decision(max((scores[k][0] for k in keys),default=0),max((scores[k][1] for k in keys),default=0))
  def conjunctive(gr):return decision(max((min(scores[k][0] for k in v) for v in gr),default=0),max((max(scores[k][1] for k in v) for v in gr),default=0))
  old=previous[cid];r={'claim_id':cid,'gold':gold[cid],'max_pair_all':mx(allkeys),'max_pair_complete_only':mx(fullkeys),'conjunctive_all_groups':conjunctive([list(dict.fromkeys(v)) for v in gs.values() if v]),'conjunctive_complete_only':conjunctive(full),'has_incomplete_group':old['has_incomplete_group'],'n_complete_groups':len(full)}
  assert r['max_pair_all']==old['max_pair'];assert r['conjunctive_complete_only']==old['complete_group_gate'];rows.append(r)
 assert len(rows)==6616
 keys=['max_pair_all','max_pair_complete_only','conjunctive_all_groups','conjunctive_complete_only']
 report={'status':'EXPLORATORY_DEV_FACTORIAL_AUDIT','n':len(rows),'source_sha256':digest(source),'cache_sha256':digest(cache),'diagnostics_sha256':digest(diag),'methods':{},'comparisons':{},'limitations':['Oracle FEVER annotations are not independent provenance','No gold NEI cases, retrieval evaluation, or independent SciFact confirmation','No model inference performed; frozen pair cache reused']}
 for k in keys:report['methods'][k]={'accuracy':sum(r[k]==r['gold'] for r in rows)/len(rows),'abstention':sum(r[k]=='NOT ENOUGH INFO' for r in rows)/len(rows)}
 for a,b in [('max_pair_all','max_pair_complete_only'),('conjunctive_all_groups','conjunctive_complete_only'),('max_pair_all','conjunctive_all_groups'),('max_pair_complete_only','conjunctive_complete_only')]:
  changed=[r for r in rows if r[a]!=r[b]];report['comparisons'][a+'__vs__'+b]={'changed':len(changed),'changed_with_incomplete_flag':sum(r['has_incomplete_group']=='1' for r in changed),'a_correct_b_wrong':sum(r[a]==r['gold'] and r[b]!=r['gold'] for r in rows),'a_wrong_b_correct':sum(r[a]!=r['gold'] and r[b]==r['gold'] for r in rows)}
 with (out/'s34c_factorial_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'s34c_factorial_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
