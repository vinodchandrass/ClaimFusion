import argparse,csv,json,hashlib,collections,random,statistics
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rate(n,d):return n/d if d else None
def decide(e,c):return 'SUPPORTS' if e>=.5 and e>c else 'REFUTES' if c>=.5 and c>e else 'NOT ENOUGH INFO'
def metrics(rows,key):
 n=len(rows);chosen=[r for r in rows if r[key]!='NOT ENOUGH INFO'];return {'n':n,'accuracy':rate(sum(r[key]==r['gold'] for r in rows),n),'abstention':rate(n-len(chosen),n),'selective_accuracy':rate(sum(r[key]==r['gold'] for r in chosen),len(chosen)),'supports_accuracy':rate(sum(r[key]=='SUPPORTS' for r in rows if r['gold']=='SUPPORTS'),sum(r['gold']=='SUPPORTS' for r in rows)),'refutes_accuracy':rate(sum(r[key]=='REFUTES' for r in rows if r['gold']=='REFUTES'),sum(r['gold']=='REFUTES' for r in rows))}
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--source');a.add_argument('--cache');a.add_argument('--s34c');a.add_argument('--s34d');a.add_argument('--out');a.add_argument('--bootstrap',type=int,default=3000);x=a.parse_args();root=Path(x.base)
 source=Path(x.source) if x.source else root/'S32_FEVER_Resolved/S32_resolved_evidence.csv'
 cache=Path(x.cache) if x.cache else root/'S33_S35_Validation_Output/s34_s35_dev/pair_probabilities.csv'
 c=Path(x.s34c) if x.s34c else root/'S33_S35_Validation_Output/s34c_audit/s34c_factorial_predictions.csv'
 d=Path(x.s34d) if x.s34d else root/'S33_S35_Validation_Output/s34d_source_audit/s34d_claim_source_diagnostics.csv'
 out=Path(x.out) if x.out else root/'S33_S35_Validation_Output/s34e_combined';out.mkdir(parents=True,exist_ok=True)
 scores={}
 for r in csv.DictReader(cache.open(encoding='utf-8-sig',newline='')):
  k=(r['claim_id'],r['page_id'],r['sentence_index']);assert k not in scores;scores[k]=(float(r['p_entailment']),float(r['p_contradiction']))
 keys=collections.defaultdict(set);rowkeys=collections.defaultdict(list);gold={};pages=collections.defaultdict(set)
 for r in csv.DictReader(source.open(encoding='utf-8-sig',newline='')):
  if r['split']!='dev':continue
  cid=r['claim_id'];gold[cid]=r['gold_label']
  if r['resolved']=='1' and r['evidence_text'].strip():
   k=(cid,r['page_id'],r['sentence_index']);assert k in scores,(cid,k);keys[cid].add(k);rowkeys[cid].append(k);pages[cid].add(r['page_id'])
 prev={r['claim_id']:r for r in csv.DictReader(c.open(encoding='utf-8-sig',newline=''))}
 diag={r['claim_id']:r for r in csv.DictReader(d.open(encoding='utf-8-sig',newline=''))}
 rows=[];duplicates_invariant=0
 for cid,ks in keys.items():
  if not ks:continue
  es=max(scores[k][0] for k in ks);cs=max(scores[k][1] for k in ks);baseline=decide(es,cs)
  assert cid in prev and cid in diag and baseline==prev[cid]['max_pair_all'],('PARITY_FAIL',cid)
  assert baseline==decide(max(scores[k][0] for k in rowkeys[cid]),max(scores[k][1] for k in rowkeys[cid]))
  if len(rowkeys[cid])>len(ks):duplicates_invariant+=1
  # Fixed, illustrative conflict flag: both classes meet original 0.5 decision threshold.
  conflict=(es>=.5 and cs>=.5)
  # Conservative abstention only on observed bidirectional NLI conflict; exploratory, no calibration.
  guarded='NOT ENOUGH INFO' if conflict else baseline
  rows.append({'claim_id':cid,'gold':gold[cid],'baseline':baseline,'conflict_guard':guarded,'max_entailment':es,'max_contradiction':cs,'conflict_flag':int(conflict),'distinct_pages':len(pages[cid]),'duplicate_rows':len(rowkeys[cid])-len(ks),'largest_page_share':diag[cid]['largest_page_share'],'unresolved_rows':diag[cid]['unresolved_rows']})
 assert len(rows)==6616 and len(scores)==11047
 rng=random.Random(3405);n=len(rows);delta=[int(r['conflict_guard']==r['gold'])-int(r['baseline']==r['gold']) for r in rows]
 boots=sorted(sum(delta[rng.randrange(n)] for _ in range(n))/n for _ in range(x.bootstrap))
 changed=[r for r in rows if r['baseline']!=r['conflict_guard']]
 report={'status':'EXPLORATORY_DEV_ONLY_SOURCE_IDENTITY_NOT_VALIDATED_PROVENANCE','n':n,'input_sha256':{'source':sha(source),'pair_cache':sha(cache),'s34c_predictions':sha(c),'s34d_diagnostics':sha(d)},'frozen_threshold':0.5,'baseline_parity':'PASS','duplicate_invariance':'PASS','claims_with_duplicate_rows':duplicates_invariant,'methods':{k:metrics(rows,k) for k in ['baseline','conflict_guard']},'conflict':{'flagged':sum(r['conflict_flag'] for r in rows),'changed':len(changed),'baseline_correct_guard_wrong':sum(r['baseline']==r['gold'] and r['conflict_guard']!=r['gold'] for r in rows),'baseline_wrong_guard_correct':sum(r['baseline']!=r['gold'] and r['conflict_guard']==r['gold'] for r in rows)},'paired_accuracy_delta_guard_minus_baseline':{'point':sum(delta)/n,'bootstrap_95ci':[boots[int(.025*x.bootstrap)],boots[min(x.bootstrap-1,int(.975*x.bootstrap))]],'replicates':x.bootstrap},'strata':{},'limitations':['Gold FEVER oracle evidence; no retrieval, no gold NEI evaluation','Page identity is not independently verified source provenance','Conflict gate is an illustrative fixed-threshold sensitivity analysis, not a validated algorithm','Accuracy is on all claims; selective accuracy conditions on nonabstentions','Bootstrap is exploratory and does not correct for multiple comparisons','No SciFact independent confirmation or sealed FEVER test evaluation']}
 for name,predicate in [('single_page',lambda r:r['distinct_pages']==1),('multi_page',lambda r:r['distinct_pages']>1),('duplicate_rows',lambda r:r['duplicate_rows']>0),('no_duplicate_rows',lambda r:r['duplicate_rows']==0),('page_share_100pct',lambda r:float(r['largest_page_share'])==1),('page_share_below_100pct',lambda r:float(r['largest_page_share'])<1),('unresolved_present',lambda r:int(r['unresolved_rows'])>0)]:
  sub=[r for r in rows if predicate(r)];report['strata'][name]={'baseline':metrics(sub,'baseline'),'guard':metrics(sub,'conflict_guard'),'conflict_count':sum(r['conflict_flag'] for r in sub)}
 with (out/'s34e_claim_predictions.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (out/'s34e_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
