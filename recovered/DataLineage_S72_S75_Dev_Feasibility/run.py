#!/usr/bin/env python3
"""S72-S75: DEVELOPMENT-ONLY HealthVer audit. No test reads, no inference."""
import argparse,csv,hashlib,json,re,collections,datetime,pathlib,math

def norm(x): return ' '.join(str(x or '').lower().split())
def tokens(x): return set(re.findall(r'[a-z0-9]+',norm(x)))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def writecsv(p,rows,cols):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=cols);w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--data-dir');a.add_argument('--output');args=a.parse_args()
 base=pathlib.Path(args.base); d=pathlib.Path(args.data_dir) if args.data_dir else base/'S67_HealthVer_Acquisition'/'HealthVer_official'/'data'; out=pathlib.Path(args.output) if args.output else base/'S72_S75_Dev_Feasibility_Output';out.mkdir(parents=True,exist_ok=True)
 paths={s:d/('healthver_'+s+'.csv') for s in ('train','dev')}
 for s,p in paths.items():
  if not p.is_file(): raise SystemExit('Missing '+s+' file: '+str(p))
 rows={s:read(p) for s,p in paths.items()}; required={'id','claim','evidence','label'}
 for s,rs in rows.items():
  if not rs or not required.issubset(rs[0]):raise SystemExit('Invalid '+s+' schema')
  if any(not r['claim'].strip() or not r['evidence'].strip() or not r['label'].strip() for r in rs):raise SystemExit('Missing required data in '+s)
 labels={s:dict(collections.Counter(norm(r['label']) for r in rs)) for s,rs in rows.items()}
 train_e={norm(r['evidence']) for r in rows['train']}; dev_e={norm(r['evidence']) for r in rows['dev']}
 train_c={norm(r['claim']) for r in rows['train']};dev_c={norm(r['claim']) for r in rows['dev']}
 # pair label contradictions; claim labels vary naturally by evidence and are NOT necessarily contradictions
 pairlabels=collections.defaultdict(set); claimlabels=collections.defaultdict(set); claim_evidence=collections.defaultdict(set); evidence_claims=collections.defaultdict(set)
 for split,rs in rows.items():
  for r in rs:
   c,e,l=norm(r['claim']),norm(r['evidence']),norm(r['label'])
   pairlabels[(c,e)].add(l);claimlabels[(split,c)].add(l);claim_evidence[(split,c)].add(e);evidence_claims[(split,e)].add(c)
 contradictions=[{'claim_sha256':hashlib.sha256(c.encode()).hexdigest(),'evidence_sha256':hashlib.sha256(e.encode()).hexdigest(),'labels':'|'.join(sorted(ls))} for (c,e),ls in pairlabels.items() if len(ls)>1]
 writecsv(out/'s73_pair_label_conflicts.csv',contradictions,['claim_sha256','evidence_sha256','labels'])
 claim_summary=[]
 for (s,c),ls in sorted(claimlabels.items()):
  claim_summary.append({'split':s,'claim_sha256':hashlib.sha256(c.encode()).hexdigest(),'evidence_count':len(claim_evidence[(s,c)]),'distinct_pair_labels':len(ls),'labels':'|'.join(sorted(ls))})
 writecsv(out/'s73_claim_evidence_structure.csv',claim_summary,['split','claim_sha256','evidence_count','distinct_pair_labels','labels'])
 # DEVELOPMENT-ONLY exploratory corpus: all unique evidence strings from train, no dev evidence inserted
 # Lexical overlap diagnostic only; not a retrieval benchmark or model validation.
 train_tok=[tokens(e) for e in train_e]; vocab=set().union(*train_tok) if train_tok else set()
 dev_unique_claims={norm(r['claim']) for r in rows['dev']}; dev_claim_with_any_train_e=collections.defaultdict(bool);dev_claim_with_any_dev_e=collections.defaultdict(bool)
 for r in rows['dev']:
  c=norm(r['claim']);e=norm(r['evidence']);dev_claim_with_any_train_e[c]|=e in train_e;dev_claim_with_any_dev_e[c]|=e in dev_e
 # document corpus NOT verified: candidate pool is only paired evidence from train
 overlap={
 'train_dev_unique_claims_exact':len(train_c&dev_c),
 'train_dev_unique_evidence_exact':len(train_e&dev_e),
 'dev_unique_claims_with_any_paired_evidence_in_train_pool':sum(dev_claim_with_any_train_e.values()),
 'dev_unique_claims_total':len(dev_unique_claims),
 'dev_unique_evidence_in_train_pool':len(dev_e&train_e),
 'dev_unique_evidence_total':len(dev_e),
 'train_pool_evidence_unique':len(train_e),
 'train_pool_source_document_ids_verified':False,
 'train_pool_is_independent_nonoracle_document_corpus':False,
 }
 s72={'phase':'S72','inputs':{s:{'path':str(paths[s]),'sha256':sha(paths[s]),'rows':len(rows[s]),'unique_claims':len({norm(r['claim']) for r in rows[s]}),'unique_evidence':len({norm(r['evidence']) for r in rows[s]}),'labels':labels[s]} for s in rows},'test_read':False}
 s73={'phase':'S73','pair_label_conflict_count':len(contradictions),'claim_label_diversity':{s:{'claims_with_multiple_pair_labels':sum(len(ls)>1 for (sp,c),ls in claimlabels.items() if sp==s),'claims_total':sum(1 for (sp,c) in claimlabels if sp==s)} for s in rows},'note':'Different labels for the same claim with different evidence are expected and must not be collapsed to claim-level verdicts.'}
 s74={'phase':'S74','candidate_pool':'unique training evidence strings only','pool_diagnostics':overlap,'note':'Pool membership is an upper-bound availability diagnostic, not a retrieval evaluation. No BM25, NLI, test labels or dev accuracy evaluated.'}
 s75={'phase':'S75','status':'DEVELOPMENT_ONLY_EXTERNAL_VALIDATION_BLOCKED','evidence_pair_classification_feasible_in_principle':True,'claim_level_three_class_policy_frozen':False,'nonoracle_source_document_retrieval_verified':False,'external_evaluator_approved':False,'prior_exposure_and_pretraining_contamination_cleared':False,'independent_holdout_authorized':False,'test_read':False,'model_inference':False,'next_scientific_requirement':'Independent source-document collection or explicitly preregistered pair-level transfer task, plus independent custody and exposure clearance.'}
 for name,obj in [('s72_schema_and_hash_report.json',s72),('s73_label_structure_report.json',s73),('s74_retrieval_feasibility_report.json',s74),('s75_scientific_gate_report.json',s75)]: (out/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
 print('S72-S75 completed: development-only; no test read, no model inference.')
 print('Train rows:',len(rows['train']),'Dev rows:',len(rows['dev']),'Pair label conflicts:',len(contradictions))
 print('Non-oracle document corpus: NOT VERIFIED; independent evaluation: BLOCKED')
if __name__=='__main__':main()
