import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(description='Development-only document-grouped NLI aggregation diagnostics; NEVER reads test data.')
 ap.add_argument('--candidates',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
 src=Path(args.candidates);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
 expected='b6724c628aebf5b112fe3c8ca74fb59eddfad1b0e1f899eaf9b300190a8dc41d'
 actual=sha(src)
 if actual!=expected:raise SystemExit('STOP: S46 candidate cache hash mismatch: '+actual)
 d=pd.read_csv(src)
 req={'claim_id','doc_id','doc_rank','p_entailment','p_contradiction','annotation_status_DEV_ONLY'}
 if not req.issubset(d.columns):raise SystemExit('Missing columns '+str(req-set(d.columns)))
 if len(d)!=4500 or d.claim_id.nunique()!=300:raise SystemExit('Unexpected cache dimensions')
 records=[]
 for cid,g in d.groupby('claim_id',sort=True):
  docs=[]
  for rank,sub in g.groupby('doc_rank',sort=True):
   if sub.doc_id.nunique()!=1:raise SystemExit('Nonunique document ID per rank')
   docs.append((int(rank),float(sub.p_entailment.mean()),float(sub.p_contradiction.mean()),int(sub.doc_id.iloc[0])))
  docs.sort(); ar=np.array([[x[1],x[2]] for x in docs]); margins=ar[:,0]-ar[:,1]
  top=ar[0]; equal=ar.mean(axis=0)
  # Three predefined diagnostics; no parameter tuning or provenance weights.
  policies={'top_doc_mean':top,'five_doc_equal_mean':equal,
            'five_doc_majority_vote':np.array([(margins>0).sum(),(margins<=0).sum()],float)}
  ann=g['annotation_status_DEV_ONLY'].unique()
  if len(ann)!=1:raise SystemExit('Inconsistent annotation status')
  label=str(ann[0]); annotated=label in ('SUPPORTS','REFUTES')
  rec={'claim_id':int(cid),'annotation_DEV_ONLY':label,'annotated_DEV_ONLY':annotated,
       'n_document_groups':len(docs),'unique_document_ids':len(set(x[3] for x in docs)),
       'doc_positive_votes':int((margins>0).sum()),'doc_negative_votes':int((margins<=0).sum()),
       'doc_margin_range':float(margins.max()-margins.min()),
       'cross_document_sign_disagreement':bool((margins>0).any() and (margins<0).any()),
       'top_doc_margin':float(top[0]-top[1]),'five_doc_margin':float(equal[0]-equal[1]),
       'top_doc_abstain_margin_015':bool(abs(top[0]-top[1])<0.15)}
  for k,p in policies.items():
   pred='SUPPORTS' if p[0]>p[1] else 'REFUTES'
   rec[k+'_prediction']=pred
   rec[k+'_correct_DEV_ONLY']=(pred==label) if annotated else None
  records.append(rec)
 r=pd.DataFrame(records);r.to_csv(out/'s76_document_group_diagnostics.csv',index=False)
 subset=r[r.annotated_DEV_ONLY].copy()
 metrics={}
 for name in ('top_doc_mean','five_doc_equal_mean','five_doc_majority_vote'):
  correct=subset[name+'_correct_DEV_ONLY'].astype(bool)
  metrics[name]={'correct':int(correct.sum()),'n':len(correct),'accuracy':float(correct.mean())}
 if metrics['top_doc_mean']['correct']!=125:raise SystemExit('Frozen S53 parity failure; STOP')
 baseline=subset['top_doc_mean_correct_DEV_ONLY'].astype(int).to_numpy()
 rng=np.random.default_rng(20261020);N=len(subset);inds=rng.integers(0,N,size=(10000,N))
 for name in ('five_doc_equal_mean','five_doc_majority_vote'):
  alt=subset[name+'_correct_DEV_ONLY'].astype(int).to_numpy();delta=alt-baseline
  metrics[name]['paired_delta_vs_top_doc']=float(delta.mean())
  metrics[name]['descriptive_claim_bootstrap_ci95']=list(map(float,np.quantile(delta[inds].mean(axis=1),[.025,.975])))
 abstain=subset['top_doc_abstain_margin_015'].astype(bool)
 retained=subset[~abstain]
 summary={'phase':'S76-S79','status':'EXPLORATORY_DEV_ONLY_NOT_EXTERNAL_VALIDATION',
          'candidate_sha256':actual,'total_claims':len(r),'annotated_binary_claims':len(subset),
          'document_group_ids_are_not_verified_independent_sources':True,
          'cross_document_sign_disagreement_claims_all':int(r.cross_document_sign_disagreement.sum()),
          'cross_document_sign_disagreement_claims_annotated':int(subset.cross_document_sign_disagreement.sum()),
          'distinct_doc_id_counts':{str(k):int(v) for k,v in r.unique_document_ids.value_counts().items()},
          'policy_comparison':metrics,
          'fixed_margin_abstention_diagnostic':{'threshold':0.15,'retained_annotated':len(retained),
           'abstained_annotated':int(abstain.sum()),'coverage':float(len(retained)/len(subset)),
           'selective_accuracy':float(retained.top_doc_mean_correct_DEV_ONLY.astype(bool).mean()) if len(retained) else None},
          'limitations':['All policies and metrics use the same previously inspected development set',
           'No unbiased generalization inference or novelty claim','Doc ID is not verified source provenance',
           'No NEI claim-level policy','Bootstrap intervals are descriptive, not selection-adjusted',
           'No sealed test content or external labels read','No new model inference; cached NLI scores only']}
 (out/'s79_development_experiment_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps({'parity':'PASS','comparison':metrics,'output':str(out)},indent=2))
if __name__=='__main__':main()
