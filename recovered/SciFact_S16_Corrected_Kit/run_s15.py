import argparse, zipfile, io, json, hashlib, pathlib
import numpy as np, pandas as pd
from sklearn.metrics import f1_score, accuracy_score

def metric(y,p):return {'n':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=['CONTRADICT','SUPPORT'],average='macro',zero_division=0))}
def decision(d):return np.where(d.p_contradiction.to_numpy()>=d.p_entailment.to_numpy(),'CONTRADICT','SUPPORT')
def aggregate(s,method):
 p=s[['p_contradiction','p_entailment','p_neutral']].to_numpy()
 if method=='mean':v=p.mean(axis=0)
 elif method=='max_renorm':v=p.max(axis=0);v=v/v.sum()
 elif method=='noisy_or_renorm':v=1-np.prod(1-p,axis=0);v=v/v.sum()
 elif method=='most_decisive':v=p[np.argmax(np.abs(p[:,0]-p[:,1]))]
 else:raise ValueError(method)
 return v

def main():
 a=argparse.ArgumentParser();a.add_argument('--s14',required=True);a.add_argument('--out',required=True);a=a.parse_args()
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 z=zipfile.ZipFile(a.s14);raw=z.read('S14_report.json');report=json.loads(raw)
 if report['status']!='PASS_INFERENCE_AND_S7_PARITY':raise ValueError('S14 gate failed')
 pairs=pd.read_csv(io.BytesIO(z.read('S14_pair_predictions.csv')));sent=pd.read_csv(io.BytesIO(z.read('S14_sentence_predictions.csv')))
 keys=['split','claim_id','doc_id']; assert not pairs.duplicated(keys).any();assert not sent.duplicated(keys+['sentence_index']).any()
 assert set(pairs['split'])=={'calibration','dev_evaluation'} and set(sent['split'])=={'calibration','dev_evaluation'}
 assert len(pairs)==314 and len(sent)==546
 assert set(pairs.gold)=={'CONTRADICT','SUPPORT'}
 for df in [pairs,sent]:
  assert np.isfinite(df[['p_contradiction','p_entailment','p_neutral']].to_numpy()).all()
  assert np.allclose(df[['p_contradiction','p_entailment','p_neutral']].sum(axis=1),1,atol=1e-4)
 counts=sent.groupby(keys).size().rename('n').reset_index();merged=pairs.merge(counts,on=keys,validate='one_to_one');assert (merged.sentence_count==merged.n).all()
 assert not set(pairs[pairs['split']=='calibration'].claim_id)&set(pairs[pairs['split']=='dev_evaluation'].claim_id)
 # Freeze rule: highest calibration coverage among ceilings with calibration selective error <= 0.10 and >= 30 accepted; fallback no abstention.
 thresholds=[0.25,0.50,0.75,0.90,0.95,1.0];cal=pairs[pairs['split']=='calibration']; cp=decision(cal); candidates=[]
 for t in thresholds:
  mask=cal.p_neutral.to_numpy()<=t
  if mask.sum()<30:continue
  candidates.append({'threshold':t,'n':int(mask.sum()),'coverage':float(mask.mean()),'risk':float(np.mean(cp[mask]!=cal.gold.to_numpy()[mask]))})
 feasible=[x for x in candidates if x['risk']<=0.10]
 chosen=max(feasible,key=lambda x:(x['coverage'],x['threshold']))['threshold'] if feasible else 1.0
 # Baseline and ablations are prespecified, not tuned on dev.
 methods=['s7_frozen','mean','max_renorm','noisy_or_renorm','most_decisive']; rows=[];preds={}
 for split in ['calibration','dev_evaluation']:
  ps=pairs[pairs['split']==split].copy().sort_values(keys).reset_index(drop=True)
  ss=sent[sent['split']==split]
  for method in methods:
   d=ps.copy()
   if method!='s7_frozen':
    agg={k:aggregate(g,method) for k,g in ss.groupby(keys,sort=False)}
    arr=np.array([agg[tuple(x)] for x in d[keys].itertuples(index=False,name=None)])
    d[['p_contradiction','p_entailment','p_neutral']]=arr
   pred=decision(d); m=metric(d.gold,pred);m.update(split=split,method=method)
   mask=d.p_neutral.to_numpy()<=chosen
   m.update(threshold=chosen,coverage=float(mask.mean()),accepted=int(mask.sum()),selective_accuracy=float(np.mean(pred[mask]==d.gold.to_numpy()[mask])) if mask.any() else None)
   rows.append(m)
   if split=='dev_evaluation':preds[method]=pd.DataFrame({'claim_id':d.claim_id,'doc_id':d.doc_id,'gold':d.gold,'pred':pred,'accepted':mask})
 # paired cluster bootstrap, each method compared against S7 on identical dev keys, fixed seed.
 rng=np.random.default_rng(1515);base=preds['s7_frozen'];clusters=base.claim_id.unique();by={c:np.flatnonzero(base.claim_id.to_numpy()==c) for c in clusters};stats=[]
 for method in methods[1:]:
  cur=preds[method];assert (cur[keys[1:]].to_numpy()==base[keys[1:]].to_numpy()).all()
  diffs=[]
  for _ in range(1000):
   sampled=rng.choice(clusters,len(clusters),replace=True);ix=np.concatenate([by[c] for c in sampled]);y=base.gold.to_numpy()[ix]
   diffs.append(metric(y,cur.pred.to_numpy()[ix])['macro_f1']-metric(y,base.pred.to_numpy()[ix])['macro_f1'])
  d=np.array(diffs);stats.append({'method':method,'delta_macro_f1':metric(base.gold,cur.pred)['macro_f1']-metric(base.gold,base.pred)['macro_f1'],'bootstrap_ci95':[float(np.quantile(d,.025)),float(np.quantile(d,.975))],'resamples':1000})
 # deterministic perturbations: delete one sentence for groups of >=2; duplicate first sentence. No synthetic 'gold' provenance.
 perturb=[];ds=sent[sent['split']=='dev_evaluation'];baseline={tuple(k):aggregate(g,'mean') for k,g in ds.groupby(keys)}
 for kind in ['remove_first','duplicate_first']:
  for key,g in ds.groupby(keys):
   if kind=='remove_first' and len(g)<2:continue
   gg=g.iloc[1:] if kind=='remove_first' else pd.concat([g,g.iloc[[0]]],ignore_index=True)
   before=baseline[key];after=aggregate(gg,'mean')
   perturb.append({'kind':kind,'claim_id':key[1],'doc_id':key[2],'n_original':len(g),'pred_changed':bool((before[0]>=before[1])!=(after[0]>=after[1])),'delta_neutral':float(after[2]-before[2])})
 assert len(ds)==366, f'Expected 366 development sentence rows, got {len(ds)}'
 assert len(perturb)>0, 'Perturbation evaluation unexpectedly empty'
 expected_remove=sum(len(g)>=2 for _,g in ds.groupby(keys))
 expected_duplicate=ds.groupby(keys).ngroups
 assert len(perturb)==expected_remove+expected_duplicate
 pd.DataFrame(rows).to_csv(out/'S15_metrics.csv',index=False);pd.DataFrame(stats).to_csv(out/'S15_bootstrap.csv',index=False);pd.DataFrame(perturb).to_csv(out/'S15_perturbations.csv',index=False)
 summary={'status':'PASS_S16_CORRECTED_PERTURBATIONS','s14_sha256':hashlib.sha256(pathlib.Path(a.s14).read_bytes()).hexdigest(),'calibration_threshold_rule':'highest coverage with calibration risk <= 0.10, minimum 30 accepted, thresholds [0.25,0.50,0.75,0.90,0.95,1.0]; fallback 1.0','threshold_candidates':candidates,'chosen_threshold':chosen,'risk_target_met':bool(feasible),'perturbation_counts':{'remove_first':int(expected_remove),'duplicate_first':int(expected_duplicate)},'dev_tuning':False,'model_training':False,'bootstrap_unit':'claim_id','limitations':['S15 evaluated official development data; not independent test','No external source reliability ground truth','Aggregation comparisons use gold evidence sentences','Threshold selection from only 105 calibration pairs','Perturbations are synthetic sensitivity analyses, not evidence of real-world source dependence']}
 (out/'S15_report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
 print(json.dumps({'status':summary['status'],'chosen_threshold':chosen,'output':str(out)},indent=2))
if __name__=='__main__':main()
