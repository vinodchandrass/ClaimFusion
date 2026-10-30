import argparse,hashlib,json,pathlib
import numpy as np,pandas as pd

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def decision(g,mode):
 e=g.p_entailment.to_numpy(float);c=g.p_contradiction.to_numpy(float);n=g.p_neutral.to_numpy(float)
 if mode=='max_pair':
  i=np.argmax(np.maximum(e,c));score=e[i]-c[i]
 elif mode=='max_class':score=e.max()-c.max()
 elif mode=='mean_prob':score=e.mean()-c.mean()
 elif mode=='sum_prob':score=e.sum()-c.sum()
 elif mode=='top_lexical':
  i=np.argmax(g.lexical_score.to_numpy(float));score=e[i]-c[i]
 elif mode=='best_margin':
  d=e-c;i=np.argmax(abs(d));score=d[i]
 elif mode=='top_doc_mean':
  z=g[g.doc_rank==g.doc_rank.min()];score=(z.p_entailment-z.p_contradiction).mean()
 elif mode=='weighted_lexical':
  w=np.maximum(g.lexical_score.to_numpy(float),0)+1e-9;score=np.average(e-c,weights=w)
 else:raise ValueError(mode)
 return ('SUPPORTS' if score>0 else 'REFUTES'),float(score)
def bootstrap(base,other,y,B=2000,seed=49052):
 a=(base==y).astype(int);b=(other==y).astype(int);d=b-a;rng=np.random.default_rng(seed)
 sims=np.empty(B)
 for i in range(B):sims[i]=d[rng.integers(0,len(d),len(d))].mean()
 return {'delta':float(d.mean()),'ci95':[float(x) for x in np.quantile(sims,[.025,.975])], 'regressions':int(((a==1)&(b==0)).sum()),'rescues':int(((a==0)&(b==1)).sum())}
def main():
 p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--diagnostics',required=True);p.add_argument('--out',required=True);x=p.parse_args();o=pathlib.Path(x.out);o.mkdir(parents=True,exist_ok=True)
 d=pd.read_csv(x.candidates);q=pd.read_csv(x.diagnostics)
 req={'claim_id','doc_id','doc_rank','sentence_index','lexical_score','gold_sentence_DEV_ONLY','annotation_status_DEV_ONLY','p_contradiction','p_entailment','p_neutral'}
 assert req.issubset(d.columns),sorted(req-set(d.columns))
 assert d[['claim_id','doc_id','sentence_index']].duplicated().sum()==0,'duplicate candidates'
 assert d.groupby('claim_id').size().eq(15).all(),'expected 15 candidates per claim'
 assert d.claim_id.nunique()==300 and q.claim_id.nunique()==300
 assert d.groupby('claim_id').annotation_status_DEV_ONLY.nunique().eq(1).all()
 probs=d[['p_contradiction','p_entailment','p_neutral']].to_numpy(float)
 assert np.isfinite(probs).all() and ((probs>=0)&(probs<=1)).all()
 assert np.max(np.abs(probs.sum(axis=1)-1))<.002
 modes=['max_class','max_pair','mean_prob','sum_prob','top_lexical','best_margin','top_doc_mean','weighted_lexical']
 rows=[];diag=[];gold=[]
 for cid,g in d.groupby('claim_id',sort=True):
  actual=g.annotation_status_DEV_ONLY.iloc[0];r={'claim_id':cid,'actual_annotation':actual,'any_gold_sentence_retrieved':int(g.gold_sentence_DEV_ONLY.astype(int).any())}
  for m in modes:r[m],r[m+'_score']=decision(g,m)
  rows.append(r)
  gg=g[g.gold_sentence_DEV_ONLY.astype(int)==1]
  if actual in ('SUPPORTS','REFUTES'):
   diag.append({'claim_id':cid,'actual':actual,'gold_retrieved':len(gg),'max_entailment':g.p_entailment.max(),'max_contradiction':g.p_contradiction.max(),'mean_entailment':g.p_entailment.mean(),'mean_contradiction':g.p_contradiction.mean(),'contradiction_gt_entailment_candidates':int((g.p_contradiction>g.p_entailment).sum()),'top_doc_id':int(g.sort_values('doc_rank').doc_id.iloc[0])})
  if len(gg):
   for _,z in gg.iterrows():gold.append({'claim_id':cid,'actual':actual,'doc_id':z.doc_id,'sentence_index':z.sentence_index,'p_entailment':z.p_entailment,'p_contradiction':z.p_contradiction,'p_neutral':z.p_neutral,'gold_sentence_predicted':('SUPPORTS' if z.p_entailment>z.p_contradiction else 'REFUTES')})
 pred=pd.DataFrame(rows);pred.to_csv(o/'s49_policy_predictions.csv',index=False)
 pd.DataFrame(diag).to_csv(o/'s50_score_diagnostics.csv',index=False)
 pd.DataFrame(gold).to_csv(o/'s51_retrieved_gold_sentence_diagnostics.csv',index=False)
 valid=pred[pred.actual_annotation.isin(['SUPPORTS','REFUTES'])].copy();y=valid.actual_annotation.to_numpy()
 summary=[];base=valid.max_class.to_numpy()
 for m in modes:
  a=valid[m].to_numpy();res={'operator':m,'n':len(y),'correct':int((a==y).sum()),'accuracy':float((a==y).mean()),'predicted_supports':int((a=='SUPPORTS').sum()),'predicted_refutes':int((a=='REFUTES').sum())}
  for label in ['SUPPORTS','REFUTES']:
   mask=y==label;res['accuracy_'+label.lower()]=float((a[mask]==y[mask]).mean())
  res.update(bootstrap(base,a,y));summary.append(res)
 pd.DataFrame(summary).to_csv(o/'s52_operator_comparison.csv',index=False)
 strat=[]
 for subset,frame in [('gold_sentence_found',valid[valid.any_gold_sentence_retrieved==1]),('gold_sentence_missing',valid[valid.any_gold_sentence_retrieved==0])]:
  for m in modes:strat.append({'subset':subset,'operator':m,'n':len(frame),'accuracy':float((frame[m]==frame.actual_annotation).mean()) if len(frame) else None})
 pd.DataFrame(strat).to_csv(o/'s52_retrieval_strata.csv',index=False)
 gdf=pd.DataFrame(gold)
 report={'status':'DEVELOPMENT_ONLY_ABLATION_NOT_INDEPENDENT','claims':len(pred),'annotated_claims':len(valid),'retrieved_gold_sentence_rows':len(gdf),'gold_sentence_accuracy_on_retrieved_only':float((gdf.gold_sentence_predicted==gdf.actual).mean()) if len(gdf) else None,'baseline_max_class_accuracy':float((base==y).mean()),'baseline_matches_previous_100_of_188':bool(int((base==y).sum())==100),'inputs_sha256':{'candidates':sha(x.candidates),'diagnostics':sha(x.diagnostics)},'restrictions':['Operators compared post hoc on same development set: no winner may be claimed as externally validated','Gold annotations used for evaluation/stratification only; no gold evidence enters operational operators','Gold-sentence NLI restricted to retrieved gold sentences; not full oracle evaluation','UNANNOTATED excluded from binary accuracy and not treated as verified NEI','Paired bootstrap is claim-level and exploratory, not a preregistered test','No test files accessed']}
 (o/'s52_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'status':'PASS','baseline_parity':report['baseline_matches_previous_100_of_188'],'outputs':str(o),'operators':len(modes)},indent=2))
if __name__=='__main__':main()
