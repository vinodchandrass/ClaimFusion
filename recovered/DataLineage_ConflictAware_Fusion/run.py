import argparse,csv,json,hashlib,math,random,statistics
from collections import defaultdict,Counter
from pathlib import Path

def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def label(v):
 v=str(v).strip().upper()
 return {'SUPPORT':'SUPPORTS','SUPPORTS':'SUPPORTS','REFUTES':'REFUTES','REFUTE':'REFUTES','CONTRADICT':'REFUTES','CONTRADICTION':'REFUTES'}.get(v)
def probs(r,prefix):
 z=[float(r[prefix+k]) for k in ('contradiction','entailment','neutral')]
 if any(not math.isfinite(v) or v<0 or v>1 for v in z) or abs(sum(z)-1)>0.002:raise ValueError('Invalid probability row')
 return z

def docs(rr,prefix):
 by=defaultdict(list)
 for r in rr:by[int(r['doc_rank'])].append(probs(r,prefix))
 out=[]
 for rank,pp in sorted(by.items()):
  p=sum(x[1] for x in pp)/len(pp);q=sum(x[0] for x in pp)/len(pp)
  out.append((rank,p-q,max(p,q),p,q))
 return out

def prediction(ds,kind,threshold=0,weight=0):
 top=ds[0][1];rest=ds[1:]
 if kind=='top':v=top
 elif kind=='mean':v=statistics.mean(d[1] for d in ds)
 elif kind=='rank':v=sum(d[1]/math.log2(d[0]+2) for d in ds)/sum(1/math.log2(d[0]+2) for d in ds)
 elif kind=='majority':v=sum(1 if d[1]>0 else -1 for d in ds)
 elif kind=='relevance':
  w=[1/(d[0]**weight) for d in ds];v=sum(d[1]*a for d,a in zip(ds,w))/sum(w)
 elif kind=='conflict_gate':
  if not rest:v=top
  else:
   others=statistics.mean(d[1] for d in rest)
   conflict=(top*others<0)
   # Allow lower-ranked evidence only when corroborating top document and its margin is weak.
   v=top if conflict or abs(top)>=threshold else (1-weight)*top+weight*others
 elif kind=='abstain':
  others=statistics.mean(d[1] for d in rest) if rest else top
  if abs(top)<threshold or (rest and top*others<0 and abs(top)<weight):return 'ABSTAIN'
  v=top
 else:raise ValueError(kind)
 return 'SUPPORTS' if v>0 else 'REFUTES'

def metric(y,p):
 keep=[i for i,z in enumerate(p) if z!='ABSTAIN'];n=len(y)
 correct=sum(y[i]==p[i] for i in keep)
 f=[]
 for c in ('SUPPORTS','REFUTES'):
  tp=sum(y[i]==c and p[i]==c for i in keep)
  pred=sum(p[i]==c for i in keep); actual=sum(y[i]==c for i in keep)
  prec=tp/pred if pred else 0;rec=tp/actual if actual else 0
  f.append(2*prec*rec/(prec+rec) if prec+rec else 0)
 return {'n':n,'coverage':len(keep)/n,'accuracy_covered':correct/len(keep) if keep else None,'accuracy_all_with_abstain_as_error':correct/n,'macro_f1_covered':sum(f)/2,'errors':len(keep)-correct,'abstentions':n-len(keep)}

def bootstrap(y,a,b,B,seed):
 rng=random.Random(seed);n=len(y);d=[]
 for _ in range(B):
  ix=[rng.randrange(n) for _ in range(n)]
  d.append(sum((a[i]==y[i])-(b[i]==y[i]) for i in ix)/n)
 d.sort();return {'delta':sum((a[i]==y[i])-(b[i]==y[i]) for i in range(n))/n,'ci95':[d[int(.025*(B-1))],d[int(.975*(B-1))]]}

def main():
 ap=argparse.ArgumentParser(description='Development-only nested claim-disjoint fusion comparison; no model inference')
 ap.add_argument('--candidates',required=True);ap.add_argument('--cache',required=True);ap.add_argument('--gold',required=True);ap.add_argument('--out',required=True)
 ap.add_argument('--folds',type=int,default=5);ap.add_argument('--bootstrap',type=int,default=2000)
 a=ap.parse_args()
 for x in (a.candidates,a.cache,a.gold):
  if 'test' in Path(x).name.lower():raise ValueError('Test-named inputs forbidden')
 original=read(a.candidates);cache=read(a.cache);g=read(a.gold)
 if len(original)!=len(cache) or not original:raise ValueError('Candidate/cache count mismatch')
 keys=('claim_id','claim','doc_id','doc_rank','sentence')
 for i,(x,z) in enumerate(zip(original,cache)):
  if any(x[k]!=z[k] for k in keys):raise ValueError(f'Cache identity mismatch row {i}')
  probs(x,'p_');probs(z,'adapted_p_')
 gold={}
 for x in g:
  k=str(x['claim_id']);v=label(x['label'])
  if not v or (k in gold and gold[k]!=v):raise ValueError('Bad or inconsistent gold label '+k)
  gold[k]=v
 by_o=defaultdict(list);by_c=defaultdict(list)
 for x,z in zip(original,cache):by_o[str(x['claim_id'])].append(x);by_c[str(z['claim_id'])].append(z)
 ids=sorted(set(by_o)&set(gold))
 if len(ids)<a.folds*4:raise ValueError('Insufficient eligible claims')
 y=[gold[k] for k in ids];D=[docs(by_c[k],'adapted_p_') for k in ids];F=[docs(by_o[k],'p_') for k in ids]
 # Fixed policy comparisons (no selection)
 fixed={name:[prediction(ds,name) for ds in D] for name in ('top','mean','rank','majority')}
 frozen=[prediction(ds,'top') for ds in F]
 # Claim-stratified deterministic folds, labels only used for balanced partitioning.
 rng=random.Random(20261021);fold=[None]*len(ids)
 for c in ('SUPPORTS','REFUTES'):
  ix=[i for i,v in enumerate(y) if v==c];rng.shuffle(ix)
  for j,i in enumerate(ix):fold[i]=j%a.folds
 # Predeclared hyperparameter candidates, fit ONLY on other folds.
 grid=[('top',0,0)]
 grid += [('relevance',0,w) for w in (0.5,1,2,4)]
 grid += [('conflict_gate',t,w) for t in (0.1,0.2,0.4,0.6) for w in (0.25,0.5,0.75)]
 # Selection is accuracy-first; ties favor top baseline, then less complexity.
 oof=[None]*len(ids);selected=[]
 for f in range(a.folds):
  train=[i for i in range(len(ids)) if fold[i]!=f];valid=[i for i in range(len(ids)) if fold[i]==f]
  scored=[]
  for j,(kind,t,w) in enumerate(grid):
   pp=[prediction(D[i],kind,t,w) for i in train]
   score=sum(p==y[i] for p,i in zip(pp,train))/len(train)
   scored.append((score,-j,j))
  _,_,chosen=max(scored);kind,t,w=grid[chosen]
  for i in valid:oof[i]=prediction(D[i],kind,t,w)
  selected.append({'fold':f,'training_claims':len(train),'held_out_claims':len(valid),'selected':{'kind':kind,'threshold':t,'weight':w},'training_accuracy':max(scored)[0]})
 # Abstention is evaluated as fixed thresholds only; no tuning on all dev labels.
 abst={str(t):metric(y,[prediction(ds,'abstain',t,t) for ds in D]) for t in (0.1,0.2,0.4,0.6)}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 report={'status':'DEVELOPMENT_ONLY_CROSSFITTED_NOT_INDEPENDENT_VALIDATION','data_sha256':{k:sha(getattr(a,k)) for k in ('candidates','cache','gold')},'n_claims':len(ids),'class_counts':dict(Counter(y)),'n_sentences':len(original),'test_accessed':False,'design':{'folds':a.folds,'seed':20261021,'selection':'accuracy on other folds; tie favors simplest','claim_disjoint':True,'frozen_retrieval':True,'no_inference':True,'no_new_training':True},'metrics':{'frozen_top':metric(y,frozen),'adapted_fixed':{k:metric(y,v) for k,v in fixed.items()},'adapted_crossfitted_selection':metric(y,oof),'fixed_abstention_diagnostics':abst},'paired_bootstrap':{'crossfitted_minus_adapted_top':bootstrap(y,oof,fixed['top'],a.bootstrap,20261021),'adapted_top_minus_frozen_top':bootstrap(y,fixed['top'],frozen,a.bootstrap,20261022)},'fold_selection':selected,'limitations':['SciFact development-only: nested claim-disjoint selection is not an independent test','Cached model weights not cryptographically bound by this script; verify checkpoint provenance separately','Claims may share retrieved documents; claim-level bootstrap does not account for document-level dependence','Exploratory candidate policy family; repeated development can bias results','Abstention thresholds are descriptive only; not validated for deployment']}
 (out/'conflict_fusion_report_DEV_ONLY.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 with (out/'conflict_fusion_predictions_DEV_ONLY.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['claim_id','fold','gold','frozen_top','adapted_top','adapted_mean','adapted_rank','adapted_majority','crossfitted'])
  for i,k in enumerate(ids):w.writerow([k,fold[i],y[i],frozen[i],*[fixed[n][i] for n in ('top','mean','rank','majority')],oof[i]])
 print('Complete:',out/'conflict_fusion_report_DEV_ONLY.json')
 print('Crossfitted:',report['metrics']['adapted_crossfitted_selection'])
 print('Adapted top:',report['metrics']['adapted_fixed']['top'])
if __name__=='__main__':main()
