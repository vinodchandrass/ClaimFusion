#!/usr/bin/env python3
"""CPU-only, development-only HEALTHVER NLI model comparison."""
import argparse,csv,hashlib,json,math,os,random,time
from collections import Counter,defaultdict
from pathlib import Path
LABELS=['Refutes','Supports','Neutral']
MODELS={'minilm':'cross-encoder/nli-MiniLM2-L6-H768','distilroberta':'cross-encoder/nli-distilroberta-base'}

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read_csv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,rows,fields):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def norm(x):return ' '.join(x.casefold().split())
def canonical(s):
 s=str(s).strip().lower()
 for label,aliases in [('Refutes',['refutes','refute','contradiction','contradict','contradictory']),('Supports',['supports','support','entailment','entails','entail']),('Neutral',['neutral','nei','not enough info'])]:
  if s in aliases:return label
 raise ValueError('Unknown label: '+repr(s))
def metrics(rows):
 mat=[[0]*3 for _ in range(3)]; probs=[]
 for r in rows:
  t=LABELS.index(r['label']);p=LABELS.index(r['pred_label']);mat[t][p]+=1
  probs.append([float(r['p_refutes']),float(r['p_supports']),float(r['p_neutral'])])
 n=len(rows); per={};f1=[]
 for i,k in enumerate(LABELS):
  tp=mat[i][i];sup=sum(mat[i]);pred=sum(row[i] for row in mat)
  precision=tp/pred if pred else 0;recall=tp/sup if sup else 0
  f=2*precision*recall/(precision+recall) if precision+recall else 0
  per[k]={'support':sup,'precision':precision,'recall':recall,'f1':f};f1.append(f)
 bins=[{'n':0,'conf_sum':0.,'correct':0} for _ in range(10)]
 brier=0
 for r,p in zip(rows,probs):
  ti=LABELS.index(r['label']);pi=LABELS.index(r['pred_label']);conf=max(p)
  brier+=sum((v-(j==ti))**2 for j,v in enumerate(p))
  b=bins[min(9,int(conf*10))];b['n']+=1;b['conf_sum']+=conf;b['correct']+=int(ti==pi)
 ece=sum((b['n']/n)*abs(b['conf_sum']/b['n']-b['correct']/b['n']) for b in bins if b['n'])
 return {'n':n,'accuracy':sum(mat[i][i] for i in range(3))/n,'macro_f1':sum(f1)/3,'per_class':per,'confusion_actual_rows_predicted_columns':mat,'brier_multiclass':brier/n,'ece_10_bin':ece,'predicted_counts':dict(Counter(r['pred_label'] for r in rows))}
def bootstrap(base,others,iterations,seed):
 rng=random.Random(seed);groups=defaultdict(list)
 for i,r in enumerate(base):groups[norm(r['claim'])].append(i)
 keys=list(groups);out={k:[] for k in others}
 for _ in range(iterations):
  inds=[i for key in rng.choices(keys,k=len(keys)) for i in groups[key]]
  b=sum(base[i]['pred_label']==base[i]['label'] for i in inds)/len(inds)
  for k,rows in others.items():
   a=sum(rows[i]['pred_label']==rows[i]['label'] for i in inds)/len(inds)
   out[k].append(a-b)
 return {k:{'delta_accuracy':sum(v)/len(v),'ci95':[sorted(v)[int(.025*(len(v)-1))],sorted(v)[int(.975*(len(v)-1))]]} for k,v in out.items()}
def check_label_map(config):
 m=config.id2label
 if len(m)!=3:raise ValueError('Expected exactly 3 labels, got '+str(m))
 ids={int(i):canonical(label) for i,label in m.items()}
 if set(ids.values())!=set(LABELS):raise ValueError('Model label map incomplete: '+str(ids))
 return ids

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',required=True,help='Previously uploaded HEALTHVER pair prediction CSV');ap.add_argument('--out',required=True);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--max-length',type=int,default=256);ap.add_argument('--bootstrap',type=int,default=1000);ap.add_argument('--models',nargs='+',choices=list(MODELS),default=list(MODELS));ap.add_argument('--smoke',action='store_true',help='First 24 pairs only; no full comparison claims');a=ap.parse_args()
 if any(x in str(a.baseline).lower() for x in ('test.csv','claims_test','healthver_test')):raise ValueError('Sealed test input prohibited')
 if a.batch_size<1 or a.threads<1 or a.bootstrap<1:raise ValueError('Invalid positive integer argument')
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 base=read_csv(a.baseline)
 required={'id','claim','evidence','label','pred_label','p_refutes','p_supports','p_neutral'}
 if not base or not required.issubset(base[0]):raise ValueError('Baseline CSV missing columns: '+str(required-set(base[0]) if base else required))
 for r in base:
  r['label']=canonical(r['label']);r['pred_label']=canonical(r['pred_label'])
  ps=[float(r['p_'+s.lower()]) for s in LABELS]
  if not all(math.isfinite(p) and 0<=p<=1 for p in ps) or abs(sum(ps)-1)>0.005:raise ValueError('Invalid baseline probabilities')
  if LABELS[ps.index(max(ps))]!=r['pred_label']:raise ValueError('Baseline prediction/probability mismatch')
 if len(base)!=1917:raise ValueError('Expected 1917 frozen HEALTHVER dev pairs, got '+str(len(base)))
 if a.smoke:base=base[:24]
 try:
  import torch
  from transformers import AutoConfig,AutoTokenizer,AutoModelForSequenceClassification
 except ImportError as e:raise SystemExit('Install dependencies: python -m pip install -r requirements.txt\n'+str(e))
 torch.set_num_threads(a.threads)
 results={};model_meta={};fields=['id','claim','evidence','label','pred_label','p_refutes','p_supports','p_neutral']
 for key in a.models:
  repo=MODELS[key];p=out/f'{key}_predictions_DEV_ONLY.csv'
  if p.exists():
   pred=read_csv(p)
   if len(pred)!=len(base) or any((r['id'],r['claim'],r['evidence'],r['label'])!=(s['id'],s['claim'],s['evidence'],s['label']) for r,s in zip(pred,base)):
    raise ValueError('Existing checkpoint mismatch: '+str(p))
   print('Reusing verified prediction checkpoint:',p,flush=True)
  else:
   print('Loading CPU model:',repo,flush=True)
   config=AutoConfig.from_pretrained(repo);idmap=check_label_map(config)
   tokenizer=AutoTokenizer.from_pretrained(repo);model=AutoModelForSequenceClassification.from_pretrained(repo,use_safetensors=True).to('cpu').eval()
   # Correct NLI direction: evidence is premise; claim is hypothesis.
   pred=[];start=time.time()
   with torch.inference_mode():
    for j in range(0,len(base),a.batch_size):
     batch=base[j:j+a.batch_size]
     toks=tokenizer([r['evidence'] for r in batch],[r['claim'] for r in batch],padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
     logits=model(**toks).logits
     probs=torch.softmax(logits.float(),dim=-1).cpu().tolist()
     for r,pp in zip(batch,probs):
      by={idmap[i]:float(pp[i]) for i in range(3)};pr=max(LABELS,key=lambda x:by[x])
      pred.append({'id':r['id'],'claim':r['claim'],'evidence':r['evidence'],'label':r['label'],'pred_label':pr,**{'p_'+k.lower():by[k] for k in LABELS}})
     if (j//a.batch_size)%25==0:print(f'  {min(j+len(batch),len(base))}/{len(base)} pairs',flush=True)
   write_csv(p,pred,fields);model_meta[key]={'model':repo,'seconds':round(time.time()-start,2),'id2label':idmap}
   del model,tokenizer
  results[key]=pred
  model_meta.setdefault(key,{'model':repo,'reused_checkpoint':True})
 baseline_metrics=metrics(base)
 allmetrics={'frozen_baseline':baseline_metrics,'always_neutral':metrics([{**r,'pred_label':'Neutral','p_refutes':0.,'p_supports':0.,'p_neutral':1.} for r in base])}
 for k,v in results.items():allmetrics[k]=metrics(v)
 paired=bootstrap(base,results,a.bootstrap,20261020)
 report={'status':'SMOKE_ONLY' if a.smoke else 'DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION','dataset':'HEALTHVER dev evidence pairs','n':len(base),'unique_normalized_claims':len({norm(r['claim']) for r in base}),'baseline_csv_sha256':sha(a.baseline),'input_order':'evidence_premise__claim_hypothesis','max_length':a.max_length,'threads':a.threads,'model_metadata':model_meta,'metrics':allmetrics,'paired_claim_cluster_bootstrap_accuracy_difference_vs_frozen_baseline':paired,'limitations':['Selection on HEALTHVER dev is not independent evaluation','Evidence strings are heavily reused across HEALTHVER train/dev; claim bootstrap does not account for all evidence dependence','This evaluates NLI pair classification, not document-level fusion','No sealed test file accessed','Cross-domain comparison is not proof of methodological novelty']}
 dest=out/'comparison_report_DEV_ONLY.json';dest.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'report':str(dest),'scores':{k:{'accuracy':round(v['accuracy'],4),'macro_f1':round(v['macro_f1'],4),'supports_recall':round(v['per_class']['Supports']['recall'],4)} for k,v in allmetrics.items()}},indent=2))
if __name__=='__main__':main()
