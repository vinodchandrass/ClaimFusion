import argparse,csv,hashlib,json,math,random,time
from collections import defaultdict,Counter
from pathlib import Path

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def label(x):
 x=str(x or '').strip().upper()
 if x in ('SUPPORTS','SUPPORT','ENTAILMENT'):return 'SUPPORTS'
 if x in ('REFUTES','REFUTE','CONTRADICTION'):return 'REFUTES'
 return None
def score(cla):
 # evidence premise, claim hypothesis; logit-based probabilities [contradiction,entailment,neutral]
 import torch
 with torch.no_grad():
  out=cla(**batch).logits
 return torch.softmax(out,dim=-1).cpu().tolist()
def decide(es,policy):
 docs=defaultdict(list)
 for r in es:docs[int(r['doc_rank'])].append(r)
 ds=[]
 for rank,rr in sorted(docs.items()):
  p=sum(float(x['p_entailment']) for x in rr)/len(rr)
  q=sum(float(x['p_contradiction']) for x in rr)/len(rr)
  ds.append((rank,p-q,p,q))
 if not ds:raise ValueError('No candidate documents')
 if policy=='top_document':v=ds[0][1]
 elif policy=='five_doc_mean':v=sum(x[1] for x in ds)/len(ds)
 elif policy=='rank_discounted':v=sum(x[1]/math.log2(x[0]+2) for x in ds)/sum(1/math.log2(x[0]+2) for x in ds)
 elif policy=='majority':v=sum(1 if x[1]>0 else -1 for x in ds)
 else:raise ValueError(policy)
 return 'SUPPORTS' if v>0 else 'REFUTES'
def metrics(y,p):
 cm={a:{b:0 for b in ('SUPPORTS','REFUTES')} for a in ('SUPPORTS','REFUTES')}
 for a,b in zip(y,p):cm[a][b]+=1
 f=[]
 for k in cm:
  tp=cm[k][k];sup=sum(cm[k].values()); pred=sum(cm[a][k] for a in cm)
  precision=tp/pred if pred else 0;recall=tp/sup if sup else 0
  f.append(2*precision*recall/(precision+recall) if precision+recall else 0)
 return {'n':len(y),'accuracy':sum(a==b for a,b in zip(y,p))/len(y),'macro_f1':sum(f)/2,'confusion':cm}
def main():
 ap=argparse.ArgumentParser(description='SciFact development-only cross-domain NLI and fixed fusion comparisons')
 ap.add_argument('--candidates',required=True);ap.add_argument('--gold',help='Optional separate CSV with claim_id,label. If omitted, use unambiguous gold_sentence_DEV_ONLY in candidates')
 ap.add_argument('--model',required=True,help='Local trained checkpoint directory');ap.add_argument('--out',required=True)
 ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--threads',type=int,default=4)
 ap.add_argument('--max-length',type=int,default=192);ap.add_argument('--bootstrap',type=int,default=2000)
 ap.add_argument('--cache-only',action='store_true',help='Generate prediction cache without gold evaluation')
 ap.add_argument('--reuse-cache',action='store_true',help='Reuse existing adapted predictions CSV after row-by-row input verification; only use with same model checkpoint and max-length')
 a=ap.parse_args()
 cp=Path(a.candidates);mp=Path(a.model);op=Path(a.out);op.mkdir(parents=True,exist_ok=True)
 if any('test' in p.name.lower() for p in [cp]+([Path(a.gold)] if a.gold else [])):
  raise ValueError('Test-like input filenames are prohibited')
 data=rows(cp)
 required={'claim_id','claim','doc_id','doc_rank','sentence','p_contradiction','p_entailment','p_neutral'}
 if not data or required-set(data[0]):raise ValueError('Candidates missing columns: '+str(required-set(data[0]) if data else required))
 if any(int(r['doc_rank'])<1 for r in data):raise ValueError('doc_rank must start at 1')
 claims=defaultdict(list)
 for r in data:claims[str(r['claim_id'])].append(r)
 gold={}
 if a.gold:
  gr=rows(a.gold)
  for r in gr:
   k=str(r['claim_id']);v=label(r.get('label'))
   if v is None:raise ValueError('Gold label must be SUPPORTS or REFUTES')
   if k in gold and gold[k]!=v:raise ValueError('Conflicting gold labels for '+k)
   gold[k]=v
 elif 'gold_sentence_DEV_ONLY' in data[0]:
  for k,rr in claims.items():
   vals={label(r.get('gold_sentence_DEV_ONLY')) for r in rr if label(r.get('gold_sentence_DEV_ONLY'))}
   if len(vals)==1:gold[k]=vals.pop()
   elif len(vals)>1:raise ValueError('Conflicting candidate gold labels for '+k)
 if not a.cache_only and not gold:
  raise ValueError('No binary gold labels inferred. Supply --gold path/to/dev_gold.csv (claim_id,label), or use --cache-only')
 start=time.time(); predictions=[]
 cache_path=op/'adapted_scifact_candidates_DEV_ONLY.csv'
 if a.reuse_cache:
  if not cache_path.is_file():raise ValueError('Prediction cache not found: '+str(cache_path))
  cached=rows(cache_path)
  if len(cached)!=len(data):raise ValueError('Cache row count differs from input: STOP')
  check_fields=('claim_id','claim','doc_id','doc_rank','sentence')
  for i,(original,old) in enumerate(zip(data,cached)):
   if any(str(original[k])!=str(old[k]) for k in check_fields):
    raise ValueError(f'Prediction cache mismatch at row {i}: STOP')
   try:
    pp=[float(old['adapted_p_'+k]) for k in ('contradiction','entailment','neutral')]
   except (KeyError,ValueError,TypeError) as e:raise ValueError(f'Invalid cache probabilities at row {i}') from e
   if not all(math.isfinite(v) and 0<=v<=1 for v in pp) or abs(sum(pp)-1)>0.001:
    raise ValueError(f'Invalid probability distribution at row {i}')
   z=dict(original)
   z.update({'p_contradiction':pp[0],'p_entailment':pp[1],'p_neutral':pp[2]})
   predictions.append(z)
  print(f'Reused and verified {len(predictions)} cached predictions. Confirm checkpoint unchanged since cache generation.',flush=True)
 else:
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  torch.set_num_threads(a.threads)
  tok=AutoTokenizer.from_pretrained(str(mp),local_files_only=True)
  model=AutoModelForSequenceClassification.from_pretrained(str(mp),local_files_only=True).eval().to('cpu')
  mapping={int(k):str(v).lower() for k,v in model.config.id2label.items()}
  if len(mapping)!=3 or any(x not in mapping[i] for i,x in enumerate(('contradiction','entailment','neutral'))):
   raise ValueError('Unexpected NLI label order: '+str(mapping))
  for j in range(0,len(data),a.batch_size):
   rr=data[j:j+a.batch_size]
   batch=tok([r['sentence'] for r in rr],[r['claim'] for r in rr],truncation=True,padding=True,max_length=a.max_length,return_tensors='pt')
   with torch.no_grad():pp=torch.softmax(model(**batch).logits,dim=-1).tolist()
   for r,p in zip(rr,pp):
    z=dict(r);z.update({'p_contradiction':p[0],'p_entailment':p[1],'p_neutral':p[2]});predictions.append(z)
   if (j//a.batch_size)%100==0:print(f'Inference {min(j+len(rr),len(data))}/{len(data)}',flush=True)
  with cache_path.open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(data[0])+['adapted_p_contradiction','adapted_p_entailment','adapted_p_neutral'],extrasaction='ignore');w.writeheader()
   for old,new in zip(data,predictions):
    z=dict(old);z.update({'adapted_p_contradiction':new['p_contradiction'],'adapted_p_entailment':new['p_entailment'],'adapted_p_neutral':new['p_neutral']});w.writerow(z)
 adapted=defaultdict(list)
 for r in predictions:adapted[str(r['claim_id'])].append(r)
 policies=('top_document','five_doc_mean','rank_discounted','majority')
 report={'status':'DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION','source_sha256':digest(cp),'model_config_sha256':digest(mp/'config.json'),
         'model_path':str(mp),'candidate_sentences':len(data),'claims':len(claims),'gold_claims':len(gold),'settings':vars(a),
         'inference_seconds':round(time.time()-start,2),'test_accessed':False,'policies':{},'paired_bootstrap':{},
         'limitations':['SciFact dev only; no test access','Frozen baseline originally selected on SciFact development','Retrieval fixed; NLI checkpoint trained on HEALTHVER train','Document-level aggregation assumes doc IDs, not independent sources','No calibrated abstention threshold or novel fusion rule tested']}
 if gold and not a.cache_only:
  eligible=sorted(k for k in claims if k in gold)
  y=[gold[k] for k in eligible]
  outputs={}
  for name,source in (('frozen',claims),('adapted',adapted)):
   for pol in policies:
    key=name+'__'+pol
    outputs[key]=[decide(source[k],pol) for k in eligible]
    report['policies'][key]=metrics(y,outputs[key])
  rng=random.Random(20261021)
  # Compare models with identical fusion policy, and policies within adapted model.
  comparisons=[('adapted__'+p,'frozen__'+p) for p in policies]+[(f'adapted__{p}','adapted__top_document') for p in policies if p!='top_document']
  for left,right in comparisons:
   d=sum(outputs[left][i]==y[i] for i in range(len(y)))/len(y)-sum(outputs[right][i]==y[i] for i in range(len(y)))/len(y)
   boots=[]
   for _ in range(a.bootstrap):
    ids=[rng.randrange(len(y)) for _ in y]
    boots.append(sum(outputs[left][i]==y[i] for i in ids)/len(ids)-sum(outputs[right][i]==y[i] for i in ids)/len(ids))
   boots.sort();ci=[boots[int(.025*(len(boots)-1))],boots[int(.975*(len(boots)-1))]] if boots else None
   report['paired_bootstrap'][left+' minus '+right]={'accuracy_delta':d,'ci95':ci}
 with (op/'crossdomain_fusion_report_DEV_ONLY.json').open('w',encoding='utf-8') as f:json.dump(report,f,indent=2)
 print('Complete:',op/'crossdomain_fusion_report_DEV_ONLY.json',flush=True)
if __name__=='__main__':main()
