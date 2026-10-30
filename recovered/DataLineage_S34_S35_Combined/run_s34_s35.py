import argparse,csv,json,hashlib,collections,random,math
from pathlib import Path
import numpy as np

def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def decide(s,r,t=.5):
 return 'SUPPORTS' if s>=t and s>r else 'REFUTES' if r>=t and r>s else 'NOT ENOUGH INFO'
def evaluate(gold,pred):
 labels=['SUPPORTS','REFUTES']; n=len(gold); c=collections.Counter(zip(gold,pred)); fs=[]
 for lab in labels:
  tp=c[lab,lab];fp=sum(c[x,lab] for x in labels if x!=lab);fn=sum(c[lab,x] for x in ['SUPPORTS','REFUTES','NOT ENOUGH INFO'] if x!=lab)
  fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return {'n':n,'accuracy':sum(a==b for a,b in zip(gold,pred))/n,'macro_f1_2class':sum(fs)/2,'abstention_rate':sum(x=='NOT ENOUGH INFO' for x in pred)/n,'selective_accuracy':sum(a==b for a,b in zip(gold,pred) if b!='NOT ENOUGH INFO')/max(1,sum(b!='NOT ENOUGH INFO' for b in pred))}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default=r'D:\DataLineage');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--bootstrap',type=int,default=2000);a=ap.parse_args()
 base=Path(a.base); source=base/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv';model=base/'S7_NLI_Model';out=base/'S33_S35_Validation_Output'/'s34_s35_dev';out.mkdir(parents=True,exist_ok=True)
 with source.open(encoding='utf-8-sig',newline='') as f: rows=[r for r in csv.DictReader(f) if r['split']=='dev']
 claims={}; pairs={}
 for r in rows:
  cid=r['claim_id'];d=claims.setdefault(cid,{'gold':r['gold_label'],'claim':r['claim'],'groups':collections.defaultdict(list)})
  if d['gold']!=r['gold_label'] or d['claim']!=r['claim']:raise ValueError('claim conflict')
  if r['resolved']=='1' and r['evidence_text'].strip():
   k=(cid,r['page_id'],r['sentence_index']);pairs[k]=(r['evidence_text'],r['claim']);d['groups'][r['group_index']].append(k)
 keys=sorted(pairs); cache=out/'pair_probabilities.csv'; scores={}
 if cache.exists():
  with cache.open(newline='',encoding='utf-8') as f:
   for r in csv.DictReader(f):scores[(r['claim_id'],r['page_id'],r['sentence_index'])]=[float(r['p_contradiction']),float(r['p_entailment']),float(r['p_neutral'])]
  if set(scores)!=set(keys):raise SystemExit('BLOCKED: cached pair keys mismatch; do not overwrite cache')
 else:
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True);net=AutoModelForSequenceClassification.from_pretrained(str(model),local_files_only=True)
  if {int(k):str(v).lower() for k,v in net.config.id2label.items()}!={0:'contradiction',1:'entailment',2:'neutral'}:raise SystemExit('BLOCKED: NLI label mapping')
  device='cuda' if torch.cuda.is_available() else 'cpu';net.to(device).eval()
  with cache.open('w',newline='',encoding='utf-8') as f:
   w=csv.writer(f);w.writerow(['claim_id','page_id','sentence_index','p_contradiction','p_entailment','p_neutral'])
   for i in range(0,len(keys),a.batch_size):
    chunk=keys[i:i+a.batch_size];x=tok([pairs[k][0] for k in chunk],[pairs[k][1] for k in chunk],padding=True,truncation=True,max_length=384,return_tensors='pt').to(device)
    with torch.inference_mode(): ps=net(**x).logits.softmax(-1).cpu().tolist()
    for k,p in zip(chunk,ps):scores[k]=p;w.writerow([*k,*p])
    if i%800==0:print('inferred',min(i+a.batch_size,len(keys)),'/',len(keys),flush=True)
 methods=['s33_conjunctive','max_pair','mean_pair','mean_group','max_group_mean']; records=[]
 for cid,d in sorted(claims.items()):
  gs=[[scores[k] for k in ks] for ks in d['groups'].values() if ks and all(k in scores for k in ks)]
  if not gs:continue
  flat=[p for g in gs for p in g]; unique=list({k:scores[k] for ks in d['groups'].values() for k in ks if k in scores}.values())
  # S33: min entailment, max contradiction within each group; max across complete groups
  gvals=[(min(p[1] for p in g),max(p[0] for p in g)) for g in gs]
  candidates={'s33_conjunctive':(max(x[0] for x in gvals),max(x[1] for x in gvals)),
   'max_pair':(max(p[1] for p in unique),max(p[0] for p in unique)),
   'mean_pair':(float(np.mean([p[1] for p in unique])),float(np.mean([p[0] for p in unique]))),
   'mean_group':(float(np.mean([x[0] for x in gvals])),float(np.mean([x[1] for x in gvals]))),
   'max_group_mean':(max(np.mean([p[1] for p in g]) for g in gs),max(np.mean([p[0] for p in g]) for g in gs))}
  rec={'claim_id':cid,'gold':d['gold'],'n_complete_groups':len(gs),'n_unique_pairs':len(unique)}
  for m,(s,r) in candidates.items():rec[m]=decide(s,r)
  records.append(rec)
 with (out/'ablation_claim_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 gold=[r['gold'] for r in records];results={m:evaluate(gold,[r[m] for r in records]) for m in methods}
 rng=np.random.default_rng(20261019);n=len(records);b=min(max(a.bootstrap,100),10000);deltas={m:[] for m in methods[1:]};basepred=np.array([r[methods[0]] for r in records]);g=np.array(gold)
 for _ in range(b):
  ix=rng.integers(0,n,n);ref=np.mean(basepred[ix]==g[ix])
  for m in methods[1:]:
   p=np.array([r[m] for r in records]);deltas[m].append(float(np.mean(p[ix]==g[ix])-ref))
 comparisons={m:{'accuracy_delta_vs_s33':results[m]['accuracy']-results['s33_conjunctive']['accuracy'],'paired_bootstrap_95ci':list(map(float,np.quantile(deltas[m],[.025,.975]))),'discordant_s33_correct_alt_wrong':sum(r['s33_conjunctive']==r['gold'] and r[m]!=r['gold'] for r in records),'discordant_s33_wrong_alt_correct':sum(r['s33_conjunctive']!=r['gold'] and r[m]==r['gold'] for r in records)} for m in methods[1:]}
 previous=base/'S33_S35_Validation_Output'/'dev_execution'/'dev_claim_predictions.csv'; parity={'checked':False}
 if previous.exists():
  with previous.open(newline='',encoding='utf-8') as f: old={r['claim_id']:r['decision'] for r in csv.DictReader(f)}
  parity={'checked':True,'same_claim_ids':set(old)=={r['claim_id'] for r in records},'mismatch_count':sum(old.get(r['claim_id'])!=r['s33_conjunctive'] for r in records)}
 report={'status':'EXPLORATORY_DEV_ONLY_NOT_CONFIRMATORY','source_sha256':digest(source),'model_directory':str(model),'pair_cache_sha256':digest(cache),'unique_pairs':len(keys),'eligible_claims':n,'methods':results,'paired_comparisons':comparisons,'s33_parity':parity,'caveats':['Oracle gold evidence only; no retrieval evaluation','No NEI gold claims in input','Pair aggregation comparators are exploratory and not provenance-aware fusion','SciFact independent validation requires exposure attestation','FEVER test not accessed','Pair cache is generated with frozen local model; verify weights and tokenizer hashes separately']}
 (out/'s34_s35_dev_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'status':report['status'],'unique_pairs':len(keys),'eligible_claims':n,'s33_parity':parity,'methods':results},indent=2))
if __name__=='__main__':main()
