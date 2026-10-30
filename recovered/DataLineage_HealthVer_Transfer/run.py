import argparse,csv,hashlib,json,collections,random,statistics,math
from pathlib import Path
EXPECTED='a2120d746171320b54f80dc344b8c707378dd79354b16f97b17048845ccc5239'
LABEL={'Refutes':0,'Supports':1,'Neutral':2}
NAMES=['Refutes','Supports','Neutral']
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def score(rows):
 n=len(rows); correct=sum(x['true_id']==x['pred_id'] for x in rows)
 by={}
 f=[]
 for k in range(3):
  tp=sum(x['true_id']==k and x['pred_id']==k for x in rows)
  support=sum(x['true_id']==k for x in rows); predicted=sum(x['pred_id']==k for x in rows)
  precision=tp/predicted if predicted else 0.; recall=tp/support if support else 0.
  f1=2*precision*recall/(precision+recall) if precision+recall else 0.
  by[NAMES[k]]={'support':support,'precision':precision,'recall':recall,'f1':f1}
  f.append(f1)
 return {'n':n,'accuracy':correct/n if n else None,'macro_f1':sum(f)/3 if n else None,'per_class':by}
def main():
 p=argparse.ArgumentParser(description='Development-only HEALTHVER NLI transfer; never accesses test data')
 p.add_argument('--dev',default=r'D:\DataLineage\S67_HealthVer_Acquisition\HealthVer_official\data\healthver_dev.csv')
 p.add_argument('--model',default=r'D:\DataLineage\S7_NLI_Model')
 p.add_argument('--out',default=r'D:\DataLineage\HealthVer_Transfer_Output')
 p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--bootstrap',type=int,default=1000)
 a=p.parse_args(); src=Path(a.dev).resolve(); model=Path(a.model).resolve(); out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
 if 'test' in src.name.lower():raise SystemExit('REFUSED: test file')
 if sha(src)!=EXPECTED:raise SystemExit('REFUSED: dev hash mismatch; do not substitute other split')
 if not model.is_dir():raise SystemExit('Local model missing: '+str(model))
 with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if len(rows)!=1917:raise SystemExit('Unexpected dev row count')
 for r in rows:
  if r['label'] not in LABEL:raise SystemExit('Unexpected label '+repr(r['label']))
 try:
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
 except ImportError as e:raise SystemExit('Install torch and transformers in Anaconda: '+str(e))
 tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
 net=AutoModelForSequenceClassification.from_pretrained(str(model),local_files_only=True).eval()
 id2label={int(k):str(v).lower() for k,v in net.config.id2label.items()}
 expected={0:'contradiction',1:'entailment',2:'neutral'}
 if len(id2label)!=3 or any(expected[i] not in id2label[i] for i in expected):
  raise SystemExit('REFUSED: model label IDs unverified: '+str(id2label))
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');net.to(device)
 results=[]
 with torch.inference_mode():
  for start in range(0,len(rows),a.batch_size):
   chunk=rows[start:start+a.batch_size]
   tokens=tokenizer([r['claim'] for r in chunk],[r['evidence'] for r in chunk],padding=True,truncation=True,max_length=512,return_tensors='pt')
   logits=net(**{k:v.to(device) for k,v in tokens.items()}).logits
   probs=torch.softmax(logits,dim=-1).cpu().tolist()
   for r,pr in zip(chunk,probs):
    pred=max(range(3),key=lambda k:pr[k])
    results.append({'id':r['id'],'claim':r['claim'],'evidence':r['evidence'],'label':r['label'],'true_id':LABEL[r['label']],'pred_id':pred,'pred_label':NAMES[pred], 'p_refutes':pr[0],'p_supports':pr[1],'p_neutral':pr[2]})
  
 groups=collections.defaultdict(list)
 for r in results:groups[' '.join(r['claim'].lower().split())].append(r)
 rng=random.Random(20261020);keys=list(groups); boots=[]
 for _ in range(max(0,a.bootstrap)):
  sample=[r for k in rng.choices(keys,k=len(keys)) for r in groups[k]]
  m=score(sample);boots.append((m['accuracy'],m['macro_f1']))
 def ci(j):
  if not boots:return None
  vals=sorted(x[j] for x in boots);return [vals[int(.025*(len(vals)-1))],vals[int(.975*(len(vals)-1))]]
 metrics=score(results)
 report={'status':'DEVELOPMENT_ONLY_NO_INDEPENDENT_HOLDOUT','dataset':'HEALTHVER dev evidence pairs','dataset_sha256':sha(src),'model_config_sha256':sha(model/'config.json'),'model_label_ids':id2label,'claim_groups':len(groups),'evaluation_unit':'evidence pair','uncertainty_unit':'normalized claim cluster','metrics':metrics,'claim_cluster_bootstrap_95pct':{'accuracy':ci(0),'macro_f1':ci(1)},'limitations':['Model may have encountered similar health evidence during pretraining','HealthVer evidence reused across train/dev; bootstrap by claim does not address evidence-level dependence','This is a three-class NLI component transfer test, not validation of SciFact fusion','No test or independently held-out data accessed','DEV results are exploratory; do not tune and then report as independent']}
 with (out/'healthver_dev_pair_predictions.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 (out/'healthver_transfer_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'status':report['status'],'n':metrics['n'],'accuracy':metrics['accuracy'],'macro_f1':metrics['macro_f1'],'claim_groups':len(groups),'out':str(out)},indent=2))
if __name__=='__main__':main()
