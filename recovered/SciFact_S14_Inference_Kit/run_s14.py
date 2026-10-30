import argparse,csv,hashlib,json,sys,zipfile
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
PINS={'claims_train.jsonl':'f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0','claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217','corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'}
COMMIT='6c749ce3425cd33b46d187e45b92bbf96ee12ec7'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
def norm(s):return ' '.join(s.casefold().split())
def prep(records,corpus,exclude=set()):
 rows=[];skips=[]
 for r in records:
  if norm(r['claim']) in exclude:
   skips.append({'claim_id':str(r['id']),'reason':'claim_text_overlap_dev'});continue
  for did,groups in r.get('evidence',{}).items():
   labels={g['label'] for g in groups}
   if len(labels)!=1 or not labels<={'SUPPORT','CONTRADICT'}:
    skips.append({'claim_id':str(r['id']),'doc_id':str(did),'reason':'mixed_or_unexpected_labels'});continue
   idx=sorted({i for g in groups for i in g['sentences']})
   abstract=corpus[str(did)]['abstract']
   if not idx or any(type(i)!=int or i<0 or i>=len(abstract) for i in idx):raise RuntimeError(f'Invalid evidence index {r["id"]} {did}')
   rows.append({'claim_id':str(r['id']),'doc_id':str(did),'claim':r['claim'],'gold':next(iter(labels)),'indices':idx,'sentences':[abstract[i] for i in idx]})
 return rows,skips
def writecsv(path,rows,cols):
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model-dir',required=True);p.add_argument('--s7-zip',required=True);p.add_argument('--out',required=True);p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
 root=Path(a.root);model_dir=Path(a.model_dir);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 for fn,expected in PINS.items():
  if sha(root/fn)!=expected:raise RuntimeError(f'Dataset integrity failure: {fn}')
 required=['config.json','model.safetensors','tokenizer.json']
 for fn in required:
  if not (model_dir/fn).is_file():raise RuntimeError(f'Missing local model file: {fn}')
 with zipfile.ZipFile(a.s7_zip) as z:
  lock=json.loads(z.read('semantic_model_lock.json'))
  if lock['resolved_commit']!=COMMIT or lock['max_length']!=384 or lock['dataset_sha256']!=PINS:raise RuntimeError('S7 lock differs; STOP')
  baseline=list(csv.DictReader(z.read('S7_predictions.csv').decode('utf-8-sig').splitlines()))
 corpus={str(x['doc_id']):x for x in read(root/'corpus.jsonl')}
 train=read(root/'claims_train.jsonl');dev=read(root/'claims_dev.jsonl')
 devnorm={norm(r['claim']) for r in dev}
 tr,skips=prep(train,corpus,devnorm);dv,dev_skips=prep(dev,corpus)
 if dev_skips:raise RuntimeError(f'Dev exclusions unexpected: {dev_skips[:3]}')
 if len(dv)!=len(baseline):raise RuntimeError('S7 dev pair count mismatch')
 # Claim-disjoint 20% deterministic calibration set, no development-label selection.
 # Hash-based partition prevents dependence on row order and avoids tuning the split.
 for r in tr:
  bucket=int(hashlib.sha256(('S14-calibration-2026:'+norm(r['claim'])).encode()).hexdigest()[:12],16)%100
  r['partition']='calibration' if bucket<20 else 'train_reserved'
 if not any(r['partition']=='calibration' for r in tr):raise RuntimeError('No calibration rows')
 model_hashes={fn:sha(model_dir/fn) for fn in required}
 tok=AutoTokenizer.from_pretrained(str(model_dir),local_files_only=True)
 model=AutoModelForSequenceClassification.from_pretrained(str(model_dir),local_files_only=True);model.eval().cpu()
 labels={int(k):v.lower() for k,v in model.config.id2label.items()}
 def label_id(name):
  hits=[k for k,v in labels.items() if name in v]
  if len(hits)!=1:raise RuntimeError('Invalid label mapping '+str(labels))
  return hits[0]
 ci=label_id('contradict');ei=label_id('entail');ni=label_id('neutral')
 # Inference on calibration-only training pairs plus the frozen dev evaluation pairs.
 selected=[r for r in tr if r['partition']=='calibration']
 selected.extend(dv)
 jobs=[(r,s,idx) for r in selected for s,idx in zip(r['sentences'],r['indices'])]
 probs=[]
 with torch.inference_mode():
  for k in range(0,len(jobs),a.batch_size):
   batch=jobs[k:k+a.batch_size]
   enc=tok([x[1] for x in batch],[x[0]['claim'] for x in batch],padding=True,truncation=True,max_length=384,return_tensors='pt')
   probs.extend(torch.softmax(model(**enc).logits,dim=-1).cpu().numpy().tolist())
 sentence_rows=[];pair_rows=[];pos=0
 for r in selected:
  arr=np.array(probs[pos:pos+len(r['sentences'])]);pos+=len(r['sentences'])
  split='dev_evaluation' if 'partition' not in r else r['partition']
  for ix,s,v in zip(r['indices'],r['sentences'],arr):
   sentence_rows.append({'split':split,'claim_id':r['claim_id'],'doc_id':r['doc_id'],'sentence_index':ix,'sentence_sha256':hashlib.sha256(s.encode()).hexdigest(),'gold':r['gold'],'p_contradiction':float(v[ci]),'p_entailment':float(v[ei]),'p_neutral':float(v[ni])})
  mean=arr.mean(axis=0)
  pair_rows.append({'split':split,'claim_id':r['claim_id'],'doc_id':r['doc_id'],'gold':r['gold'],'sentence_count':len(arr),'p_contradiction':float(mean[ci]),'p_entailment':float(mean[ei]),'p_neutral':float(mean[ni]),'pred':'CONTRADICT' if mean[ci]>=mean[ei] else 'SUPPORT'})
 lookup={(r['claim_id'],r['doc_id']):r for r in pair_rows if r['split']=='dev_evaluation'}
 diffs=[];mismatch=[]
 for b in baseline:
  key=(b['claim_id'],b['doc_id']);r=lookup.get(key)
  if r is None or r['gold']!=b['gold'] or r['pred']!=b['pred'] or int(r['sentence_count'])!=int(b['sentence_count']):mismatch.append(key);continue
  diffs.extend(abs(r[k]-float(b[k])) for k in ['p_contradiction','p_entailment','p_neutral'])
 maxdiff=max(diffs,default=float('inf'))
 # Floating-point CPU library differences are expected to be small; do not silently accept substantive drift.
 if mismatch or maxdiff>0.005:raise RuntimeError(f'S7 PARITY FAILURE: mismatch={mismatch[:10]}, max_abs_prob_diff={maxdiff}')
 scols=['split','claim_id','doc_id','sentence_index','sentence_sha256','gold','p_contradiction','p_entailment','p_neutral']
 pcols=['split','claim_id','doc_id','gold','sentence_count','p_contradiction','p_entailment','p_neutral','pred']
 writecsv(out/'S14_sentence_predictions.csv',sentence_rows,scols);writecsv(out/'S14_pair_predictions.csv',pair_rows,pcols)
 report={'status':'PASS_INFERENCE_AND_S7_PARITY','protocol':'S14 inference only; no thresholds selected; no fusion trained; dev evaluation only','dataset_sha256':PINS,'model_id':lock['model_id'],'model_commit':COMMIT,'local_model_hashes':model_hashes,'max_length':384,'input_direction':lock['input_direction'],'partition_rule':'sha256(S14-calibration-2026: + normalized claim text) first 12 hex modulo 100 < 20','train_overlap_and_other_exclusions':skips,'calibration_pairs':len([r for r in pair_rows if r['split']=='calibration']),'calibration_claims':len({r['claim_id'] for r in selected if r.get('partition')=='calibration'}),'reserved_training_pairs':len([r for r in tr if r['partition']=='train_reserved']),'dev_pairs':len(dv),'sentence_predictions':len(sentence_rows),'s7_max_abs_probability_difference':maxdiff,'s7_pair_mismatches':len(mismatch),'python':sys.version,'torch':torch.__version__,'limitations':['Threshold not selected','No source-reliability ground truth','No independent test labels','Gold evidence only']}
 (out/'S14_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','calibration_pairs','dev_pairs','sentence_predictions','s7_max_abs_probability_difference']},indent=2))
if __name__=='__main__':main()
