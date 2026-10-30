import argparse,csv,hashlib,json,platform,sys,random
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix,precision_recall_fscore_support
from transformers import AutoTokenizer,AutoModelForSequenceClassification, __version__ as transformers_version
from huggingface_hub import HfApi
import torch

PINS={'claims_train.jsonl':'f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0','claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217','corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'}
LABELS=['CONTRADICT','SUPPORT']
def read(p):
 with p.open(encoding='utf-8-sig') as f:return [json.loads(s) for s in f if s.strip()]
def norm(s):return ' '.join(s.casefold().split())
def prepare(rows,corpus):
 out=[];mixed=[]
 for r in rows:
  for docid,groups in r.get('evidence',{}).items():
   labs={g['label'] for g in groups}
   if len(labs)!=1 or not labs<=set(LABELS):mixed.append([r['id'],docid,sorted(labs)]);continue
   doc=corpus[str(docid)];ix=sorted({i for g in groups for i in g['sentences']});abstract=doc['abstract']
   if not ix or any(not isinstance(i,int) or i<0 or i>=len(abstract) for i in ix):raise RuntimeError(f'Bad sentence index {r["id"]} {docid}')
   out.append({'claim_id':str(r['id']),'doc_id':str(docid),'claim':r['claim'],'sentences':[abstract[i] for i in ix],'label':next(iter(labs))})
 return out,mixed
def metrics(y,p):
 precision,recall,f1,_=precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0)
 return {'n':len(y),'macro_f1':float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),'accuracy':float(accuracy_score(y,p)),'confusion_matrix':confusion_matrix(y,p,labels=LABELS).tolist(),'per_class':{l:{'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i,l in enumerate(LABELS)}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--out',required=True);ap.add_argument('--model',default='cross-encoder/nli-deberta-v3-base');ap.add_argument('--revision',default='main');ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--bootstrap',type=int,default=10000);ap.add_argument('--max-length',type=int,default=384);a=ap.parse_args()
 root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 for f,sha in PINS.items():
  if hashlib.sha256((root/f).read_bytes()).hexdigest()!=sha:raise RuntimeError(f'Source hash mismatch: {f}')
 corpus={str(d['doc_id']):d for d in read(root/'corpus.jsonl')};train=read(root/'claims_train.jsonl');dev=read(root/'claims_dev.jsonl')
 devtexts={norm(r['claim']) for r in dev};excluded=sorted(str(r['id']) for r in train if norm(r['claim']) in devtexts)
 rows,mixed=prepare(dev,corpus)
 if mixed:raise RuntimeError(f'Mixed labels need separate handling: {mixed[:5]}')
 if not rows:raise RuntimeError('No gold evidence pairs')
 commit='6c749ce3425cd33b46d187e45b92bbf96ee12ec7'
 # Prevent accidental model swapping after the first execution.
 lock=out/'semantic_model_lock.json'
 frozen={'model_id':a.model,'resolved_commit':commit,'max_length':a.max_length,'input_direction':'premise=evidence sentence, hypothesis=claim','aggregation':'mean of sentence softmax probabilities; binary argmax between contradiction and entailment; neutral margin reported separately','zero_shot':True,'dataset_sha256':PINS,'seed':2026}
 if lock.exists():
  if json.loads(lock.read_text(encoding='utf-8'))!=frozen:raise RuntimeError('Model/protocol lock changed; use a new output directory and explicitly document protocol amendment')
 else:lock.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 tok=AutoTokenizer.from_pretrained(r"D:\DataLineage\S7_NLI_Model",local_files_only=True);model=AutoModelForSequenceClassification.from_pretrained(r"D:\DataLineage\S7_NLI_Model",local_files_only=True);model.eval()
 id2label={int(k):str(v).lower() for k,v in model.config.id2label.items()}
 def find_label(terms):
  hits=[i for i,v in id2label.items() if any(t in v for t in terms)]
  if len(hits)!=1:raise RuntimeError(f'Cannot uniquely map {terms} in {id2label}; STOP, do not guess')
  return hits[0]
 con=find_label(['contradiction','contradict']);ent=find_label(['entailment','entail']);neutral=find_label(['neutral'])
 # Store resolved architecture and label mapping before scoring.
 (out/'resolved_model_metadata.json').write_text(json.dumps({'model':a.model,'commit':commit,'id2label':id2label,'torch':torch.__version__,'transformers':transformers_version,'python':sys.version,'device':'cpu','max_length':a.max_length},indent=2),encoding='utf-8')
 pairs=[(s,r['claim']) for r in rows for s in r['sentences']];lens=[len(r['sentences']) for r in rows];prob=[];truncated=0
 with torch.inference_mode():
  for start in range(0,len(pairs),a.batch_size):
   batch=pairs[start:start+a.batch_size]
   encoded=tok([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=True,max_length=a.max_length,return_tensors='pt')
   logits=model(**encoded).logits
   prob.extend(torch.softmax(logits,dim=-1).cpu().numpy().tolist())
 scores=[];pos=0
 for n in lens:
  s=np.array(prob[pos:pos+n]);pos+=n;scores.append(s.mean(axis=0))
 pred=[LABELS[0] if s[con]>=s[ent] else LABELS[1] for s in scores]
 y=[r['label'] for r in rows];major=['SUPPORT']*len(rows)
 # Pairwise, claim-cluster bootstrap: same resampled claim IDs for both methods.
 groups=defaultdict(list)
 for i,r in enumerate(rows):groups[r['claim_id']].append(i)
 ids=list(groups);rng=np.random.default_rng(2026);deltas=[]
 yy=np.array(y);pp=np.array(pred);mm=np.array(major)
 for _ in range(a.bootstrap):
  sample=rng.integers(0,len(ids),len(ids));ix=[j for k in sample for j in groups[ids[k]]]
  deltas.append(float(f1_score(yy[ix],pp[ix],labels=LABELS,average='macro',zero_division=0)-f1_score(yy[ix],mm[ix],labels=LABELS,average='macro',zero_division=0)))
 delta=float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0)-f1_score(y,major,labels=LABELS,average='macro',zero_division=0))
 report={'protocol':'S7 zero-shot semantic NLI, gold evidence only; NOT provenance fusion','model_lock':frozen,'train_text_duplicate_exclusions_for_future_training':excluded,'gold_pairs':len(rows),'claims':len(ids),'sentence_inferences':len(pairs),'model_metrics':metrics(y,pred),'majority_metrics':metrics(y,major),'paired_claim_bootstrap':{'resamples':a.bootstrap,'delta_macro_f1':delta,'ci_95':np.percentile(deltas,[2.5,97.5]).tolist()},'limitations':['No document retrieval','No SciFact test gold labels','Neutral output not treated as annotated NEI','Gold evidence supplied to both models','Not a provenance-aware fusion method','No training or dev tuning','No independent reliability ground truth']}
 with (out/'S7_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['claim_id','doc_id','gold','pred','p_contradiction','p_entailment','p_neutral','sentence_count'])
  for r,p,s in zip(rows,pred,scores):w.writerow([r['claim_id'],r['doc_id'],r['label'],p,float(s[con]),float(s[ent]),float(s[neutral]),len(r['sentences'])])
 (out/'S7_results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('S7 COMPLETED',json.dumps({'macro_f1':report['model_metrics']['macro_f1'],'delta':delta,'CI':report['paired_claim_bootstrap']['ci_95']},indent=2))
if __name__=='__main__':main()

