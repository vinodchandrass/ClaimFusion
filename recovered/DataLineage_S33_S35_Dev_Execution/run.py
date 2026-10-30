import argparse,csv,json,hashlib,collections,math,random,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def metrics(rows,field):
 labels=['SUPPORTS','REFUTES','NOT ENOUGH INFO']; c=collections.Counter((r['gold'],r[field]) for r in rows)
 per={}
 for x in labels:
  tp=c[x,x]; fp=sum(c[y,x] for y in labels if y!=x); fn=sum(c[x,y] for y in labels if y!=x)
  per[x]={'precision':tp/(tp+fp) if tp+fp else 0,'recall':tp/(tp+fn) if tp+fn else 0,'f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0,'support':sum(c[x,y] for y in labels)}
 return {'accuracy':sum(c[x,x] for x in labels)/len(rows) if rows else None,'macro_f1':sum(v['f1'] for v in per.values())/3,'per_class':per,'confusion':{x:{y:c[x,y] for y in labels} for x in labels}}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default=r'D:\DataLineage');ap.add_argument('--model',default=None);ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--max-length',type=int,default=384);ap.add_argument('--max-pairs',type=int,default=0,help='Pilot cap; results explicitly non-full');a=ap.parse_args()
 base=Path(a.base); source=base/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv'; model=Path(a.model) if a.model else base/'S7_NLI_Model';out=base/'S33_S35_Validation_Output'/'dev_execution';out.mkdir(parents=True,exist_ok=True)
 if not source.is_file() or not model.is_dir():raise SystemExit('BLOCKED: evidence CSV or model directory missing')
 with source.open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f); expected={'split','claim_id','claim','gold_label','group_index','page_id','sentence_index','resolved','evidence_text'}
  if not expected.issubset(rd.fieldnames):raise SystemExit('BLOCKED: unexpected evidence schema')
  rows=[r for r in rd if r['split']=='dev']
 if not rows:raise SystemExit('BLOCKED: no development rows')
 claims={};pairs={}; unresolved=0
 for r in rows:
  cid=r['claim_id']; key=(cid,r['page_id'],r['sentence_index']);
  if cid in claims and (claims[cid]['claim']!=r['claim'] or claims[cid]['gold']!=r['gold_label']):raise SystemExit('BLOCKED: inconsistent claim identity')
  claims.setdefault(cid,{'claim':r['claim'],'gold':r['gold_label'],'groups':collections.defaultdict(list)})
  if r['resolved']=='1' and r['evidence_text'].strip():
   if key in pairs and pairs[key]['text']!=r['evidence_text']:raise SystemExit('BLOCKED: conflicting evidence sentence identity')
   pairs.setdefault(key,{'text':r['evidence_text'],'claim':r['claim']})
   claims[cid]['groups'][r['group_index']].append(key)
  else:unresolved+=1
 keys=sorted(pairs)
 if a.max_pairs:keys=keys[:a.max_pairs]
 try:
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
 except ImportError as e:raise SystemExit('BLOCKED: install torch and transformers: '+str(e))
 tok=AutoTokenizer.from_pretrained(str(model),local_files_only=True)
 net=AutoModelForSequenceClassification.from_pretrained(str(model),local_files_only=True)
 labels={int(k):str(v).lower() for k,v in net.config.id2label.items()}
 required={0:'contradiction',1:'entailment',2:'neutral'}
 if labels!=required:raise SystemExit(f'BLOCKED: label mapping {labels} != {required}')
 device='cuda' if torch.cuda.is_available() else 'cpu';net.to(device).eval();scores={}
 for start in range(0,len(keys),a.batch_size):
  chunk=keys[start:start+a.batch_size];inputs=tok([pairs[k]['text'] for k in chunk],[pairs[k]['claim'] for k in chunk],padding=True,truncation=True,max_length=a.max_length,return_tensors='pt').to(device)
  with torch.inference_mode(): p=net(**inputs).logits.softmax(-1).cpu().tolist()
  for k,v in zip(chunk,p):scores[k]=v
  if (start//a.batch_size)%100==0:print('processed',min(start+a.batch_size,len(keys)),'/',len(keys),flush=True)
 pred=[]
 for cid,d in sorted(claims.items()):
  groups=[]
  for g,ks in d['groups'].items():
   if ks and all(k in scores for k in ks):
    # Conjunctive evidence group: all sentences must entail for group support.
    sup=min(scores[k][1] for k in ks);ref=max(scores[k][0] for k in ks);groups.append((sup,ref))
  if not groups:continue
  support=max(x[0] for x in groups);refute=max(x[1] for x in groups)
  # fixed, untuned threshold; exploratory rule, not a validated classifier
  decision='SUPPORTS' if support>=0.5 and support>refute else 'REFUTES' if refute>=0.5 and refute>support else 'NOT ENOUGH INFO'
  pred.append({'claim_id':cid,'gold':d['gold'],'decision':decision,'support_score':support,'refute_score':refute,'n_complete_groups':len(groups)})
 with (out/'dev_claim_predictions.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['claim_id','gold','decision','support_score','refute_score','n_complete_groups']);w.writeheader();w.writerows(pred)
 full=not bool(a.max_pairs); report={'status':'EXPLORATORY_DEV_ONLY' if full else 'PARTIAL_PILOT_NOT_FULL_EVALUATION','model_dir':str(model),'model_label_map':labels,'device':device,'evidence_sha256':sha(source),'development_evidence_rows':len(rows),'development_claims_with_gold_evidence':len(claims),'unique_pairs_available':len(pairs),'pairs_inferred':len(keys),'unresolved_dev_rows':unresolved,'evaluated_claims':len(pred),'coverage_among_evidence_claims':len(pred)/len(claims),'important_limits':['Gold-evidence oracle condition, not retrieval performance','No NOT ENOUGH INFO claim denominator: NEI rows absent from this CSV','Threshold 0.5 fixed a priori for exploratory analysis, not tuned','SciFact independent validation blocked by human-exposure attestation','FEVER test split not read into evaluation'],'metrics_conditional_on_complete_gold_groups':metrics(pred,'decision') if pred else None,'s34_ablation_status':'NOT_RUN_REQUIRES_VALIDATED_COMPARATORS','s35_confirmatory_status':'NOT_RUN_INDEPENDENCE_AND_DENOMINATOR_GATES'}
 (out/'dev_validation_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({k:report[k] for k in ['status','pairs_inferred','evaluated_claims','coverage_among_evidence_claims']},indent=2))
if __name__=='__main__':main()
