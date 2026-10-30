import argparse,csv,hashlib,json,math,re,collections,random
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def readjson(p):
 with open(p,encoding='utf-8-sig') as f:
  for l in f:
   if l.strip():yield json.loads(l)
def csvread(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rs,fields):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rs)
def bm25_tokens(s):return re.findall(r'(?u)\b\w+\b',str(s).lower())
def sent_tokens(s):return re.findall(r'[a-z0-9]+',str(s).lower())
def make(claims,corpus):
 docs=[];df=collections.Counter();lengths=[];byid={}
 for d in readjson(corpus):
  did=str(d['doc_id']);ss=d.get('abstract',[])
  if not isinstance(ss,list) or did in byid:raise ValueError('Invalid corpus record '+did)
  t=bm25_tokens(str(d.get('title',''))+' '+' '.join(map(str,ss)));tf=collections.Counter(t)
  docs.append((did,d.get('title',''),ss,tf,len(t)));byid[did]=d;df.update(tf.keys());lengths.append(len(t))
 n=len(docs);avdl=sum(lengths)/n;k1=1.2;b=.75
 retrieval=[];candidates=[]
 for c in readjson(claims):
  cid=str(c['id']);q=str(c['claim']);qt=set(bm25_tokens(q));scored=[]
  for did,title,ss,tf,dl in docs:
   score=0.
   for t in qt:
    freq=tf.get(t,0)
    if freq:
     idf=math.log(1+(n-df[t]+.5)/(df[t]+.5))
     score+=idf*(freq*(k1+1))/(freq+k1*(1-b+b*dl/avdl))
   if score>0:scored.append((score,did,title,ss))
  scored.sort(key=lambda z:(-z[0],z[1]))
  ct=collections.Counter(sent_tokens(q))
  for rank,(score,did,title,ss) in enumerate(scored[:5],1):
   retrieval.append({'claim_id':cid,'claim':q,'rank':rank,'doc_id':did,'score':format(score,'.12g')})
   ranked=[]
   for i,s in enumerate(ss):
    st=collections.Counter(sent_tokens(s));overlap=sum(min(ct[t],v) for t,v in st.items());norm=overlap/math.sqrt(max(1,sum(st.values())))
    ranked.append((norm,i,s))
   for norm,i,s in sorted(ranked,key=lambda z:(-z[0],z[1]))[:3]:
    candidates.append({'claim_id':cid,'claim':q,'doc_id':did,'doc_rank':rank,'sentence_index':i,'sentence':s,'lexical_score':round(norm,8)})
 return retrieval,candidates

def parity(root,out):
 base=root/'ScientificClaim_Benchmark_SciFact/data/data';corpus=base/'corpus.jsonl';dev=base/'claims_dev.jsonl'
 oldret=root/'S42_S44_SciFact_Dev_Output/s42_dev_retrieval.csv';oldcand=root/'S46_S48_Dev_NonOracle_Output/s46_sentence_candidates.csv'
 for p in (corpus,dev,oldret,oldcand):
  if not p.is_file():raise FileNotFoundError(str(p))
 rr,cc=make(dev,corpus);orows=csvread(oldret);oc=csvread(oldcand)
 retfields=('claim_id','claim','rank','doc_id','score');candfields=('claim_id','claim','doc_id','doc_rank','sentence_index','sentence','lexical_score')
 mismatches=[]
 for name,new,old,fields in (('retrieval',rr,orows,retfields),('candidates',cc,oc,candfields)):
  if len(new)!=len(old):mismatches.append({'stage':name,'error':'row count mismatch','generated':len(new),'frozen':len(old)})
  for i,(a,b) in enumerate(zip(new,old)):
   for k in fields:
    if str(a[k])!=str(b[k]):
     if len(mismatches)<25:mismatches.append({'stage':name,'row':i,'field':k,'generated':str(a[k])[:120],'frozen':str(b[k])[:120]})
     break
 report={'status':'PASS' if not mismatches else 'FAIL','retrieval_rows':len(rr),'candidate_rows':len(cc),'mismatches':mismatches,'input_sha256':{str(p):sha(p) for p in (dev,corpus,oldret,oldcand)}}
 (out/'dev_equivalence_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');return report

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',default=r'D:\DataLineage');p.add_argument('--out',default=r'D:\DataLineage\Final_Independent_Validation');p.add_argument('--mode',choices=['preflight','predict'],default='preflight');p.add_argument('--frozen-model');p.add_argument('--adapted-model');p.add_argument('--batch-size',type=int,default=8);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
 root=Path(a.base);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 r=parity(root,out);print('DEV EQUIVALENCE:',r['status'],r['mismatches'][:3],flush=True)
 if r['status']!='PASS':raise SystemExit('STOP: dev equivalence mismatch; test not accessed')
 if a.mode=='preflight':print('PASS: test not accessed. Review protocol before predict.');return
 if not a.frozen_model or not a.adapted_model:raise SystemExit('Both model paths required')
 # This command deliberately reads test claims only after preflight and explicit --mode predict.
 test=root/'ScientificClaim_Benchmark_SciFact/data/data/claims_test.jsonl';corpus=root/'ScientificClaim_Benchmark_SciFact/data/data/corpus.jsonl'
 if not test.is_file():raise FileNotFoundError(test)
 rr,cc=make(test,corpus)
 writecsv(out/'test_retrieval_UNLABELED.csv',rr,['claim_id','claim','rank','doc_id','score'])
 writecsv(out/'test_candidates_UNLABELED.csv',cc,['claim_id','claim','doc_id','doc_rank','sentence_index','sentence','lexical_score'])
 import torch
 from transformers import AutoTokenizer,AutoModelForSequenceClassification
 torch.set_num_threads(a.threads)
 for key,path,length in (('frozen',a.frozen_model,512),('adapted',a.adapted_model,192)):
  modelpath=Path(path);tok=AutoTokenizer.from_pretrained(str(modelpath),local_files_only=True)
  model=AutoModelForSequenceClassification.from_pretrained(str(modelpath),local_files_only=True).eval().to('cpu')
  labels={int(k):str(v).lower() for k,v in model.config.id2label.items()}
  if any(s not in labels.get(i,'') for i,s in enumerate(('contradiction','entailment','neutral'))):raise ValueError('Unexpected label mapping '+str(labels))
  fields=['claim_id','claim','doc_id','doc_rank','sentence_index','sentence','p_contradiction','p_entailment','p_neutral'];results=[]
  for i in range(0,len(cc),a.batch_size):
   part=cc[i:i+a.batch_size];batch=tok([x['sentence'] for x in part],[x['claim'] for x in part],padding=True,truncation=True,max_length=length,return_tensors='pt')
   with torch.no_grad():probs=torch.softmax(model(**batch).logits,dim=-1).tolist()
   for x,v in zip(part,probs):results.append({**{k:x[k] for k in fields if k in x},'p_contradiction':v[0],'p_entailment':v[1],'p_neutral':v[2]})
   if i%800==0:print(key,'inference',min(i+len(part),len(cc)),'/',len(cc),flush=True)
  writecsv(out/f'test_{key}_predictions_UNLABELED.csv',results,fields)
  del model,tok
 manifest={'status':'PREDICTIONS_GENERATED_UNSCORED','test_sha256':sha(test),'corpus_sha256':sha(corpus),'models':{'frozen':{'config_sha256':sha(Path(a.frozen_model)/'config.json'),'max_length':512},'adapted':{'config_sha256':sha(Path(a.adapted_model)/'config.json'),'max_length':192}},'predictions_sha256':{key:sha(out/f'test_{key}_predictions_UNLABELED.csv') for key in ('frozen','adapted')},'test_labels_used':False,'test_claim_count':len(set(x['claim_id'] for x in cc)),'candidate_count':len(cc)}
 (out/'test_prediction_lock.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('PREDICTIONS LOCKED. Do not change model or rules. No scoring performed.')
if __name__=='__main__':main()
