import argparse,csv,hashlib,json,re,collections,math,sys
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def jsonl(p):
 with open(p,encoding='utf-8-sig') as f:
  for line in f:
   if line.strip():yield json.loads(line)
def tok(s):return re.findall(r'[a-z0-9]+',s.lower())
def writecsv(p,rows,fields):
 with open(p,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--claims');a.add_argument('--corpus');a.add_argument('--retrieval');a.add_argument('--model',default='');a.add_argument('--out');a.add_argument('--top-docs',type=int,default=5);a.add_argument('--sentences-per-doc',type=int,default=3);a.add_argument('--batch-size',type=int,default=16);a.add_argument('--nli',action='store_true');x=a.parse_args()
 base=Path(x.base);claims=Path(x.claims) if x.claims else base/'ScientificClaim_Benchmark_SciFact/data/data/claims_dev.jsonl';corpus=Path(x.corpus) if x.corpus else base/'ScientificClaim_Benchmark_SciFact/data/data/corpus.jsonl';retr=Path(x.retrieval) if x.retrieval else base/'S42_S44_SciFact_Dev_Output/s42_dev_retrieval.csv';out=Path(x.out) if x.out else base/'S46_S48_Dev_NonOracle_Output';out.mkdir(parents=True,exist_ok=True)
 for p in (claims,corpus,retr):
  if not p.is_file():raise FileNotFoundError(str(p))
 if 'test' in claims.name.lower():raise ValueError('TEST claims forbidden in development runner')
 cl={str(v['id']):v for v in jsonl(claims)};docs={str(v['doc_id']):v for v in jsonl(corpus)}
 retrieved=collections.defaultdict(list)
 with open(retr,encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):retrieved[str(r['claim_id'])].append(r)
 if set(retrieved)!=set(cl):raise ValueError(f'Claim ID mismatch retrieval={len(retrieved)} dev={len(cl)}')
 candidates=[];claims_rows=[];doc_hit=collections.Counter();gold_sent_tot=gold_sent_found=0
 for cid,c in cl.items():
  ev=c.get('evidence') or {}; gold=set();labs=set()
  for did,annotations in ev.items():
   for annotation in annotations:
    labs.add(annotation.get('label'))
    for si in annotation.get('sentences',[]):gold.add((str(did),int(si)))
  status='UNANNOTATED' if not labs else 'SUPPORTS' if labs=={'SUPPORT'} else 'REFUTES' if labs=={'CONTRADICT'} else 'MIXED'
  ranked=sorted(retrieved[cid],key=lambda r:int(r['rank']))[:x.top_docs];candidates_by_claim=[]
  for r in ranked:
   did=str(r['doc_id']);doc=docs.get(did)
   if doc is None:raise ValueError('Missing corpus doc '+did)
   # SciFact abstract is a list of original indexed sentences; do not split flattened text.
   sentences=doc.get('abstract',[])
   if not isinstance(sentences,list):raise ValueError('Expected list abstract for '+did)
   ct=collections.Counter(tok(c['claim']));scores=[]
   for i,s in enumerate(sentences):
    st=collections.Counter(tok(s)); overlap=sum(min(ct[t],n) for t,n in st.items());norm=overlap/math.sqrt(max(1,sum(st.values())))
    scores.append((norm,i,s))
   for score,i,s in sorted(scores,key=lambda v:(-v[0],v[1]))[:x.sentences_per_doc]:
    candidates_by_claim.append({'claim_id':cid,'claim':c['claim'],'doc_id':did,'doc_rank':int(r['rank']),'sentence_index':i,'sentence':s,'lexical_score':round(score,8),'gold_sentence_DEV_ONLY':int((did,i) in gold),'annotation_status_DEV_ONLY':status})
  candidates.extend(candidates_by_claim)
  got={(r['doc_id'],r['sentence_index']) for r in candidates_by_claim};gold_sent_tot+=len(gold);gold_sent_found+=len(gold&got)
  if gold:doc_hit['eligible']+=1;doc_hit['any']+=bool(gold&got);doc_hit['all']+=gold.issubset(got)
  claims_rows.append({'claim_id':cid,'annotation_status_DEV_ONLY':status,'gold_sentences':len(gold),'candidate_sentences':len(candidates_by_claim),'gold_sentences_found':len(gold&got),'any_gold_sentence_found':int(bool(gold&got)) if gold else '', 'all_gold_sentences_found':int(gold.issubset(got)) if gold else ''})
 if x.nli:
  if not x.model:raise ValueError('--model required with --nli; point to frozen local model')
  import torch
  from transformers import AutoTokenizer,AutoModelForSequenceClassification
  model=AutoModelForSequenceClassification.from_pretrained(x.model,local_files_only=True);tokenizer=AutoTokenizer.from_pretrained(x.model,local_files_only=True)
  print('NLI label map:',model.config.id2label)
  # frozen expected mapping: contradiction=0, entailment=1, neutral=2
  names={i:str(v).lower() for i,v in model.config.id2label.items()}
  if not ('contradiction' in names[0] and 'entailment' in names[1] and 'neutral' in names[2]):raise ValueError('Unexpected NLI label mapping: '+str(names))
  model.eval();device='cuda' if torch.cuda.is_available() else 'cpu';model.to(device)
  for i in range(0,len(candidates),x.batch_size):
   batch=candidates[i:i+x.batch_size];tokens=tokenizer([r['sentence'] for r in batch],[r['claim'] for r in batch],padding=True,truncation=True,max_length=512,return_tensors='pt').to(device)
   with torch.no_grad():probs=torch.softmax(model(**tokens).logits,dim=-1).cpu().tolist()
   for r,p in zip(batch,probs):r.update({'p_contradiction':round(p[0],8),'p_entailment':round(p[1],8),'p_neutral':round(p[2],8)})
   if i%1600==0:print('NLI candidates',i+len(batch),'/',len(candidates),flush=True)
  byclaim=collections.defaultdict(list)
  for r in candidates:byclaim[r['claim_id']].append(r)
  for r in claims_rows:
   rows=byclaim[r['claim_id']];best_e=max(rows,key=lambda z:z['p_entailment']) if rows else None;best_c=max(rows,key=lambda z:z['p_contradiction']) if rows else None
   e=best_e['p_entailment'] if best_e else 0;c=best_c['p_contradiction'] if best_c else 0
   # Exploratory, no tuned thresholds; score competition can be miscalibrated.
   r.update({'predicted_binary_exploratory':('SUPPORTS' if e>=c else 'REFUTES') if rows else 'ABSTAIN','max_entailment':e,'max_contradiction':c,'winner_doc_id':(best_e if e>=c else best_c)['doc_id'] if rows else '', 'winner_sentence_index':(best_e if e>=c else best_c)['sentence_index'] if rows else ''})
 fields=['claim_id','claim','doc_id','doc_rank','sentence_index','sentence','lexical_score','gold_sentence_DEV_ONLY','annotation_status_DEV_ONLY']+(['p_contradiction','p_entailment','p_neutral'] if x.nli else [])
 writecsv(out/'s46_sentence_candidates.csv',candidates,fields)
 fields2=['claim_id','annotation_status_DEV_ONLY','gold_sentences','candidate_sentences','gold_sentences_found','any_gold_sentence_found','all_gold_sentences_found']+(['predicted_binary_exploratory','max_entailment','max_contradiction','winner_doc_id','winner_sentence_index'] if x.nli else [])
 writecsv(out/'s47_claim_diagnostics.csv',claims_rows,fields2)
 nli_summary={}
 if x.nli:
  eligible=[r for r in claims_rows if r['annotation_status_DEV_ONLY'] in ('SUPPORTS','REFUTES')]
  for group in ('SUPPORTS','REFUTES','ALL_ANNOTATED'):
   subset=[r for r in eligible if group=='ALL_ANNOTATED' or r['annotation_status_DEV_ONLY']==group]
   nli_summary[group]={'n':len(subset),'binary_accuracy_exploratory':sum(r['predicted_binary_exploratory']==r['annotation_status_DEV_ONLY'] for r in subset)/len(subset) if subset else None}
 report={'status':'DEVELOPMENT_ONLY_NOT_INDEPENDENT','claims':len(cl),'corpus_docs':len(docs),'candidate_sentences':len(candidates),'retrieval_claims':len(retrieved),'top_docs':x.top_docs,'sentences_per_doc':x.sentences_per_doc,'annotation_distribution':dict(collections.Counter(r['annotation_status_DEV_ONLY'] for r in claims_rows)),'gold_sentence_recall':{'gold_sentence_count':gold_sent_tot,'retrieved_gold_sentences':gold_sent_found,'micro_recall':gold_sent_found/gold_sent_tot if gold_sent_tot else None,'annotated_claims':doc_hit['eligible'],'any_gold_sentence_retrieved':doc_hit['any'],'all_gold_sentences_retrieved':doc_hit['all']},'nli_executed':x.nli,'nli_binary_exploratory':nli_summary,'input_sha256':{str(p):sha(p) for p in (claims,corpus,retr)},'limitations':['SciFact development claims only; no independent holdout','Lexical sentence selection from retrieved documents; no gold evidence in ranking','UNANNOTATED is not automatically NEI','Binary max-score NLI comparison is exploratory, uncalibrated, and not a novel fusion method','Source independence and publication-version provenance unverified','No test claims accessed']}
 (out/'s48_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps({'out':str(out),'claims':len(cl),'candidates':len(candidates),'nli':x.nli},indent=2))
if __name__=='__main__':main()
