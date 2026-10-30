import argparse,collections,csv,hashlib,json,math,re
from pathlib import Path

def tokens(s): return re.findall(r'(?u)\b\w+\b',str(s).lower())
def read_jsonl(p):
 with p.open(encoding='utf-8-sig') as f:
  for i,line in enumerate(f,1):
   if line.strip():
    try: yield json.loads(line)
    except json.JSONDecodeError as e: raise ValueError(f'{p} line {i}: {e}')
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(description='SciFact DEVELOPMENT ONLY: claim-independent BM25 document retrieval; no sealed test access')
 ap.add_argument('--base',default=r'D:\DataLineage');ap.add_argument('--top-k',type=int,default=5);ap.add_argument('--max-claims',type=int,default=0);ap.add_argument('--out');a=ap.parse_args()
 if a.top_k<1:ap.error('--top-k must be >=1')
 root=Path(a.base)/'ScientificClaim_Benchmark_SciFact'/'data'/'data'; cp=root/'claims_dev.jsonl';dp=root/'corpus.jsonl'
 for p in (cp,dp):
  if not p.is_file():ap.error(f'Missing required development input: {p}')
 out=Path(a.out) if a.out else Path(a.base)/'S42_S44_SciFact_Dev_Output';out.mkdir(parents=True,exist_ok=True)
 docs=[]; dfs=collections.Counter(); lengths=[]; seen=set()
 for d in read_jsonl(dp):
  did=str(d.get('doc_id','')); title=d.get('title',''); abstract=d.get('abstract',[])
  if not did or did in seen:raise ValueError('Missing or duplicate corpus doc_id: '+did)
  seen.add(did);sentences=abstract if isinstance(abstract,list) else [str(abstract)]
  body=' '.join(str(s) for s in sentences); toks=tokens(str(title)+' '+body);tf=collections.Counter(toks)
  docs.append((did,str(title),sentences,tf,len(toks)));dfs.update(tf.keys());lengths.append(len(toks))
 if not docs:raise ValueError('Empty corpus')
 n=len(docs);avdl=sum(lengths)/n
 claims=list(read_jsonl(cp));claims=claims[:a.max_claims] if a.max_claims else claims
 if not claims:raise ValueError('Empty dev claims')
 k1=1.2;b=.75; labels=collections.Counter();hits=collections.Counter();valid=0
 with (out/'s42_dev_retrieval.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.writer(f);w.writerow(['claim_id','claim','gold_label_DEV_ONLY','rank','doc_id','score','title','abstract','corpus_sha256'])
  corp_hash=digest(dp)
  for c in claims:
   cid=str(c.get('id',''));q=str(c.get('claim',''));label=str(c.get('label',''));labels[label]+=1
   if not cid or not q:raise ValueError('Missing dev claim id/text')
   qtokens=set(tokens(q));scored=[]
   for did,title,sents,tf,dl in docs:
    score=0.
    for t in qtokens:
     freq=tf.get(t,0)
     if freq:
      idf=math.log(1+(n-dfs[t]+.5)/(dfs[t]+.5))
      score+=idf*(freq*(k1+1))/(freq+k1*(1-b+b*dl/avdl))
    if score>0:scored.append((score,did,title,sents))
   scored.sort(key=lambda z:(-z[0],z[1]));chosen=scored[:a.top_k]
   for rank,(score,did,title,sents) in enumerate(chosen,1):w.writerow([cid,q,label,rank,did,format(score,'.12g'),title,' '.join(str(s) for s in sents),corp_hash])
   # Oracle doc IDs are used ONLY to compute development retrieval recall, never ranking.
   gold={str(e) for e in c.get('cited_doc_ids',[]) if e is not None}
   if gold:
    valid+=1
    got={x[1] for x in chosen}
    if gold&got:hits['any_gold_doc_at_k']+=1
    if gold<=got:hits['all_gold_docs_at_k']+=1
 report={'status':'DEVELOPMENT_ONLY_NOT_INDEPENDENT_VALIDATION','claims':len(claims),'corpus_documents':n,'top_k':a.top_k,'dev_label_distribution':dict(labels),'claims_with_gold_cited_doc_ids':valid,'recall_any_gold_document_at_k':hits['any_gold_doc_at_k']/valid if valid else None,'recall_all_gold_documents_at_k':hits['all_gold_docs_at_k']/valid if valid else None,'claims_sha256':digest(cp),'corpus_sha256':corp_hash,'retrieval_method':'BM25 title + full abstract, deterministic ties, no oracle evidence in ranking','limitations':['SciFact dev has prior project exposure; not independent','document-level retrieval only; no NLI or three-class claim classification','gold cited_doc_ids used only for evaluation after ranking','no publisher/source independence or document version provenance established','sealed claims_test.jsonl deliberately not read']}
 (out/'s42_s44_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({k:report[k] for k in ['status','claims','corpus_documents','claims_with_gold_cited_doc_ids','recall_any_gold_document_at_k']},indent=2));print('Outputs:',out)
if __name__=='__main__':main()
