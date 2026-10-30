import argparse,csv,json,hashlib,pathlib,datetime,collections,re,math

def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def norm(s):return ' '.join(re.findall(r'\w+',str(s).casefold()))
def write(p,rows,fields):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--claims');a.add_argument('--corpus');a.add_argument('--reference-claims');a.add_argument('--top-k',type=int,default=5);a.add_argument('--output');a.add_argument('--custody-attested',action='store_true',help='Record user attestation only; does not independently verify custody');x=a.parse_args()
 if bool(x.claims)!=bool(x.corpus):a.error('Provide both --claims and --corpus, or neither for readiness-only audit')
 if x.top_k<1:a.error('--top-k must be positive')
 out=pathlib.Path(x.output or pathlib.Path(x.base)/'S33_S35_Validation_Output'/'s39_s41_combined');out.mkdir(parents=True,exist_ok=True)
 report={'status':'PREPARATION_ONLY_NO_EXTERNAL_VALIDATION','created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_hashes':{},'gates':{},'warnings':[],'retrieval_policy':{'algorithm':'BM25','top_k':x.top_k,'k1':1.2,'b':0.75,'tie_break':'document_id_ascending','query':'claim_text_only','oracle_groups_used':False}}
 if not x.claims:
  report['gates']={'external_claims':'MISSING','external_corpus':'MISSING','retrieval':'NOT_EXECUTED','three_class':'NOT_EVALUATED','provenance':'NOT_VERIFIED','independent_validation':'BLOCKED'}
 else:
  claims=read(x.claims);docs=read(x.corpus)
  if not claims or not {'claim_id','claim'}.issubset(claims[0]):raise ValueError('Claims CSV requires claim_id,claim')
  if not docs or not {'document_id','text'}.issubset(docs[0]):raise ValueError('Corpus CSV requires document_id,text')
  if len(set(r['claim_id'] for r in claims))!=len(claims):raise ValueError('Duplicate claim IDs')
  if len(set(r['document_id'] for r in docs))!=len(docs):raise ValueError('Duplicate document IDs')
  report['input_hashes']={'claims':digest(x.claims),'corpus':digest(x.corpus)}
  report['counts']={'claims':len(claims),'documents':len(docs),'claims_with_gold_label':sum(bool(r.get('gold_label','').strip()) for r in claims)}
  report['gates']['custody']='SELF_ATTESTED_NOT_INDEPENDENTLY_VERIFIED' if x.custody_attested else 'NOT_ATTESTED'
  metadata=('source_url','published_at','author','license','collected_at')
  report['metadata_missing_counts']={k:sum(not d.get(k,'').strip() for d in docs) for k in metadata}
  report['gates']['provenance']='METADATA_PRESENT_NOT_AUTHENTICATED' if not any(report['metadata_missing_counts'].values()) else 'INCOMPLETE_AND_UNVERIFIED'
  corpus_norm=collections.Counter(norm(d['text']) for d in docs)
  report['corpus_exact_duplicate_text_groups']=sum(v>1 for v in corpus_norm.values())
  if x.reference_claims:
   ref=read(x.reference_claims);field='claim' if ref and 'claim' in ref[0] else 'claim_text' if ref and 'claim_text' in ref[0] else None
   if not field:raise ValueError('Reference CSV needs claim or claim_text')
   known={norm(r[field]) for r in ref};overlap=[{'claim_id':r['claim_id'],'exact_overlap':int(norm(r['claim']) in known)} for r in claims]
   write(out/'s39_exact_overlap.csv',overlap,['claim_id','exact_overlap']);n=sum(r['exact_overlap'] for r in overlap)
   report['input_hashes']['reference_claims']=digest(x.reference_claims);report['overlap']={'exact_hits':n,'reference_count':len(ref)};report['gates']['exact_overlap']='FAIL' if n else 'SCREEN_PASS_ONLY'
  else:report['gates']['exact_overlap']='NOT_SCREENED'
  # Corpus must be supplied independently; no FEVER gold evidence, gold groups, or labels are read into scoring.
  tok=[norm(d['text']).split() for d in docs];freq=[collections.Counter(t) for t in tok];df=collections.Counter(t for c in freq for t in c);N=len(docs);avg=sum(len(t) for t in tok)/N;retr=[]
  for claim in claims:
   query=set(norm(claim['claim']).split());scores=[]
   for i,c in enumerate(freq):
    score=0.
    for t in query:
     if t not in c:continue
     idf=math.log(1+(N-df[t]+0.5)/(df[t]+0.5));tf=c[t];score+=idf*tf*2.2/(tf+1.2*(0.25+0.75*len(tok[i])/max(avg,1)))
    if score>0:scores.append((score,docs[i]['document_id'],i))
   for rank,(score,_,i) in enumerate(sorted(scores,key=lambda z:(-z[0],z[1]))[:x.top_k],1):
    d=docs[i];retr.append({'claim_id':claim['claim_id'],'rank':rank,'document_id':d['document_id'],'score':format(score,'.12g'),'text':d['text'],'source_url':d.get('source_url',''),'published_at':d.get('published_at',''),'author':d.get('author',''),'license':d.get('license',''),'collected_at':d.get('collected_at','')})
  write(out/'s40_retrieved_evidence.csv',retr,['claim_id','rank','document_id','score','text','source_url','published_at','author','license','collected_at'])
  report['retrieval']={'retrieved_rows':len(retr),'claims_without_positive_hits':len(claims)-len({r['claim_id'] for r in retr})};report['gates']['retrieval']='EXECUTED_NON_ORACLE_NOT_VALIDATED';report['gates']['three_class']='NOT_EVALUATED';report['gates']['independent_validation']='BLOCKED_PENDING_CUSTODY_LABELS_LEAKAGE_AND_EVALUATOR'
  report['warnings']+=['Exact-text screening cannot exclude semantic overlap or training contamination','Metadata field presence does not authenticate a source','No model inference or accuracy was computed','Never expose holdout labels to development or tune on these retrieval outputs']
 report['artifact_hashes']={p.name:digest(p) for p in sorted(out.glob('*.csv'))}
 (out/'s39_s41_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'output':str(out),'gates':report['gates']},indent=2))
if __name__=='__main__':main()
