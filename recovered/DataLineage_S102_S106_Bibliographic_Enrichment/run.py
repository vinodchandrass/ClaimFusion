import argparse,csv,json,re,time,hashlib,urllib.parse,urllib.request,urllib.error,unicodedata,pathlib,datetime,difflib

def norm(s):
 s=unicodedata.normalize('NFKC',str(s or '')).casefold(); return ' '.join(re.findall(r'[a-z0-9]+',s))
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def api(title,email,timeout):
 params={'query.bibliographic':title,'rows':3,'select':'DOI,title,author,published,container-title,type,URL,score'}
 if email:params['mailto']=email
 url='https://api.crossref.org/works?'+urllib.parse.urlencode(params)
 req=urllib.request.Request(url,headers={'User-Agent':'DataLineage-ProvenanceAudit/1.0 (mailto:'+ (email or 'unspecified') +')','Accept':'application/json'})
 with urllib.request.urlopen(req,timeout=timeout) as r:return json.load(r)['message']['items']
def classify(title,items):
 ranked=[]
 for item in items:
  cand=(item.get('title') or [''])[0]; similarity=difflib.SequenceMatcher(None,norm(title),norm(cand)).ratio()
  ranked.append((similarity,item,cand))
 ranked.sort(key=lambda x:x[0],reverse=True)
 if not ranked:return {'status':'NO_MATCH','doi':'','similarity':0,'second_similarity':0,'matched_title':'','source_reference':''},[]
 best=ranked[0]; second=ranked[1][0] if len(ranked)>1 else 0
 doi=best[1].get('DOI',''); valid=bool(re.match(r'^10\.\d{4,9}/\S+$',doi,re.I))
 # Auto candidate is NOT a human-reviewed verification.
 if norm(title)==norm(best[2]) and valid and (best[0]-second>=0.03):status='EXACT_TITLE_CANDIDATE'
 elif best[0]>=.97 and valid and (best[0]-second>=.08):status='HIGH_SIMILARITY_CANDIDATE'
 else:status='MANUAL_REVIEW_REQUIRED'
 result={'status':status,'doi':doi if valid else '', 'similarity':round(best[0],6),'second_similarity':round(second,6),'matched_title':best[2],'source_reference':best[1].get('URL','')}
 return result,[{'title':r[2],'doi':r[1].get('DOI',''),'similarity':round(r[0],6)} for r in ranked]
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--email',default='');p.add_argument('--limit',type=int,default=100);p.add_argument('--delay',type=float,default=1.0);p.add_argument('--timeout',type=int,default=25);p.add_argument('--offline',action='store_true');a=p.parse_args()
 src=pathlib.Path(a.manifest);out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 with src.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
 if not {'doc_id','title'}.issubset(rows[0]):raise SystemExit('manifest missing doc_id/title')
 if len({r['doc_id'] for r in rows})!=len(rows):raise SystemExit('duplicate doc_id')
 state=out/'s103_lookup_records.jsonl'; prior={}
 if state.exists():
  for line in state.read_text(encoding='utf8').splitlines():
   try:
    v=json.loads(line);prior[v['doc_id']]=v
   except (ValueError,KeyError):pass
 processed=0
 for row in rows:
  did=row['doc_id'];title=row['title'];key=hashlib.sha256(title.encode()).hexdigest()
  if did in prior and prior[did].get('title_sha256')==key:continue
  if a.offline or processed>=a.limit:break
  try:
   items=api(title,a.email,a.timeout);r,cands=classify(title,items)
   rec={'doc_id':did,'title_sha256':key,'query_title':title,**r,'candidates':cands,'lookup_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'api':'Crossref REST /works'}
  except (urllib.error.URLError,TimeoutError,ValueError,KeyError) as e:
   rec={'doc_id':did,'title_sha256':key,'query_title':title,'status':'LOOKUP_ERROR','error':str(e),'doi':'','similarity':0,'second_similarity':0,'matched_title':'','source_reference':''}
  with state.open('a',encoding='utf8') as f:f.write(json.dumps(rec,ensure_ascii=False)+'\n')
  prior[did]=rec;processed+=1
  if a.delay>0:time.sleep(a.delay)
 fields=['doc_id','title','doi_candidate','match_status','title_similarity','second_similarity','matched_title','source_reference','manual_review_decision','reviewer','review_date']
 with (out/'s104_bibliographic_candidates.csv').open('w',encoding='utf8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for row in rows:
   v=prior.get(row['doc_id'],{});v=v if v.get('title_sha256')==hashlib.sha256(row['title'].encode()).hexdigest() else {}
   w.writerow({'doc_id':row['doc_id'],'title':row['title'],'doi_candidate':v.get('doi',''),'match_status':v.get('status','NOT_QUERIED'),'title_similarity':v.get('similarity',''),'second_similarity':v.get('second_similarity',''),'matched_title':v.get('matched_title',''),'source_reference':v.get('source_reference',''),'manual_review_decision':'','reviewer':'','review_date':''})
 counts={}
 for row in rows:
  v=prior.get(row['doc_id'],{});s=v.get('status','NOT_QUERIED') if v.get('title_sha256')==hashlib.sha256(row['title'].encode()).hexdigest() else 'NOT_QUERIED';counts[s]=counts.get(s,0)+1
 report={'stage':'S102-S106','manifest_sha256':sha(src),'manifest_rows':len(rows),'records_newly_queried_this_run':processed,'statuses':counts,'verified_doi_count':0,'independent_source_relationships_verified':0,'test_accessed':False,'gold_labels_accessed':False,'accuracy_evaluated':False,'gate':'BIBLIOGRAPHIC_CANDIDATES_ONLY_HUMAN_REVIEW_REQUIRED','limitations':['Crossref title match is a bibliographic candidate, not proof of evidence reliability or source independence.','No DOI is promoted to verified status without separate documented review.','Different DOIs do not imply independent evidence.','Crossref coverage is incomplete; missing results do not prove absence of a DOI.']}
 (out/'s106_bibliographic_gate.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
