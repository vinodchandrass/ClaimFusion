import argparse,csv,json,hashlib,re,collections,datetime,pathlib,itertools
from difflib import SequenceMatcher
EXPECTED={'healthver_train.csv':'414b62c2382ae299c057f4c02fc0b528787303e48f7f33c65e4d6a19331ad359','healthver_dev.csv':'a2120d746171320b54f80dc344b8c707378dd79354b16f97b17048845ccc5239'}
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def norm(x):return ' '.join(re.findall(r'\w+',str(x).casefold()))
def tokenize(x):return set(norm(x).split())
def read_csv(p):
 with p.open(encoding='utf-8-sig',newline='') as f:
  r=csv.DictReader(f);return r.fieldnames or [],list(r)
def read_scifact(p):
 if not p.exists():return [],'NOT_FOUND'
 a=[]
 with p.open(encoding='utf-8') as f:
  for line in f:
   if line.strip():
    obj=json.loads(line);a.append(obj.get('claim',''))
 return a,'READ_DEV_OR_TRAIN_ONLY'
def write_csv(p,rows,fields):
 with p.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser(description='HEALTHVER train/dev only; NEVER reads HEALTHVER test or SciFact test')
 ap.add_argument('--base',default=r'D:\DataLineage');ap.add_argument('--source',default=None);ap.add_argument('--output',default=None);a=ap.parse_args()
 base=pathlib.Path(a.base);source=pathlib.Path(a.source) if a.source else base/'S67_HealthVer_Acquisition'/'HealthVer_official'/'data';out=pathlib.Path(a.output) if a.output else base/'S68_S71_HealthVer_Audit_Output';out.mkdir(parents=True,exist_ok=True)
 report={'phase':'S68-S71','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'SCHEMA_AND_DEVELOPMENT_OVERLAP_AUDIT_ONLY','source_directory':str(source),'heldout_test_files_read':False,'model_inference_performed':False,'independent_validation_authorized':False,'inputs':{},'splits':{},'overlap':{},'limitations':[]}
 data={}
 for split in ['train','dev']:
  p=source/('healthver_'+split+'.csv');info={'path':str(p),'exists':p.is_file()};report['inputs'][split]=info
  if not p.is_file():continue
  info['sha256']=digest(p);info['expected_sha256']=EXPECTED[p.name];info['snapshot_hash_match']=info['sha256']==EXPECTED[p.name]
  if not info['snapshot_hash_match']:
   report['status']='BLOCKED_SOURCE_HASH_MISMATCH';continue
  cols,rows=read_csv(p);data[split]=(cols,rows)
  profile={'rows':len(rows),'columns':cols,'missing_by_column':{c:sum(not str(r.get(c,'')).strip() for r in rows) for c in cols},'unique_by_column':{c:len(set(str(r.get(c,'')) for r in rows)) for c in cols}}
  for c in cols:
   if re.search('label|verdict|stance|class',c,re.I):profile.setdefault('label_distributions',{})[c]=dict(collections.Counter(str(r.get(c,'')) for r in rows).most_common(30))
  report['splits'][split]=profile
 if len(data)!=2:
  report['limitations'].append('Both exact-hash-matching train and dev files required; no inference authorized')
 else:
  cols=data['train'][0]; lower={c.lower():c for c in cols}
  def choose(patterns):
   for pat in patterns:
    matches=[c for c in cols if re.search(pat,c,re.I)]
    if matches:return matches[0]
   return None
  claim_col=choose([r'^claim$',r'claim_text',r'claim']);evidence_col=choose([r'^evidence$',r'evidence_text',r'^sentence$',r'sentence',r'evidence']);label_col=choose([r'^label$',r'^verdict$',r'^stance$',r'label',r'verdict'])
  report['detected_columns']={'claim':claim_col,'evidence':evidence_col,'label':label_col}
  if not claim_col:report['limitations'].append('Claim field not identifiable; manual schema review required')
  else:
   train=data['train'][1];dev=data['dev'][1]
   tclaims={norm(r.get(claim_col,'')) for r in train if norm(r.get(claim_col,''))};dclaims={norm(r.get(claim_col,'')) for r in dev if norm(r.get(claim_col,''))}
   report['overlap']['train_dev_exact_claim_overlap_unique']=len(tclaims & dclaims)
   report['overlap']['train_unique_claims']=len(tclaims);report['overlap']['dev_unique_claims']=len(dclaims)
   if evidence_col:
    te={norm(r.get(evidence_col,'')) for r in train if norm(r.get(evidence_col,''))};de={norm(r.get(evidence_col,'')) for r in dev if norm(r.get(evidence_col,''))}
    report['overlap']['train_dev_exact_evidence_overlap_unique']=len(te&de)
    report['overlap']['train_unique_evidence']=len(te);report['overlap']['dev_unique_evidence']=len(de)
    report['retrieval_feasibility']={'paired_evidence_present':True,'independent_document_corpus_verified':False,'reason':'Pair-level evidence text alone does not establish a non-oracle source-document collection with complete coverage and identifiers'}
   else:report['retrieval_feasibility']={'paired_evidence_present':False,'independent_document_corpus_verified':False}
   ref={}
   for split in ['dev','train']:
    p=base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/('claims_'+split+'.jsonl');items,status=read_scifact(p);ref[split]={'status':status,'path':str(p),'count':len(items),'sha256':digest(p) if p.exists() else None};known={norm(x) for x in items if norm(x)}
    report['overlap']['healthver_'+ 'dev'+'_scifact_'+split+'_exact_unique']=len(dclaims & known)
    report['overlap']['healthver_train_scifact_'+split+'_exact_unique']=len(tclaims & known)
   report['scifact_reference']=ref
   # Near overlap: candidate-generation using token inverted index, no test data.
   reference=[(split,norm(x)) for split in ['dev','train'] for x in read_scifact(base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/('claims_'+split+'.jsonl'))[0] if norm(x)]
   inv=collections.defaultdict(set)
   for i,(_,s) in enumerate(reference):
    for tok in tokenize(s):
     if len(tok)>=4:inv[tok].add(i)
   candidates=[]
   for split,rows in [('healthver_train',train),('healthver_dev',dev)]:
    for idx,r in enumerate(rows):
     s=norm(r.get(claim_col,''));t=tokenize(s)
     if not t:continue
     hit=collections.Counter(j for tok in t if len(tok)>=4 for j in inv.get(tok,()))
     for j,_ in hit.most_common(10):
      refsplit,reftext=reference[j];u=tokenize(reftext);jac=len(t&u)/len(t|u) if t|u else 0
      if jac>=.8:
       candidates.append({'healthver_split':split,'healthver_row_index':idx,'scifact_split':refsplit,'jaccard':round(jac,5),'healthver_claim_sha256':hashlib.sha256(s.encode()).hexdigest(),'scifact_claim_sha256':hashlib.sha256(reftext.encode()).hexdigest()})
   candidates.sort(key=lambda x:(-x['jaccard'],x['healthver_split'],x['healthver_row_index']))
   write_csv(out/'s70_near_overlap_review_candidates.csv',candidates,['healthver_split','healthver_row_index','scifact_split','jaccard','healthver_claim_sha256','scifact_claim_sha256'])
   report['overlap']['near_overlap_candidates_jaccard_ge_0_8']=len(candidates)
   report['limitations'].append('Jaccard candidate screen is NOT a semantic overlap clearance; human assessment remains required')
  report['limitations']+=['HEALTHVER test split intentionally not opened','Public benchmark may have pretraining exposure; contamination not cleared','Evidence-pair labels cannot automatically be interpreted as claim-level labels','No signed independent evaluator or investigator authorization','No new independent performance estimates produced']
 (out/'s71_scientific_gate_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 (out/'s68_schema_summary.json').write_text(json.dumps({'inputs':report['inputs'],'splits':report['splits'],'detected_columns':report.get('detected_columns')},indent=2,ensure_ascii=False),encoding='utf-8')
 print('S68-S71:',report['status'],'| test files read: NO | independent evaluation: BLOCKED')
 print('Report:',out/'s71_scientific_gate_report.json')
if __name__=='__main__':main()
