import argparse,collections,hashlib,json,pathlib,urllib.request,datetime
SOURCES={'paper_dev':'https://fever.ai/download/fever/paper_dev.jsonl','paper_test':'https://fever.ai/download/fever/paper_test.jsonl','shared_task_dev':'https://fever.ai/download/fever/shared_task_dev.jsonl'}
LABELS={'SUPPORTS','REFUTES','NOT ENOUGH INFO'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def acquire(dest):
 dest.mkdir(parents=True,exist_ok=True)
 for name,url in SOURCES.items():
  target=dest/(name+'.jsonl')
  if target.exists():print('PRESERVE_EXISTING',target);continue
  tmp=target.with_suffix('.download')
  try:
   with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'ResearchIntegrityAudit/1.0'}),timeout=120) as r, tmp.open('wb') as out:
    import shutil;shutil.copyfileobj(r,out)
   if tmp.stat().st_size==0:raise ValueError('empty download')
   tmp.replace(target);print('DOWNLOADED',target,sha(target))
  except Exception as e:
   tmp.unlink(missing_ok=True);print('DOWNLOAD_FAILED',name,str(e));raise

def audit(inp,out):
 out.mkdir(parents=True,exist_ok=True); report={'status':'PASS_SCHEMA_ONLY_NOT_EVALUATED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':{},'overlaps':{},'warnings':[]};sets={}
 for name in SOURCES:
  p=inp/(name+'.jsonl')
  if not p.exists():report['warnings'].append('MISSING '+str(p));continue
  counts=collections.Counter();ids=set();texts=set();dupid=0;duptext=0;bad=[];evidence_groups=0;missing_evidence=0
  with p.open(encoding='utf-8') as f:
   for i,line in enumerate(f,1):
    try:
     r=json.loads(line);assert isinstance(r,dict);assert isinstance(r.get('claim'),str) and r['claim'].strip();assert r.get('label') in LABELS;assert 'id' in r
     ident=str(r['id']);norm=' '.join(r['claim'].casefold().split())
     if ident in ids:dupid+=1
     if norm in texts:duptext+=1
     ids.add(ident);texts.add(norm);counts[r['label']]+=1
     ev=r.get('evidence')
     if not isinstance(ev,list):missing_evidence+=1
     else:
      for group in ev:
       if not isinstance(group,list):raise ValueError('evidence group not list')
       evidence_groups+=1
       for item in group:
        if not isinstance(item,list) or len(item)!=4:raise ValueError('evidence tuple shape')
    except Exception as e:
     if len(bad)<15:bad.append({'line':i,'error':str(e)})
  report['files'][name]={'path':str(p),'sha256':sha(p),'size_bytes':p.stat().st_size,'valid_rows':sum(counts.values()),'labels':dict(counts),'duplicate_ids':dupid,'duplicate_normalized_claims':duptext,'evidence_groups':evidence_groups,'missing_evidence_field':missing_evidence,'errors':bad}
  sets[name]=(ids,texts)
  if bad or dupid:report['status']='FAIL_INTEGRITY'
 names=list(sets)
 for i,a in enumerate(names):
  for b in names[i+1:]:report['overlaps'][a+'__'+b]={'claim_id_overlap':len(sets[a][0]&sets[b][0]),'normalized_text_overlap':len(sets[a][1]&sets[b][1])}
 if not sets:report['status']='BLOCKED_MISSING_DATA'
 elif report['warnings'] and report['status']!='FAIL_INTEGRITY':report['status']='PARTIAL_DATA'
 (out/'S30_integrity_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(report['status']);print(json.dumps({'files':{k:v['valid_rows'] for k,v in report['files'].items()},'overlaps':report['overlaps']},indent=2))
 return 1 if report['status']=='FAIL_INTEGRITY' else 0
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['acquire','audit']);p.add_argument('--input',type=pathlib.Path);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
 if a.command=='acquire':acquire(a.output)
 else:
  if not a.input:p.error('--input required for audit')
  raise SystemExit(audit(a.input,a.output))
