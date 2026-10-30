import argparse,csv,hashlib,json,subprocess,sys,datetime
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--repo-url',default='https://github.com/sarrouti/HealthVer.git');a.add_argument('--skip-clone',action='store_true');z=a.parse_args()
 base=Path(z.base);root=base/'S67_HealthVer_Acquisition';repo=root/'HealthVer_official';out=base/'S67_HealthVer_Acquisition_Output';out.mkdir(parents=True,exist_ok=True);root.mkdir(parents=True,exist_ok=True)
 report={'phase':'S67','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'repository_url':z.repo_url,'repository_path':str(repo),'download_status':'NOT_ATTEMPTED','commit':None,'data_files':0,'no_test_labels_read':True,'no_model_inference':True,'independent_evaluation_authorized':False,'limitations':['Dataset provenance and local inventory do not establish independence','Public data can have prior training exposure','No external holdout scoring permitted']}
 if not z.skip_clone:
  try:
   if (repo/'.git').exists(): report['download_status']='EXISTING_REPOSITORY_NOT_UPDATED'
   elif repo.exists() and any(repo.iterdir()): report['download_status']='NONEMPTY_DIRECTORY_NOT_GIT'
   else:
    p=subprocess.run(['git','clone','--depth','1',z.repo_url,str(repo)],capture_output=True,text=True,timeout=240)
    report['download_status']='CLONED' if p.returncode==0 else 'CLONE_FAILED';report['clone_error']=p.stderr[-1200:] if p.returncode else None
  except Exception as e:report['download_status']='CLONE_ERROR';report['clone_error']=str(e)
 if (repo/'.git').exists():
  try:report['commit']=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True,timeout=15).strip()
  except Exception as e:report['commit_error']=str(e)
 rows=[]
 if repo.exists():
  for p in sorted(repo.rglob('*')):
   if not p.is_file() or '.git' in p.parts:continue
   # metadata only; do not parse or inspect labels/records
   rows.append({'relative_path':p.relative_to(repo).as_posix(),'size_bytes':p.stat().st_size,'sha256':sha(p),'data_directory':p.relative_to(repo).parts[0].lower()=='data'})
 report['files_in_inventory']=len(rows);report['data_files']=sum(r['data_directory'] for r in rows)
 report['status']='SNAPSHOT_INVENTORIED_NOT_ELIGIBILITY_CLEARED' if report['data_files'] else 'SOURCE_DATA_NOT_AVAILABLE'
 with (out/'s67_source_file_inventory.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['relative_path','size_bytes','sha256','data_directory']);w.writeheader();w.writerows(rows)
 (out/'s67_acquisition_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('S67:',report['status'],'files:',len(rows),'data files:',report['data_files']);print('Report:',out/'s67_acquisition_report.json')
if __name__=='__main__':main()
