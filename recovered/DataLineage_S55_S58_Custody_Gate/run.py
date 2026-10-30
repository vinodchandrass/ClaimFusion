import argparse,csv,hashlib,json,pathlib,datetime,collections,re

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',str(s).casefold()).strip()
def read_claims(p):
 if p.suffix.lower()=='.jsonl':
  with p.open(encoding='utf-8-sig') as f:
   for line in f:
    if line.strip():
     x=json.loads(line);yield str(x.get('id',x.get('claim_id',''))),norm(x.get('claim',''))
 elif p.suffix.lower()=='.csv':
  with p.open(encoding='utf-8-sig',newline='') as f:
   for x in csv.DictReader(f):yield str(x.get('id',x.get('claim_id',''))),norm(x.get('claim',x.get('claim_text','')))
 else:raise ValueError('Claims must be JSONL or CSV')
def main():
 a=argparse.ArgumentParser(description='S55-S58 read-only local evidence and custody audit; NEVER reads test content')
 a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--out');a.add_argument('--external-manifest',help='Optional separately authorized non-test claims CSV/JSONL for exact overlap screening')
 x=a.parse_args();base=pathlib.Path(x.base);out=pathlib.Path(x.out) if x.out else base/'S55_S58_Custody_Gate_Output';out.mkdir(parents=True,exist_ok=True)
 assets={
 'frozen_policy':base/'DataLineage_S54_Frozen_Operational'/'s53_frozen_policy.json',
 'frozen_transitions':base/'DataLineage_S54_Frozen_Operational'/'s53_paired_transitions.csv',
 'candidate_cache':base/'S46_S48_Dev_NonOracle_Output'/'s46_sentence_candidates.csv',
 'retrieval_cache':base/'S42_S44_SciFact_Dev_Output'/'s42_dev_retrieval.csv',
 'scifact_dev':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_dev.jsonl',
 'scifact_train':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_train.jsonl',
 'scifact_corpus':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'corpus.jsonl',
 'scifact_overlap_prior':base/'SciFact_S20_Output'/'S20_claim_overlap_and_eligibility.csv',
 'model_config':base/'S7_NLI_Model'/'config.json',
 'model_weights_safetensors':base/'S7_NLI_Model'/'model.safetensors',
 'model_weights_bin':base/'S7_NLI_Model'/'pytorch_model.bin',
 }
 inventory=[]
 for name,p in assets.items():
  exists=p.is_file();inventory.append({'asset':name,'path':str(p),'exists':exists,'size_bytes':p.stat().st_size if exists else None,'sha256':sha(p) if exists else None})
 # Test: metadata-only existence check, absolutely no content reads, hashes, or previews.
 test=base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_test.jsonl'
 test_metadata={'path':str(test),'exists':test.is_file(),'content_read':False,'hash_computed':False}
 policy_path=assets['frozen_policy'];policy=json.loads(policy_path.read_text(encoding='utf-8')) if policy_path.is_file() else None
 candidate=next((i for i in inventory if i['asset']=='candidate_cache'),None)
 parity={'status':'NOT_CHECKED','reason':'No expected hash or missing cache'}
 if candidate and candidate['exists']:
  expected='b6724c628aebf5b112fe3c8ca74fb59eddfad1b0e1f899eaf9b300190a8dc41d'
  parity={'status':'PASS' if candidate['sha256']==expected else 'FAIL','expected':expected,'observed':candidate['sha256']}
 overlap={'status':'NOT_RUN','note':'External manifest not provided; sealed test is never opened'}
 if x.external_manifest:
  p=pathlib.Path(x.external_manifest).resolve()
  if not p.is_file():raise FileNotFoundError(p)
  if 'test' in p.name.lower() or p==test.resolve():raise ValueError('Refusing test-like manifest; use independent evaluator custody')
  dev=assets['scifact_dev'];train=assets['scifact_train']
  if not (dev.is_file() and train.is_file()):raise FileNotFoundError('Train and dev claims needed for overlap screen')
  refs=collections.defaultdict(set)
  for group,q in [('train',train),('dev',dev)]:
   for cid,s in read_claims(q):
    if s:refs[s].add(group)
  n=0;hits=[];unique=set()
  for cid,s in read_claims(p):
   n+=1
   if s in refs:hits.append({'claim_id':cid,'overlap_splits':';'.join(sorted(refs[s]))})
   unique.add(s)
  overlap={'status':'SCREENED_NOT_CERTIFIED','external_manifest':str(p),'manifest_sha256':sha(p),'claim_count':n,'unique_normalized_claims':len(unique),'exact_overlap_count':len(hits),'semantic_overlap_checked':False,'training_contamination_checked':False,'prior_human_exposure_checked':False}
  with (out/'s57_exact_overlap.csv').open('w',encoding='utf-8',newline='') as f:
   w=csv.DictWriter(f,fieldnames=['claim_id','overlap_splits']);w.writeheader();w.writerows(hits)
 with (out/'s55_artifact_inventory.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['asset','path','exists','size_bytes','sha256']);w.writeheader();w.writerows(inventory)
 gates={
  'frozen_policy_present':bool(policy),
  'frozen_candidate_hash_parity':parity['status']=='PASS',
  'external_manifest_exact_overlap_screen':overlap['status']=='SCREENED_NOT_CERTIFIED',
  'external_evaluator_custody_attested':False,
  'human_prior_exposure_attested':False,
  'semantic_overlap_certified':False,
  'training_data_contamination_cleared':False,
  'three_class_NEI_policy_frozen':False,
  'genuine_provenance_verified':False,
 }
 report={'status':'DEVELOPMENT_FREEZE_AUDIT_ONLY_EXTERNAL_EVALUATION_BLOCKED','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(base),'test_metadata_only':test_metadata,'policy_name':policy.get('policy_name') if policy else None,'candidate_hash_parity':parity,'external_overlap':overlap,'gates':gates,'authorization_to_run_sealed_test':False,'model_inference_performed':False,'test_content_read':False}
 (out/'s58_gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (out/'s58_independent_evaluator_handoff.md').write_text('''# Independent evaluator handoff — NOT AN AUTHORIZATION\n\n1. Preserve frozen policy and input hashes. Document who had access to all prior dev/test claims.\n2. Independently acquire claims and corpus with recorded source/version/license and label protocol.\n3. Perform exact and semantic overlap, prior-human-exposure and model-training contamination checks.\n4. Register target task: current policy is binary SUPPORTS/REFUTES; no NEI classifier exists.\n5. Predefine metrics, abstention handling, uncertainty, subgroup analysis and comparator BEFORE labels are accessed.\n6. Keep evaluator-controlled gold labels unavailable to development team and prediction code.\n7. Do not read existing SciFact test or FEVER test without separate verified custody and authorization.\n8. Re-run frozen pipeline on authorized unlabeled claims, hash predictions, and have evaluator score them independently.\n\nCurrent gate: BLOCKED. Presence of an external file or an overlap screen is not custody attestation.\n''',encoding='utf-8')
 print(json.dumps({'status':report['status'],'out':str(out),'candidate_parity':parity['status'],'assets_present':sum(i['exists'] for i in inventory),'test_content_read':False},indent=2))
if __name__=='__main__':main()
