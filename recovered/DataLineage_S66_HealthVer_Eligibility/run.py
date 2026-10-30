#!/usr/bin/env python3
"""S66 metadata-only external candidate intake. Never opens dataset contents."""
import argparse,csv,hashlib,json,datetime
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--base',default=r'D:\DataLineage');p.add_argument('--candidate-dir',default=None,help='Optional directory containing downloaded HEALTHVER files; filenames and sizes only')
 p.add_argument('--output',default=None)
 a=p.parse_args();base=Path(a.base);out=Path(a.output or base/'S66_HealthVer_Eligibility_Output');out.mkdir(parents=True,exist_ok=True)
 candidates=[]
 if a.candidate_dir:
  root=Path(a.candidate_dir)
  if root.is_dir():
   for path in sorted(root.rglob('*')):
    if path.is_file():
     candidates.append({'relative_path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':'NOT_COMPUTED_CONTENT_SEALED','contents_read':False})
 with (out/'s66_candidate_file_inventory.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['relative_path','bytes','sha256','contents_read']);w.writeheader();w.writerows(candidates)
 # Local research assets: read only previously authorized hashes, not sealed test.
 known={'frozen_policy':base/'DataLineage_S54_Frozen_Operational'/'s53_frozen_policy.json','candidate_cache':base/'S46_S48_Dev_NonOracle_Output'/'s46_sentence_candidates.csv','scifact_dev':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_dev.jsonl','scifact_train':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'claims_train.jsonl','scifact_corpus':base/'ScientificClaim_Benchmark_SciFact'/'data'/'data'/'corpus.jsonl','prior_overlap':base/'SciFact_S20_Output'/'S20_claim_overlap_and_eligibility.csv'}
 local=[{'artifact':k,'exists':v.is_file(),'sha256':sha(v) if v.is_file() else None} for k,v in known.items()]
 with (out/'s66_local_frozen_reference_hashes.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['artifact','exists','sha256']);w.writeheader();w.writerows(local)
 report={'phase':'S66','generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'CANDIDATE_METADATA_INTAKE_ONLY_NOT_AUTHORIZED','candidate_dir_provided':bool(a.candidate_dir),'candidate_file_count':len(candidates),'candidate_contents_read':False,'candidate_hashes_computed':False,'sealed_scifact_test_accessed':False,'model_inference_performed':False,'independent_validation_performed':False,'local_reference_hashes':local,'eligibility_gates':{'source_license_confirmed':False,'independent_evaluator_designated':False,'signed_prior_exposure_attestation':False,'claim_unit_label_semantics_verified':False,'non_oracle_retrieval_corpus_verified':False,'exact_near_semantic_overlap_cleared':False,'pretrained_model_contamination_cleared':False,'independent_authorization':False},'authorized_to_run_holdout':False,'next_requirement':'Independent evaluator obtains source snapshot and performs split-aware schema and overlap review under authorized custody. No evaluation before signed approval.'}
 (out/'s66_eligibility_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('S66 metadata intake complete; candidate files:',len(candidates),'\nHOLDOUT EXECUTION: BLOCKED\nOutput:',out)
if __name__=='__main__':main()
