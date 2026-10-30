#!/usr/bin/env python3
"""S64 metadata-only fail-closed independent evaluation preflight. No dataset reads."""
import argparse,csv,hashlib,json,datetime,pathlib
REQUIRED=['dataset_id','version','source_url_or_reference','license','annotation_protocol','claim_count_metadata_only','eligible_count_metadata_only','excluded_count_metadata_only','claim_classes','independent_custodian','manifest_sha256','eligibility_report_reference','approval_status']
ATTEST=['independent_custody','developer_non_exposure','historical_exposure_review','exact_overlap_clearance','semantic_overlap_clearance','training_contamination_assessed','dataset_rights_verified','frozen_artifact_verification','prospective_protocol_frozen','non_oracle_retrieval','access_logs_retained']
YES={'YES'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--manifest');a.add_argument('--attestation');a.add_argument('--artifact-inventory');a.add_argument('--out');x=a.parse_args()
 base=pathlib.Path(x.base);out=pathlib.Path(x.out) if x.out else base/'S64_Preflight_Output';out.mkdir(parents=True,exist_ok=True)
 issues=[];checks={};metadata={};files={}
 inv=pathlib.Path(x.artifact_inventory) if x.artifact_inventory else base/'S59_S62_Handoff_Output'/'s59_frozen_hash_verification.csv'
 if inv.exists():
  with inv.open(encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
  if not rows:issues.append('Artifact inventory empty')
  for r in rows:
   p=pathlib.Path(r.get('path',''));expected=r.get('expected_sha256','').lower();actual=sha(p) if p.is_file() and p.name.lower()!='claims_test.jsonl' else None
   ok=bool(actual and expected and actual==expected)
   files[r.get('asset','unknown')]=dict(path=str(p),expected_sha256=expected,observed_sha256=actual,pass_hash=ok)
   if not ok:issues.append('Artifact hash not verified: '+r.get('asset','unknown'))
  checks['artifact_hashes_verified']=bool(rows) and all(z['pass_hash'] for z in files.values())
 else:checks['artifact_hashes_verified']=False;issues.append('S59 artifact inventory missing')
 manifest=pathlib.Path(x.manifest) if x.manifest else None
 if manifest and manifest.is_file():
  with manifest.open(encoding='utf-8-sig',newline='') as f:mr=list(csv.DictReader(f))
  if len(mr)!=1:issues.append('Dataset manifest must contain exactly one record')
  elif any(not mr[0].get(k,'').strip() for k in REQUIRED):issues.append('Dataset manifest missing required metadata')
  else:
   metadata={k:mr[0][k] for k in REQUIRED};checks['dataset_manifest_complete']=True
   if metadata['approval_status'].upper()!='APPROVED_BY_INDEPENDENT_CUSTODIAN':issues.append('Dataset manifest not independently approved')
   try:
    n=int(metadata['claim_count_metadata_only']);e=int(metadata['eligible_count_metadata_only']);z=int(metadata['excluded_count_metadata_only'])
    if min(n,e,z)<0 or e+z!=n:issues.append('Dataset count consistency failure')
   except ValueError:issues.append('Dataset counts are not integers')
 else:issues.append('Independent dataset manifest not supplied')
 checks.setdefault('dataset_manifest_complete',False)
 att=pathlib.Path(x.attestation) if x.attestation else None
 if att and att.is_file():
  try:
   d=json.loads(att.read_text(encoding='utf-8'));answers=d.get('declarations',{});missing=[k for k in ATTEST if answers.get(k,{}).get('answer','').upper() not in YES or not answers.get(k,{}).get('evidence_reference','').strip()]
   checks['attestation_complete']=not missing and bool(d.get('evaluator_name','').strip()) and bool(d.get('signed_date','').strip()) and bool(d.get('signature_reference','').strip())
   if not checks['attestation_complete']:issues.append('Attestation incomplete or lacks evidence: '+', '.join(missing))
   checks['independent_investigator_authorization_recorded']=bool(d.get('investigator_authorization_reference','').strip())
  except (ValueError,TypeError,AttributeError) as exc:issues.append('Invalid attestation JSON: '+str(exc));checks['attestation_complete']=False
 else:issues.append('Signed evaluator attestation not supplied');checks['attestation_complete']=False
 checks.setdefault('independent_investigator_authorization_recorded',False)
 # S64 is intentionally NEVER an execution permit, even if metadata checks pass.
 checks['sealed_test_content_read']=False;checks['model_inference_performed']=False
 report=dict(phase='S64',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PREFLIGHT_METADATA_ONLY_NOT_EXECUTION_AUTHORIZATION',checks=checks,issues=issues,artifact_verification=files,dataset_manifest_metadata=metadata,readiness_for_independent_human_review=not issues and all(checks.get(k,False) for k in ('artifact_hashes_verified','dataset_manifest_complete','attestation_complete')),authorized_to_execute=False,independent_evaluation_performed=False,notice='Even a clean preflight requires separately documented independent investigator approval; no holdout data or labels are opened by this tool.')
 (out/'s64_preflight_report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
 with (out/'s64_artifact_hash_checks.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['asset','path','expected_sha256','observed_sha256','pass_hash']);w.writeheader();[w.writerow(dict(asset=k,**v)) for k,v in files.items()]
 print('S64:', 'READY_FOR_HUMAN_REVIEW_ONLY' if report['readiness_for_independent_human_review'] else 'BLOCKED');print('Verified artifacts:',sum(z['pass_hash'] for z in files.values()),'/',len(files));print('Execution authorization: FALSE');print('Report:',out/'s64_preflight_report.json')
if __name__=='__main__':main()
