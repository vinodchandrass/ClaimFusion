import argparse,csv,json,hashlib,collections,pathlib,datetime

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def rows(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,records,fields):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(records)
def replay(candidates):
 groups=collections.defaultdict(list)
 for r in candidates:groups[r['claim_id']].append(r)
 out=[]
 for cid,items in sorted(groups.items(),key=lambda z:int(z[0]) if z[0].isdigit() else z[0]):
  top=[r for r in items if int(r['doc_rank'])==1]
  if len(top)!=3:raise ValueError(f'{cid}: expected exactly 3 sentences in rank 1, got {len(top)}')
  ec=sum(float(r['p_entailment']) for r in top)/3
  cc=sum(float(r['p_contradiction']) for r in top)/3
  out.append({'claim_id':cid,'predicted_label':'SUPPORTS' if ec>cc else 'REFUTES','mean_entailment':f'{ec:.10f}','mean_contradiction':f'{cc:.10f}','selected_doc_id':top[0]['doc_id'],'decision_margin':f'{abs(ec-cc):.10f}'})
 return out

def main():
 p=argparse.ArgumentParser(description='S54 frozen replay and independent eligibility gates; no test data access')
 p.add_argument('--candidates',required=True);p.add_argument('--frozen-policy',required=True);p.add_argument('--transitions');p.add_argument('--output',required=True);p.add_argument('--external-manifest',help='OPTIONAL metadata-only CSV, no claim text or labels');p.add_argument('--custody-attestation',help='OPTIONAL independent signed attestation JSON');x=p.parse_args()
 dest=pathlib.Path(x.output);dest.mkdir(parents=True,exist_ok=True)
 policy=json.loads(pathlib.Path(x.frozen_policy).read_text(encoding='utf-8'))
 if policy.get('policy_name')!='top_doc_mean':raise ValueError('policy mismatch')
 candidates=rows(x.candidates)
 required={'claim_id','doc_id','doc_rank','p_entailment','p_contradiction'}
 if not candidates or not required.issubset(candidates[0]):raise ValueError('candidate columns missing')
 predictions=replay(candidates)
 write(dest/'s54_frozen_operational_predictions.csv',predictions,list(predictions[0]))
 report={'status':'DEVELOPMENT_REPLAY_NOT_INDEPENDENT_VALIDATION','candidate_count':len(candidates),'claim_count':len(predictions),'candidate_sha256':sha(x.candidates),'policy_sha256':sha(x.frozen_policy),'frozen_policy_name':policy['policy_name'],'test_read':False,'custody_authorized':False,'independent_validation_executed':False,'warnings':['Development-selected policy; previous development accuracy is selection-biased','Binary only; no NEI claim inference','SciFact provenance/source independence not established']}
 if x.transitions:
  t=rows(x.transitions);lookup={r['claim_id']:r for r in t}
  if not set(lookup).issubset(set(r['claim_id'] for r in predictions)):raise ValueError('transition IDs missing from candidates')
  # No gold annotations are read into operational replay; transition CSV is separate diagnostic audit.
  count=collections.Counter()
  for r in predictions:
   if r['claim_id'] not in lookup:continue
   old=lookup[r['claim_id']];actual=old['annotation'];correct=int(r['predicted_label']==actual)
   if correct!=int(old['candidate_correct']):raise ValueError('Frozen policy parity failure claim '+r['claim_id'])
   count['correct']+=correct;count['total']+=1
  report['dev_diagnostic_parity']={'status':'PASS','correct':count['correct'],'n':count['total'],'accuracy':count['correct']/count['total'],'transitions_sha256':sha(x.transitions)}
 if x.external_manifest:
  m=rows(x.external_manifest)
  safe={'dataset_id','split_id','record_count','corpus_sha256','claims_sha256','source','acquired_at','custodian','disjointness_report_sha256','exposure_attestation_sha256'}
  if m and set(m[0])-safe:raise ValueError('Manifest contains unsupported fields; do not pass claim text or labels')
  report['external_manifest']={'row_count':len(m),'sha256':sha(x.external_manifest),'metadata_only':True,'validation_eligible':False}
 if x.custody_attestation:
  a=json.loads(pathlib.Path(x.custody_attestation).read_text(encoding='utf-8'))
  required_att={'independent_custodian','signed_at','claim_disjointness_verified','semantic_overlap_reviewed','no_prior_exposure_attested','sealed_labels_unread_by_developers'}
  missing=sorted(required_att-set(a))
  report['custody_attestation']={'sha256':sha(x.custody_attestation),'missing_fields':missing,'review_required':True}
 (dest/'s54_replay_and_gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print('S54 replay complete:',report['claim_count'],'claims; independent evaluation NOT performed')
 if 'dev_diagnostic_parity' in report:print('Frozen parity:',report['dev_diagnostic_parity'])
if __name__=='__main__':main()
