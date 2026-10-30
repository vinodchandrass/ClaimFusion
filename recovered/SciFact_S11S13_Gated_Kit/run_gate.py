import argparse,csv,hashlib,io,json,zipfile,pathlib,collections,sys

def read_zip_csv(z,name):
 with zipfile.ZipFile(z) as f:
  names=[n for n in f.namelist() if pathlib.PurePosixPath(n).name==name]
  if len(names)!=1: raise RuntimeError(f'Expected one {name}; found {names}')
  return list(csv.DictReader(io.StringIO(f.read(names[0]).decode('utf-8-sig'))))

def main():
 p=argparse.ArgumentParser();p.add_argument('--s7',required=True);p.add_argument('--out',required=True);p.add_argument('--calibration',default='');p.add_argument('--sentence-predictions',default='');a=p.parse_args()
 out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 rows=read_zip_csv(a.s7,'S7_predictions.csv')
 expected={'claim_id','doc_id','gold','pred','p_contradiction','p_entailment','p_neutral','sentence_count'}
 if not rows or not expected.issubset(rows[0]): raise RuntimeError('S7 predictions schema mismatch')
 errors=[];seen=set()
 for i,r in enumerate(rows):
  key=(r['claim_id'],r['doc_id'])
  if key in seen:errors.append(f'duplicate pair {key}')
  seen.add(key)
  if r['gold'] not in ('SUPPORT','CONTRADICT'):errors.append(f'invalid gold row {i}')
  probs=[float(r[k]) for k in ('p_contradiction','p_entailment','p_neutral')]
  if abs(sum(probs)-1)>0.001:errors.append(f'probability sum row {i}')
  if r['pred']!=('CONTRADICT' if probs[0]>probs[1] else 'SUPPORT'):errors.append(f'prediction mismatch row {i}')
 # The evaluation-only file cannot be used to select thresholds.
 calibration_status='MISSING';cal_n=0
 if a.calibration:
  cp=pathlib.Path(a.calibration)
  if not cp.exists():raise RuntimeError('Calibration file not found')
  with cp.open(encoding='utf-8-sig',newline='') as f:cal=list(csv.DictReader(f))
  required={'claim_id','doc_id','gold','p_contradiction','p_entailment','p_neutral','split'}
  if not cal or not required.issubset(cal[0]):raise RuntimeError('Calibration CSV schema missing required fields')
  if any(r['split'] not in ('train_oof','train_calibration') for r in cal):raise RuntimeError('Calibration must be train-only out-of-fold or train-only held-out')
  if any((r['claim_id'],r['doc_id']) in seen for r in cal):raise RuntimeError('Calibration overlaps official development claim-document pairs')
  calibration_status='SUPPLIED_UNVERIFIED_PROVENANCE';cal_n=len(cal)
 sentence_status='MISSING'
 if a.sentence_predictions:
  sp=pathlib.Path(a.sentence_predictions)
  if not sp.exists():raise RuntimeError('Sentence predictions file not found')
  with sp.open(encoding='utf-8-sig',newline='') as f:sr=list(csv.DictReader(f))
  needed={'claim_id','doc_id','sentence_index','p_contradiction','p_entailment','p_neutral'}
  if not sr or not needed.issubset(sr[0]):raise RuntimeError('Sentence predictions schema mismatch')
  sentence_status='SUPPLIED_UNVERIFIED_ALIGNMENT'
 high=[r for r in rows if float(r['p_neutral'])>=0.75]
 wrong=[r for r in rows if r['gold']!=r['pred']]
 highwrong=[r for r in high if r['gold']!=r['pred']]
 report={'phase':'S11-S13_INPUT_GATE','status':'PASS_INPUT_AUDIT_ONLY' if not errors else 'FAIL','evaluation_pairs':len(rows),'distinct_claims':len(set(r['claim_id'] for r in rows)),'error_count':len(wrong),'high_neutral_ge_0_75_count':len(high),'high_neutral_error_count':len(highwrong),'errors':errors,'calibration_status':calibration_status,'calibration_pairs':cal_n,'sentence_probabilities_status':sentence_status,'s11_threshold_freeze':'BLOCKED: requires verified training-only calibration with claim-disjoint partition and precommitted objective','s12_aggregation':'BLOCKED: S7 provides aggregated pair probabilities only; no sentence-level probabilities','s13_perturbation':'BLOCKED: requires frozen perturbation protocol and sentence-level outputs; no simulated reliability labels permitted','data_restriction':'Official dev gold labels used only for audit; no threshold/model selection','source_sha256':hashlib.sha256(pathlib.Path(a.s7).read_bytes()).hexdigest()}
 (out/'S11S13_gate_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (out/'S11S13_required_inputs.md').write_text('# Required to proceed\n\n1. Training-only, claim-disjoint calibration predictions with gold labels and three NLI probabilities; freeze threshold selection objective before evaluating dev.\n2. Sentence-level NLI probabilities with claim_id, doc_id, sentence_index and immutable source/model identifiers for both calibration and evaluation.\n3. A frozen perturbation protocol for missing, duplicated, and contradictory evidence, with original-vs-perturbed pair IDs and fixed seeds.\n4. Original model revision, verified local weight hashes, tokenizer hashes, and inference script hashes.\n\nNo reliability ground truth is available in SciFact; provenance simulation cannot be described as observed source reliability.\n',encoding='utf-8')
 print(json.dumps(report,indent=2))
 if errors:sys.exit(2)
if __name__=='__main__':main()
