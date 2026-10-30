import argparse,csv,hashlib,json,pathlib,datetime,collections
P=pathlib.Path

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');a.add_argument('--out',default=None);x=a.parse_args()
 base=P(x.base);out=P(x.out) if x.out else base/'S33_S35_Validation_Output';out.mkdir(parents=True,exist_ok=True)
 model=base/'S7_NLI_Model';evidence=base/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv'
 required=[model/'config.json',model/'tokenizer.json',model/'model.safetensors',evidence]
 report={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'S33-S35','inputs':{},'gates':{},'limitations':[]}
 for p in required:
  report['inputs'][str(p)]={'exists':p.is_file(),'bytes':p.stat().st_size if p.is_file() else None,'sha256':sha(p) if p.is_file() else None}
 report['gates']['required_inputs']='PASS' if all(p.is_file() for p in required) else 'BLOCKED'
 if (model/'config.json').is_file():
  cfg=json.loads((model/'config.json').read_text(encoding='utf-8-sig'));labels={str(k):v for k,v in cfg.get('id2label',{}).items()};report['model_labels']=labels
  report['gates']['nli_label_mapping']='PASS' if labels=={'0':'contradiction','1':'entailment','2':'neutral'} else 'BLOCKED'
 if evidence.is_file():
  with evidence.open(encoding='utf-8-sig',newline='') as f:
   reader=csv.DictReader(f);report['evidence_columns']=reader.fieldnames;report['evidence_row_count']=sum(1 for _ in reader)
 report['gates']['independent_scifact_confirmation']='BLOCKED_HUMAN_EXPOSURE_ATTESTATION_REQUIRED'
 report['gates']['fever_model_inference']='NOT_RUN'
 report['gates']['fusion_ablations']='NOT_RUN'
 report['gates']['paired_statistics']='NOT_RUN'
 report['limitations']=['Inventory-only handoff contains no model weights or resolved evidence rows. Local files must be checked before inference.','SciFact prior test exposure prevents treating historical results as untouched confirmatory evidence.','No model accuracy, calibration, ablation, or statistical superiority is asserted by this audit.']
 (out/'S33_S35_preflight_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps({'report':str(out/'S33_S35_preflight_report.json'),'gates':report['gates'],'evidence_rows':report.get('evidence_row_count')},indent=2))
if __name__=='__main__':main()
