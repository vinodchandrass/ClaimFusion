import argparse,csv,hashlib,io,json,math,zipfile
from pathlib import Path
from collections import Counter,defaultdict
def h(b):return hashlib.sha256(b).hexdigest()
def load_jsonl(z,name):return [json.loads(x) for x in z.read(name).decode('utf-8-sig').splitlines() if x.strip()]
def savecsv(p,rows,cols):
    with open(p,'w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input-zip',required=True);ap.add_argument('--s20-zip',required=True)
    ap.add_argument('--model',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--batch-size',type=int,default=8)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.input_zip) as z:
        claims=load_jsonl(z,'claims.jsonl')
        docs={str(r['doc_id']):r for r in load_jsonl(z,'corpus_candidates.jsonl')}
        input_hash={n:h(z.read(n)) for n in ['claims.jsonl','claims_metadata.jsonl','corpus_candidates.jsonl','retrievals.jsonl']}
    with zipfile.ZipFile(a.s20_zip) as z:
        gate=json.loads(z.read('S20_structural_gate.json'))
    if gate.get('status')!='PASS_STRUCTURAL_AUDIT_HUMAN_EXPOSURE_ATTESTATION_REQUIRED':raise RuntimeError('S20 gate mismatch')
    if h(Path(a.input_zip).read_bytes())!=gate['s18_archive_sha256']:raise RuntimeError('S18 archive SHA256 differs from frozen S20 input')
    if len(claims)!=gate['counts']['claims']:raise RuntimeError('S20 claim count mismatch')
    items=[];all_keys=set()
    for c in claims:
        cid=str(c['id'])
        for did,ev in c.get('evidence',{}).items():
            did=str(did);d=docs.get(did)
            if not d:raise RuntimeError('Missing document '+did)
            inds=ev.get('sentences',[])
            abstract=d.get('abstract')
            if not isinstance(abstract,list) or not abstract or not all(isinstance(t,str) for t in abstract):
                raise RuntimeError('Invalid or empty abstract '+cid+'/'+did)
            if not isinstance(inds,list) or any(type(i) is not int or i<0 or i>=len(abstract) for i in inds):
                raise RuntimeError('Invalid evidence sentence index '+cid+'/'+did)
            evidence_mode='annotated_sentences' if inds else 'full_abstract_fallback'
            if ev.get('label') not in ('SUPPORT','CONTRADICT'):raise RuntimeError('Unsupported annotation label')
            k=(cid,did)
            if k in all_keys:raise RuntimeError('Duplicate evidence key')
            all_keys.add(k)
            # Gold sentence indices used for oracle evidence selection ONLY, not as predicted labels.
            passage=' '.join(abstract[i] for i in sorted(set(inds))) if inds else ' '.join(abstract)
            items.append({'claim_id':cid,'doc_id':did,'claim':c['claim'],'evidence_text':passage,
                          'gold_document_label':ev['label'],'provenance':ev.get('provenance','unknown'),
                          'gold_sentence_indices':','.join(map(str,inds)), 'evidence_mode':evidence_mode})
    if len(items)!=gate['counts']['evidence_pairs']:raise RuntimeError('S20 evidence count mismatch')
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    import torch
    modeldir=Path(a.model)
    if not modeldir.is_dir():raise RuntimeError('Local model directory missing')
    tok=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(modeldir),local_files_only=True)
    model.eval()
    ids={str(v).lower():int(k) for k,v in model.config.id2label.items()}
    if not {'contradiction','entailment','neutral'}.issubset(ids):raise RuntimeError('Unexpected model label mapping')
    for start in range(0,len(items),a.batch_size):
        batch=items[start:start+a.batch_size]
        enc=tok([r['claim'] for r in batch],[r['evidence_text'] for r in batch],
                padding=True,truncation=True,max_length=512,return_tensors='pt')
        with torch.inference_mode():p=torch.softmax(model(**enc).logits,dim=-1).cpu().numpy()
        for r,pr in zip(batch,p):
            r['p_contradict']=float(pr[ids['contradiction']])
            r['p_support']=float(pr[ids['entailment']])
            r['p_neutral']=float(pr[ids['neutral']])
            r['pred_document_label']='SUPPORT' if r['p_support']>=r['p_contradict'] else 'CONTRADICT'
            r['correct']=int(r['pred_document_label']==r['gold_document_label'])
    columns=['claim_id','doc_id','provenance','gold_document_label','gold_sentence_indices','evidence_mode',
             'p_contradict','p_support','p_neutral','pred_document_label','correct']
    savecsv(out/'S21_oracle_document_predictions.csv',items,columns)
    groups=defaultdict(list)
    for r in items:groups[r['claim_id']].append(r)
    claimrows=[]
    for cid,g in groups.items():
        if len(g)<2:continue
        labs={r['gold_document_label'] for r in g}
        # Mean log-odds uses model outputs only, not annotation labels.
        odds=[math.log(max(1e-9,r['p_support'])/max(1e-9,r['p_contradict'])) for r in g]
        avg=sum(odds)/len(odds)
        raw=sum(odds)
        # Observed duplicate-document ID correction is identity here because IDs are unique.
        claimrows.append({'claim_id':cid,'n_documents':len(g),'mixed_gold_document_labels':int(len(labs)>1),
                          'mean_log_odds':avg,'sum_log_odds':raw,
                          'mean_rule_pred':'SUPPORT' if avg>=0 else 'CONTRADICT',
                          'sum_rule_pred':'SUPPORT' if raw>=0 else 'CONTRADICT',
                          'has_neutral_dominant_document':int(any(r['p_neutral']>max(r['p_support'],r['p_contradict']) for r in g)),
                          'model_document_disagreement':int(len({r['pred_document_label'] for r in g})>1)})
    savecsv(out/'S21_multidocument_descriptives.csv',claimrows,
            ['claim_id','n_documents','mixed_gold_document_labels','mean_log_odds','sum_log_odds',
             'mean_rule_pred','sum_rule_pred','has_neutral_dominant_document','model_document_disagreement'])
    correct=sum(r['correct'] for r in items)
    byprov={}
    for prov in sorted({r['provenance'] for r in items}):
        subset=[r for r in items if r['provenance']==prov]
        byprov[prov]={'pairs':len(subset),'correct':sum(r['correct'] for r in subset),
                      'accuracy':sum(r['correct'] for r in subset)/len(subset)}
    bymode={}
    for mode in ('annotated_sentences','full_abstract_fallback'):
        subset=[r for r in items if r['evidence_mode']==mode]
        bymode[mode]={'pairs':len(subset),'correct':sum(r['correct'] for r in subset),
                      'accuracy':sum(r['correct'] for r in subset)/len(subset) if subset else None}
    report={'evidence_mode_descriptives':bymode,'status':'PASS_EXPLORATORY_ORACLE_DOCUMENT_INFERENCE_ONLY',
            'model_id2label':model.config.id2label,'model_directory':str(modeldir),
            'input_file_sha256':input_hash,'s20_archive_sha256':h(Path(a.s20_zip).read_bytes()),
            'claim_count':len(claims),'document_pair_count':len(items),'document_accuracy':correct/len(items),
            'provenance_descriptives':byprov,'multi_document_claims':len(claimrows),
            'mixed_gold_document_label_claims':sum(r['mixed_gold_document_labels'] for r in claimrows),
            'observed_duplicate_doc_id_pairs':0,
            'limitations':['Gold sentence indices used where present; otherwise full abstract fallback; neither is retrieval performance',
                           'No claim-level gold label inferred for mixed-document claims',
                           'No provenance reliability weights or measured document dependence',
                           'Mean and sum log-odds have identical sign for fixed document sets',
                           'This is exploratory: no independently validated fusion improvement',
                           'Researcher exposure attestation still requires completion']}
    (out/'S21_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (out/'S21_methodological_decision.md').write_text(
        '# S21 decision\\n\\n'
        'Mixed sentence-annotated and full-abstract fallback inference is diagnostic, not a fusion breakthrough. '
        'Do not report claim-level accuracy on mixed-label claims without independently defined claim-level gold. '
        'No duplicate-ID dependence correction is identifiable here. '
        'Do not claim independent validation until researcher exposure and benchmark independence are resolved.\\n',
        encoding='utf-8')
    resultzip=out.parent/'SciFact_S21_Results.zip'
    with zipfile.ZipFile(resultzip,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file():z.write(p,p.name)
    print(json.dumps({'status':report['status'],'pairs':len(items),'multi_document_claims':len(claimrows),
                      'mixed_label_claims':report['mixed_gold_document_label_claims'],
                      'output':str(resultzip)},indent=2))
if __name__=='__main__':main()
