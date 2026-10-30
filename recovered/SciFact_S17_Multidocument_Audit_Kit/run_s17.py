import argparse,csv,hashlib,json,zipfile
from collections import Counter,defaultdict
from pathlib import Path

PINS={'claims_train.jsonl':'f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0',
'claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217',
'corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'}
def read(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def norm(x): return ' '.join(x.casefold().split())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def csvwrite(path,rows,fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True)
    p.add_argument('--s14-zip',required=True)
    p.add_argument('--out',required=True)
    a=p.parse_args()
    data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    for fn,pin in PINS.items():
        if not (data/fn).is_file() or sha(data/fn)!=pin:
            raise RuntimeError('STOP: SciFact file absent or hash mismatch: '+fn)
    corpus={str(r['doc_id']):r for r in read(data/'corpus.jsonl')}
    train=read(data/'claims_train.jsonl');dev=read(data/'claims_dev.jsonl')
    with zipfile.ZipFile(a.s14_zip) as z:
        report=json.loads(z.read('S14_report.json'))
        if report['status']!='PASS_INFERENCE_AND_S7_PARITY' or report['dataset_sha256']!=PINS:
            raise RuntimeError('STOP: S14 provenance mismatch')
        predictions=list(csv.DictReader(z.read('S14_pair_predictions.csv').decode('utf-8-sig').splitlines()))
    pred={(r['split'],r['claim_id'],r['doc_id']):r for r in predictions}
    if len(pred)!=len(predictions):raise RuntimeError('Duplicate S14 prediction keys')
    dev_text={norm(x['claim']) for x in dev}
    rows=[];anomalies=[];summaries={}
    for split,claims in [('train',train),('dev_evaluation',dev)]:
        for c in claims:
            cid=str(c['id']);e=c.get('evidence',{})
            docs=[];labels=[];valid=0
            for did,groups in e.items():
                did=str(did)
                if did not in corpus:
                    anomalies.append({'split':split,'claim_id':cid,'doc_id':did,'reason':'document_not_in_corpus'});continue
                labs={g.get('label') for g in groups}
                idx={i for g in groups for i in g.get('sentences',[])}
                abstract=corpus[did].get('abstract',[])
                if len(labs)!=1 or not labs.issubset({'SUPPORT','CONTRADICT'}) or not idx or any(type(i)!=int or i<0 or i>=len(abstract) for i in idx):
                    anomalies.append({'split':split,'claim_id':cid,'doc_id':did,'reason':'invalid_or_mixed_annotation'});continue
                valid+=1;docs.append(did);labels.append(next(iter(labs)))
            kinds=set(labels)
            multi=valid>=2
            conflict=multi and len(kinds)>1
            predicted=sum((split,cid,d) in pred for d in docs)
            rows.append({'split':split,'claim_id':cid,'annotated_documents':len(e),
                         'valid_annotated_documents':valid,'multi_document':int(multi),
                         'opposing_labels_across_documents':int(conflict),
                         'support_documents':labels.count('SUPPORT'),
                         'contradict_documents':labels.count('CONTRADICT'),
                         's14_predicted_documents':predicted,
                         'exact_text_overlaps_dev':int(split=='train' and norm(c['claim']) in dev_text)})
        rr=[r for r in rows if r['split']==split]
        summaries[split]={'claims':len(rr),'multi_document_claims':sum(r['multi_document'] for r in rr),
                          'opposing_label_claims':sum(r['opposing_labels_across_documents'] for r in rr),
                          'annotated_document_pairs':sum(r['valid_annotated_documents'] for r in rr),
                          'claims_with_no_annotated_documents':sum(r['valid_annotated_documents']==0 for r in rr)}
    csvwrite(out/'S17_claim_document_inventory.csv',rows,list(rows[0]))
    csvwrite(out/'S17_annotation_anomalies.csv',anomalies,['split','claim_id','doc_id','reason'])
    devpairs={(r['claim_id'],r['doc_id']) for r in predictions if r['split']=='dev_evaluation'}
    actual={(str(c['id']),str(d)) for c in dev for d in c.get('evidence',{})}
    mismatch=sorted(actual^devpairs)
    # Multi-document verification is only possible if sufficiently many actual claims have
    # multiple labeled documents. No synthetic labels or unannotated NEI pairs are introduced.
    n=summaries['dev_evaluation']['multi_document_claims']
    status='DESCRIPTIVE_AUDIT_COMPLETE_MULTI_DOC_FEASIBILITY_UNDETERMINED'
    if mismatch or anomalies:status='STOP_RECONCILIATION_REQUIRED'
    elif n<30:status='PASS_INTEGRITY_INSUFFICIENT_DEV_MULTI_DOC_SUPPORT'
    else:status='PASS_INTEGRITY_MULTI_DOC_EXPLORATORY_ONLY'
    result={'status':status,'dataset_sha256':PINS,'s14_model_commit':report['model_commit'],
            'summary':summaries,'annotation_anomalies':len(anomalies),
            'dev_s14_pair_key_symmetric_difference':len(mismatch),
            'dev_s14_pair_key_difference_examples':mismatch[:20],
            'limitations':['Only annotated evidence document pairs are included',
                           'No verified source reliability or document dependence labels',
                           'Unlabeled SciFact test data cannot support supervised confirmatory metrics',
                           'Official development set has already been used for model exploration',
                           'Multi-document evaluation requires sufficient opposing and concordant labeled claims']}
    (out/'S17_feasibility_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/'S17_protocol_next_steps.md').write_text(
        '# S17 evidence feasibility decision\n\n'
        'This audit makes no performance or fusion superiority claim.\n\n'
        'If multi-document support is limited, do not fabricate cross-document gold labels. '
        'Use an externally sourced benchmark with verified claim/document provenance and independent labels, '
        'or redesign the paper around bounded selective risk and failure analysis. '
        'If support is adequate, preregister the dependence-aware operator and train-only tuning, '
        'then reserve independent evaluation data.\n',encoding='utf-8')
    with zipfile.ZipFile(out.parent/'SciFact_S17_Results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.is_file():z.write(f,f.name)
    print(json.dumps({'status':status,'summary':summaries,'anomalies':len(anomalies),'dev_pair_key_difference':len(mismatch)},indent=2))
if __name__=='__main__':main()
