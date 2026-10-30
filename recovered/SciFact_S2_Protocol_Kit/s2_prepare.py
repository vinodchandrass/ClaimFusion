import argparse, csv, hashlib, json, collections
from pathlib import Path

def read(path):
    with path.open(encoding='utf-8-sig') as f:
        return [json.loads(line) for line in f if line.strip()]

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def norm(s): return ' '.join(s.split()).casefold()
def dump(path,obj): path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def rows_csv(path, rows, fields):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    root,out=Path(a.root),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    names=['claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl','corpus.jsonl']
    for n in names:
        if not (root/n).is_file(): raise SystemExit('MISSING '+str(root/n))
    corpus=read(root/'corpus.jsonl'); docs={str(x['doc_id']):x for x in corpus}
    splits={n:read(root/n) for n in names[:3]}
    files={n:sha(root/n) for n in names}; errors=[]; overlaps=[]; candidates=[]; excluded=[]; fold_summaries=[]
    def make_candidates(split, data):
        for rec in data:
            ev=rec.get('evidence')
            if ev is None: continue # unlabeled test
            if not isinstance(ev,dict):errors.append(f'{split}:{rec.get("id")} malformed evidence');continue
            annotated=set()
            for doc_id,groups in ev.items():
                doc_id=str(doc_id)
                if doc_id not in docs: errors.append(f'{split}:{rec["id"]} missing document {doc_id}');continue
                annotated.add(doc_id)
                if not isinstance(groups,list):errors.append(f'{split}:{rec["id"]} malformed groups {doc_id}');continue
                for group_idx,g in enumerate(groups):
                    if not isinstance(g,dict):errors.append(f'{split}:{rec["id"]} malformed group {doc_id}:{group_idx}');continue
                    label=g.get('label');sentences=g.get('sentences');abstract=docs[doc_id].get('abstract',[])
                    if label not in ('SUPPORT','CONTRADICT'):errors.append(f'{split}:{rec["id"]} unexpected label {label}')
                    if not isinstance(sentences,list) or not all(type(i) is int and 0<=i<len(abstract) for i in sentences):errors.append(f'{split}:{rec["id"]} invalid sentences {doc_id}:{sentences}')
                    candidates.append({'split':split,'claim_id':rec['id'],'claim_text':rec['claim'],'doc_id':doc_id,'annotation_status':'GOLD_EVIDENCE','label':label,'group_index':group_idx,'sentence_indices':json.dumps(sentences),'claim_text_sha256':hashlib.sha256(norm(rec['claim']).encode()).hexdigest()})
            cited=rec.get('cited_doc_ids',[])
            if cited is None:cited=[]
            if not isinstance(cited,list):errors.append(f'{split}:{rec["id"]} invalid cited_doc_ids');continue
            for did in cited:
                did=str(did)
                if did not in docs:errors.append(f'{split}:{rec["id"]} missing cited document {did}');continue
                if did not in annotated:
                    candidates.append({'split':split,'claim_id':rec['id'],'claim_text':rec['claim'],'doc_id':did,'annotation_status':'UNANNOTATED_CITED_PAIR','label':'UNKNOWN','group_index':'','sentence_indices':'','claim_text_sha256':hashlib.sha256(norm(rec['claim']).encode()).hexdigest()})
    make_candidates('train',splits[names[0]]);make_candidates('dev',splits[names[1]])
    def leakage(track,train,eval_rows):
        e={norm(x['claim']) for x in eval_rows}
        clean=[]
        for r in train:
            if norm(r['claim']) in e:
                excluded.append({'track':track,'claim_id':r['id'],'claim_text':r['claim'],'reason':'normalized_exact_text_in_evaluation'})
            else:clean.append(r)
        return clean
    train=splits[names[0]];dev=splits[names[1]];test=splits[names[2]]
    clean_dev=leakage('official_train_to_dev',train,dev)
    clean_test=leakage('official_train_to_unlabeled_test',train,test)
    for i in range(1,6):
        tr=root/'cross_validation'/f'fold_{i}'/f'claims_train_{i}.jsonl'
        dv=root/'cross_validation'/f'fold_{i}'/f'claims_dev_{i}.jsonl'
        if not tr.is_file() or not dv.is_file():errors.append(f'fold_{i} files missing');continue
        files[str(tr.relative_to(root))]=sha(tr);files[str(dv.relative_to(root))]=sha(dv)
        trr,dvr=read(tr),read(dv);clean=leakage(f'fold_{i}',trr,dvr)
        fold_summaries.append({'fold':i,'train_original':len(trr),'train_after_exact_text_exclusion':len(clean),'dev':len(dvr),'excluded':len(trr)-len(clean)})
        rows_csv(out/f'fold_{i}_training_ids.csv',[{'claim_id':r['id']} for r in clean],['claim_id'])
    fields=['split','claim_id','claim_text','doc_id','annotation_status','label','group_index','sentence_indices','claim_text_sha256']
    rows_csv(out/'annotated_and_unknown_cited_pairs.csv',candidates,fields)
    rows_csv(out/'exact_text_exclusions.csv',excluded,['track','claim_id','claim_text','reason'])
    rows_csv(out/'official_dev_training_ids.csv',[{'claim_id':r['id']} for r in clean_dev],['claim_id'])
    rows_csv(out/'unlabeled_test_training_ids.csv',[{'claim_id':r['id']} for r in clean_test],['claim_id'])
    counts=dict(collections.Counter(x['annotation_status'] for x in candidates))
    report={'stage':'S2 protocol preparation ONLY; no model evaluation','source_sha256':files,'counts':{'train':len(train),'dev':len(dev),'test_unlabeled':len(test),'corpus':len(corpus),'candidates_by_status':counts},'official_dev_training_retained':len(clean_dev),'test_training_retained':len(clean_test),'folds':fold_summaries,'errors':errors,'restrictions':['UNKNOWN is NOT a gold NEI label','Cited documents are not exhaustive negative labels','Official test lacks gold labels','Cross-validation folds are alternative partitions, not independent replications','No model training or statistical superiority claim was produced']}
    dump(out/'s2_protocol_manifest.json',report)
    dump(out/'s2_pre_registered_evaluation.json',{'primary_unit':'claim-document pair with gold SUPPORT/CONTRADICT evidence; sentence groups retained','primary_split':'official train/dev with exact-text overlap exclusions on training','secondary':'five-fold CV with per-fold exact-text training exclusions','candidate_scope':'gold evidence groups and cited-but-unannotated UNKNOWN pairs; unknowns excluded from supervised accuracy','metrics':['macro-F1','per-class precision and recall','confusion matrix','claim-group bootstrap 95% confidence intervals'],'comparisons':['majority class','TF-IDF + linear classifier using claim and evidence sentence text','frozen pretrained NLI baseline (model and weights must be declared before use)','proposed fusion method with component ablations'],'frozen_decisions_pending':['source/model selection and exact weights','train-only threshold selection and hyperparameters','gold candidate availability at inference versus retrieval-based evaluation','handling of multiple contradictory evidence groups','claim-level aggregation and document-level evaluation','statistical test and seed budget'],'no_claims':['Do not claim independent test accuracy','Do not treat UNKNOWN as NOT_ENOUGH_INFO','Do not claim novelty or superiority without experiments']})
    print(json.dumps({'status':'PASS' if not errors else 'REVIEW_REQUIRED','candidate_status_counts':counts,'folds':fold_summaries,'errors':len(errors),'output':str(out)},indent=2))
    if errors:raise SystemExit(2)
if __name__=='__main__':main()
