import argparse, collections, csv, hashlib, json, pathlib, sys

def load(p, issues):
    out=[]
    if not p.is_file():
        issues.append(['ERROR','MISSING_FILE',str(p),'']); return out
    with p.open(encoding='utf-8-sig') as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            try:
                obj=json.loads(line)
                if not isinstance(obj,dict): raise ValueError('not object')
                out.append(obj)
            except Exception as e: issues.append(['ERROR','PARSE',str(p),f'line {n}: {e}'])
    return out

def norm(x): return str(x)
def hashtext(x): return hashlib.sha256(x.strip().encode('utf-8')).hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=pathlib.Path); ap.add_argument('output',type=pathlib.Path); a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=True)
    issues=[]; summaries={}; data={}
    for name in ['corpus.jsonl','claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl']:
        data[name]=load(a.root/name,issues)
    corpus={}; doc_duplicates=[]
    for d in data['corpus.jsonl']:
        did=d.get('doc_id'); k=norm(did)
        if did is None: issues.append(['ERROR','CORPUS_NO_ID','corpus','']); continue
        if k in corpus: doc_duplicates.append(k)
        corpus[k]=d
        if not isinstance(d.get('abstract'),list): issues.append(['ERROR','INVALID_ABSTRACT',k,'abstract must be list'])
    if doc_duplicates: issues.append(['ERROR','DUPLICATE_CORPUS_IDS','corpus',str(len(doc_duplicates))])
    splitrows={}; exacttext={}; ids={}; labels=collections.Counter(); evidence_count=0; empty_count=0
    for name in ['claims_train.jsonl','claims_dev.jsonl','claims_test.jsonl']:
        rows=data[name]; splitrows[name]=rows; ids[name]=set(); exacttext[name]=collections.defaultdict(list); seen_ids=set()
        for row in rows:
            cid=row.get('id'); ckey=norm(cid)
            if cid is None: issues.append(['ERROR','MISSING_CLAIM_ID',name,''])
            if ckey in seen_ids: issues.append(['ERROR','DUPLICATE_ID',name,ckey])
            seen_ids.add(ckey); ids[name].add(ckey)
            claim=row.get('claim')
            if not isinstance(claim,str) or not claim.strip(): issues.append(['ERROR','BAD_CLAIM_TEXT',name,ckey]); continue
            exacttext[name][hashtext(claim)].append({'id':cid,'claim':claim})
            cited=row.get('cited_doc_ids')
            if name!='claims_test.jsonl':
                if not isinstance(cited,list): issues.append(['ERROR','BAD_CITED_IDS',name,ckey]); cited=[]
                for did in cited:
                    if norm(did) not in corpus: issues.append(['ERROR','MISSING_CITED_DOCUMENT',name,f'{ckey}:{did}'])
                ev=row.get('evidence')
                if not isinstance(ev,dict): issues.append(['ERROR','INVALID_EVIDENCE',name,ckey]); continue
                if not ev: empty_count+=1
                for did,groups in ev.items():
                    doc=corpus.get(norm(did))
                    if doc is None: issues.append(['ERROR','MISSING_EVIDENCE_DOCUMENT',name,f'{ckey}:{did}'])
                    if not isinstance(groups,list) or not groups: issues.append(['ERROR','INVALID_EVIDENCE_GROUPS',name,f'{ckey}:{did}']); continue
                    for gi,g in enumerate(groups):
                        evidence_count+=1
                        loc=f'{name}:{ckey}:{did}:group{gi}'
                        if not isinstance(g,dict): issues.append(['ERROR','INVALID_EVIDENCE_GROUP',loc,'']); continue
                        label=g.get('label'); labels[str(label)]+=1
                        if label not in ('SUPPORT','CONTRADICT'): issues.append(['ERROR','UNKNOWN_LABEL',loc,str(label)])
                        sentences=g.get('sentences')
                        if not isinstance(sentences,list) or not sentences: issues.append(['ERROR','BAD_SENTENCES',loc,str(sentences)]); continue
                        abstract=doc.get('abstract') if doc else None
                        for s in sentences:
                            if type(s) is not int or s<0: issues.append(['ERROR','INVALID_SENTENCE_INDEX',loc,str(s)])
                            elif isinstance(abstract,list) and s>=len(abstract): issues.append(['ERROR','OUT_OF_RANGE_SENTENCE',loc,f'{s} >= {len(abstract)}'])
            elif 'evidence' in row or 'cited_doc_ids' in row:
                issues.append(['WARNING','TEST_HAS_EXTRA_FIELDS',name,ckey])
        summaries[name]={'rows':len(rows),'unique_ids':len(ids[name]),'duplicate_text_excess':sum(len(v)-1 for v in exacttext[name].values())}
    overlap=[]; names=list(splitrows)
    for i,x in enumerate(names):
        for y in names[i+1:]:
            common=sorted(set(exacttext[x]) & set(exacttext[y]))
            for h in common:
                overlap.append({'split_a':x,'split_b':y,'sha256':h,'records_a':exacttext[x][h],'records_b':exacttext[y][h]})
            shared_ids=ids[x]&ids[y]
            if shared_ids: issues.append(['ERROR','CROSS_SPLIT_ID_OVERLAP',f'{x}|{y}',str(len(shared_ids))])
            if common: issues.append(['WARNING','CROSS_SPLIT_TEXT_OVERLAP',f'{x}|{y}',str(len(common))])
    folds={}
    for i in range(1,6):
        base=a.root/'cross_validation'/f'fold_{i}'
        tr=load(base/f'claims_train_{i}.jsonl',issues); dv=load(base/f'claims_dev_{i}.jsonl',issues)
        t_ids={norm(r.get('id')) for r in tr}; d_ids={norm(r.get('id')) for r in dv}
        t_text={hashtext(r['claim']) for r in tr if isinstance(r.get('claim'),str)}
        d_text={hashtext(r['claim']) for r in dv if isinstance(r.get('claim'),str)}
        folds[str(i)]={'train_rows':len(tr),'dev_rows':len(dv),'shared_ids':len(t_ids&d_ids),'shared_text_hashes':len(t_text&d_text)}
        if t_ids&d_ids: issues.append(['ERROR','FOLD_ID_OVERLAP',str(i),str(len(t_ids&d_ids))])
        if t_text&d_text: issues.append(['WARNING','FOLD_TEXT_OVERLAP',str(i),str(len(t_text&d_text))])
    hashes={str(p.relative_to(a.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.root.rglob('*.jsonl')}
    report={'schema':'SciFact nested evidence: document-key -> list of {sentences,label}', 'splits':summaries,'corpus_documents':len(corpus),'evidence_groups':evidence_count,'empty_evidence_claims':empty_count,'observed_evidence_labels':dict(labels),'cross_split_exact_text_overlap':overlap,'folds':folds,'issues':issues,'file_sha256':hashes,'methodological_caveats':['Empty evidence is not automatically a universal claim-level NEI gold label.','Exact-text screening does not establish absence of semantic or document-level leakage.','Official test file lacks gold evidence annotations; do not score it.','Fold files are alternative partitions, not independent datasets.']}
    (a.output/'scifact_final_integrity_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    with (a.output/'issues.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['severity','type','location','detail']); w.writerows(issues)
    with (a.output/'cross_split_duplicates.json').open('w',encoding='utf-8') as f: json.dump(overlap,f,indent=2,ensure_ascii=False)
    errors=sum(i[0]=='ERROR' for i in issues); warnings=sum(i[0]=='WARNING' for i in issues)
    print(json.dumps({'splits':summaries,'corpus_documents':len(corpus),'evidence_groups':evidence_count,'labels':dict(labels),'errors':errors,'warnings':warnings,'folds':folds,'output':str(a.output)},indent=2))
    print('INTEGRITY_GATE:', 'BLOCKED' if errors else 'STRUCTURAL_PASS_WITH_REVIEW' if warnings else 'STRUCTURAL_PASS')
    return 0 if errors==0 else 2

if __name__=='__main__': sys.exit(main())
