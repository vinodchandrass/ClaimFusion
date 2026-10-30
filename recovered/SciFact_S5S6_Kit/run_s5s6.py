import argparse, csv, hashlib, json, random, zipfile
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_recall_fscore_support

PINS={'claims_train.jsonl':'f4c8fa82d8bd0653a9cc8d61a6ea48c25eacea64e90af5dbf390ebb1b74372f0','claims_dev.jsonl':'86f0435d08fdb65d1aa41d1472684f57e6e71930626497bdf4d7a9ec1a632217','corpus.jsonl':'b8d6c89624cb2ed74dee8938effc4f5d8bd2086887880af8110d64be4ceade62'}
LABELS=['CONTRADICT','SUPPORT']
def read(p):
    with p.open(encoding='utf-8-sig') as f: return [json.loads(s) for s in f if s.strip()]
def norm(s): return ' '.join(s.casefold().split())
def prepare(rows,corpus,excluded=set()):
    out=[]; ambiguous=[]; absent=[]
    for row in rows:
        if str(row['id']) in excluded: continue
        for docid,groups in row.get('evidence',{}).items():
            doc=corpus.get(str(docid))
            if doc is None: raise RuntimeError(f'Missing document {docid}')
            labels={g.get('label') for g in groups}
            if not labels <= set(LABELS): raise RuntimeError(f'Unexpected labels: {labels}')
            if len(labels)!=1:
                ambiguous.append({'claim_id':row['id'],'doc_id':docid,'labels':sorted(labels)}); continue
            abstract=doc.get('abstract',[])
            ix=sorted({i for g in groups for i in g.get('sentences',[])})
            if not ix or any(not isinstance(i,int) or i<0 or i>=len(abstract) for i in ix):
                absent.append({'claim_id':row['id'],'doc_id':docid,'indices':ix}); continue
            ev=' '.join(abstract[i] for i in ix)
            out.append({'claim_id':str(row['id']),'doc_id':str(docid),'claim':row['claim'],'evidence':ev,'label':next(iter(labels)),'text_hash':hashlib.sha256(norm(row['claim']).encode()).hexdigest()})
    return out,ambiguous,absent

def scores(y,p):
    pr,rc,f1,_=precision_recall_fscore_support(y,p,labels=LABELS,zero_division=0)
    return {'n':len(y),'accuracy':float(accuracy_score(y,p)),'macro_f1':float(f1_score(y,p,labels=LABELS,average='macro',zero_division=0)),'labels':LABELS,'confusion_matrix':confusion_matrix(y,p,labels=LABELS).tolist(),'per_class':{lab:{'precision':float(pr[i]),'recall':float(rc[i]),'f1':float(f1[i])} for i,lab in enumerate(LABELS)}}
def feature(rows,mode):
    if mode=='claim':return [x['claim'] for x in rows]
    if mode=='evidence':return [x['evidence'] for x in rows]
    return [x['claim']+' [EVIDENCE] '+x['evidence'] for x in rows]
def model(mode,weighted,c):
    return Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=100000,sublinear_tf=True)),('lr',LogisticRegression(C=c,class_weight='balanced' if weighted else None,max_iter=2000,random_state=2026))])
def tune(train,mode,weighted):
    y=np.array([x['label'] for x in train]); groups=np.array([x['claim_id'] for x in train]); X=feature(train,mode)
    if len(set(groups))<3 or min(Counter(y).values())<3:raise RuntimeError('Not enough grouped samples')
    splitter=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026)
    folds=list(splitter.split(X,y,groups))
    best=None; trials=[]
    for c in [0.1,1.0,10.0]:
        # Select threshold only on held-out training-fold predictions.
        probs=np.full(len(train),np.nan)
        for tr,va in folds:
            if len(set(y[tr]))!=2:raise RuntimeError('Training fold missing a class')
            m=model(mode,weighted,c);m.fit([X[i] for i in tr],y[tr]);ci=list(m.named_steps['lr'].classes_).index('CONTRADICT')
            probs[va]=m.predict_proba([X[i] for i in va])[:,ci]
        if np.isnan(probs).any():raise RuntimeError('Missing out-of-fold probabilities')
        for threshold in [0.3,0.4,0.5,0.6,0.7]:
            pred=np.where(probs>=threshold,'CONTRADICT','SUPPORT')
            value=float(f1_score(y,pred,labels=LABELS,average='macro',zero_division=0))
            trials.append({'C':c,'threshold':threshold,'train_oof_macro_f1':value})
            candidate=(value,-abs(threshold-0.5),-c,c,threshold)
            if best is None or candidate>best:best=candidate
    return {'C':best[3],'threshold':best[4],'train_oof_macro_f1':best[0]},trials

def bootstrap(rows,y,p,B,seed=2026):
    rng=np.random.default_rng(seed); ix=defaultdict(list)
    for i,r in enumerate(rows):ix[r['claim_id']].append(i)
    ids=list(ix); vals=[]
    for _ in range(B):
        chosen=rng.integers(0,len(ids),size=len(ids)); ind=[j for k in chosen for j in ix[ids[k]]]
        vals.append(float(f1_score(np.array(y)[ind],np.array(p)[ind],labels=LABELS,average='macro',zero_division=0)))
    return [float(v) for v in np.percentile(vals,[2.5,97.5])]

def main():
    a=argparse.ArgumentParser();a.add_argument('--root',required=True);a.add_argument('--out',required=True);a.add_argument('--bootstrap',type=int,default=2000);args=a.parse_args()
    root=Path(args.root);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    for f,expected in PINS.items():
        actual=hashlib.sha256((root/f).read_bytes()).hexdigest()
        if actual!=expected:raise RuntimeError(f'PIN MISMATCH: {f}: {actual}')
    corpus={str(d['doc_id']):d for d in read(root/'corpus.jsonl')}
    original_train=read(root/'claims_train.jsonl'); original_dev=read(root/'claims_dev.jsonl')
    def eval_split(name,train_rows,dev_rows):
        devtexts={norm(x['claim']) for x in dev_rows}; excluded={str(x['id']) for x in train_rows if norm(x['claim']) in devtexts}
        train,ambtr,badtr=prepare(train_rows,corpus,excluded);dev,ambdev,baddev=prepare(dev_rows,corpus)
        if badtr or baddev:raise RuntimeError(f'{name}: invalid evidence sentence indices: {badtr[:2]} {baddev[:2]}')
        if not train or not dev:raise RuntimeError(f'{name}: empty train or evaluation')
        if len(set(x['claim_id'] for x in train)&set(x['claim_id'] for x in dev)):raise RuntimeError('Claim-ID overlap')
        if set(x['text_hash'] for x in train)&set(x['text_hash'] for x in dev):raise RuntimeError('Claim-text overlap')
        y=[x['label'] for x in dev];output={'train_pairs':len(train),'dev_pairs':len(dev),'train_claims':len(set(x['claim_id'] for x in train)),'dev_claims':len(set(x['claim_id'] for x in dev)),'excluded_train_claim_ids':sorted(excluded),'mixed_label_pairs_excluded':{'train':ambtr,'dev':ambdev},'models':{}}
        predrows=[]
        for mode in ['claim','evidence','pair']:
            for weighted in [False,True]:
                name_model=f'{mode}_'+('balanced' if weighted else 'unweighted')
                config,trials=tune(train,mode,weighted);m=model(mode,weighted,config['C']);m.fit(feature(train,mode),[x['label'] for x in train]);ci=list(m.named_steps['lr'].classes_).index('CONTRADICT')
                probs=m.predict_proba(feature(dev,mode))[:,ci];pred=np.where(probs>=config['threshold'],'CONTRADICT','SUPPORT').tolist()
                result=scores(y,pred);result['bootstrap_macro_f1_95ci']=bootstrap(dev,y,pred,args.bootstrap);result['selected_training_only']=config;result['training_cv_grid']=trials
                output['models'][name_model]=result
                for row,pr,pp in zip(dev,pred,probs):predrows.append({'split':name,'model':name_model,'claim_id':row['claim_id'],'doc_id':row['doc_id'],'gold':row['label'],'predicted':pr,'p_contradict':float(pp)})
        return output,predrows
    allresults={};allpred=[]
    result,preds=eval_split('official_dev',original_train,original_dev);allresults['official_dev']=result;allpred+=preds
    for k in range(1,6):
        tr=read(root/f'cross_validation/fold_{k}/claims_train_{k}.jsonl');dv=read(root/f'cross_validation/fold_{k}/claims_dev_{k}.jsonl')
        result,preds=eval_split(f'fold_{k}',tr,dv);allresults[f'fold_{k}']=result;allpred+=preds
    (out/'S5S6_results.json').write_text(json.dumps({'protocol':'Gold-evidence binary claim-document classification; no retrieval, no NLI, no novel fusion','bootstrap_resamples':args.bootstrap,'results':allresults},indent=2),encoding='utf-8')
    with (out/'S5S6_predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['split','model','claim_id','doc_id','gold','predicted','p_contradict']);w.writeheader();w.writerows(allpred)
    print(json.dumps({'status':'PASS','splits':{k:{'train_pairs':v['train_pairs'],'dev_pairs':v['dev_pairs'],'excluded_claims':len(v['excluded_train_claim_ids']),'best_macro_f1':max(m['macro_f1'] for m in v['models'].values())} for k,v in allresults.items()},'output':str(out)},indent=2))
if __name__=='__main__':main()
