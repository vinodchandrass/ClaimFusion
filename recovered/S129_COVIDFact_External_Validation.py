"""S129: frozen supplied-evidence COVID-Fact evaluation. No training/tuning.
Usage: python S129_COVIDFact_External_Validation.py --root D:\\DataLineage --batch-size 8 --device cpu
"""
import argparse, csv, hashlib, json, math, os, random, time
from pathlib import Path
import numpy as np
import pandas as pd

def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def norm(s): return ' '.join(str(s).casefold().split())

def metrics(y,p):
    from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
    a=precision_recall_fscore_support(y,p,labels=[0,1],zero_division=0)
    return {'n':len(y),'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1_score(y,p,average='macro',labels=[0,1],zero_division=0)), 'supported_recall':float(a[1][1]),'refuted_recall':float(a[1][0]),'confusion_matrix_refuted_supported':confusion_matrix(y,p,labels=[0,1]).tolist()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(r'D:\DataLineage'));ap.add_argument('--batch-size',type=int,default=8);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--limit-records',type=int,default=0,help='Diagnostic only; not publication results');args=ap.parse_args()
    import torch
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA not available')
    root=args.root;src=root/'S127_External_Data'/'covidfact.jsonl';out=root/'S129_COVIDFact_External_Validation';out.mkdir(parents=True,exist_ok=True)
    if not src.is_file():raise FileNotFoundError(src)
    records=[];pairs=[]
    with src.open(encoding='utf-8-sig') as f:
        for lineno,line in enumerate(f,1):
            if not line.strip():continue
            x=json.loads(line);claim=x.get('claim');lab=x.get('label');evidence=x.get('evidence')
            if not isinstance(claim,str) or not claim.strip() or lab not in ('SUPPORTED','REFUTED') or not isinstance(evidence,list) or not evidence or any(not isinstance(e,str) or not e.strip() for e in evidence):raise ValueError(f'Invalid schema at line {lineno}')
            rid=len(records);records.append({'record_id':rid,'claim':claim,'gold':1 if lab=='SUPPORTED' else 0,'normalized_claim':norm(claim),'evidence_count':len(evidence),'source':str(x.get('gold_source','')),'flair':str(x.get('flair',''))})
            for j,ev in enumerate(evidence):pairs.append({'record_id':rid,'evidence_index':j,'evidence':ev,'claim':claim})
            if args.limit_records and len(records)>=args.limit_records:break
    if not records:raise ValueError('No records')
    df=pd.DataFrame(pairs);rd=pd.DataFrame(records)
    fingerprint=hashlib.sha256(('\n'.join(f'{r.record_id}\t{r.evidence_index}\t{r.evidence}\t{r.claim}' for r in df.itertuples(index=False))).encode('utf-8')).hexdigest()
    manifest={'source_sha256':digest(src),'pairs_sha256':fingerprint,'records':len(rd),'pairs':len(df),'max_length':192,'premise':'evidence','hypothesis':'claim','mode':'diagnostic' if args.limit_records else 'full','models':{}}
    print('Preflight:',len(rd),'records;',len(df),'evidence pairs;',rd.normalized_claim.nunique(),'unique normalized claims',flush=True)
    pd.DataFrame({'record_id':rd.record_id,'evidence_count':rd.evidence_count}).to_csv(out/'evidence_counts.csv',index=False)
    for name,relative in [('frozen','S7_NLI_Model'),('weighted','CPU_Domain_Adaptation_Output/checkpoint_weighted')]:
        modeldir=root/relative;weights=modeldir/'model.safetensors';tokfile=modeldir/'tokenizer.json'
        for p in (weights,tokfile,modeldir/'config.json'):
            if not p.is_file():raise FileNotFoundError(p)
        expected={'source_sha256':manifest['source_sha256'],'pairs_sha256':fingerprint,'weights_sha256':digest(weights),'tokenizer_sha256':digest(tokfile),'max_length':192,'premise':'evidence','hypothesis':'claim','model':name}
        final=out/f'{name}_evidence_probabilities.csv';part=out/f'{name}_evidence_probabilities.partial.csv';state=out/f'{name}_state.json'
        done=0
        if state.exists():
            old=json.loads(state.read_text(encoding='utf-8'))
            if any(old.get(k)!=v for k,v in expected.items()):raise ValueError(f'{name} state does not match source/checkpoint/protocol; preserve old output separately')
        elif final.exists() or part.exists():raise ValueError(f'{name} output without provenance state')
        if final.exists():
            prev=pd.read_csv(final);done=len(prev)
            if done!=len(df) or not np.array_equal(prev[['record_id','evidence_index']].to_numpy(),df[['record_id','evidence_index']].to_numpy()):raise ValueError('Completed file misaligned')
            if digest(final)!=old.get('output_sha256'):raise ValueError('Completed file hash mismatch')
            print('Reusing completed',name,flush=True)
        else:
            if part.exists():
                prev=pd.read_csv(part);done=len(prev)
                if done>len(df) or not np.array_equal(prev[['record_id','evidence_index']].to_numpy(),df.iloc[:done][['record_id','evidence_index']].to_numpy()):raise ValueError('Partial file misaligned')
            config=json.loads((modeldir/'config.json').read_text(encoding='utf-8'));labels={int(k):str(v).lower() for k,v in config['id2label'].items()}
            if set(labels.values())!={'contradiction','entailment','neutral'}:raise ValueError(f'Unexpected labels: {labels}')
            tok=AutoTokenizer.from_pretrained(str(modeldir),local_files_only=True)
            model=AutoModelForSequenceClassification.from_pretrained(str(modeldir),local_files_only=True,use_safetensors=True).to(args.device).eval()
            state.write_text(json.dumps({**expected,'status':'running'},indent=2),encoding='utf-8')
            start=time.time()
            for i in range(done,len(df),args.batch_size):
                batch=df.iloc[i:i+args.batch_size]
                tokens=tok(batch.evidence.tolist(),batch.claim.tolist(),truncation=True,padding=True,max_length=192,return_tensors='pt')
                with torch.inference_mode():p=torch.softmax(model(**{k:v.to(args.device) for k,v in tokens.items()}).logits.float(),dim=-1).cpu().numpy()
                block=batch[['record_id','evidence_index']].copy()
                for j,label in labels.items():block['p_'+label]=p[:,j]
                block.to_csv(part,mode='a',header=(i==0),index=False,float_format='%.10g')
                if i==done or (i//args.batch_size)%100==0:print(name,i+len(batch),'/',len(df),flush=True)
            check=pd.read_csv(part)
            if len(check)!=len(df) or not np.array_equal(check[['record_id','evidence_index']].to_numpy(),df[['record_id','evidence_index']].to_numpy()):raise ValueError('Output misalignment')
            probs=check[['p_contradiction','p_entailment','p_neutral']].to_numpy()
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1),1,atol=2e-5):raise ValueError('Invalid probabilities')
            os.replace(part,final);state.write_text(json.dumps({**expected,'status':'complete','output_sha256':digest(final),'elapsed_seconds':round(time.time()-start,2)},indent=2),encoding='utf-8')
            del model
            if args.device=='cuda':torch.cuda.empty_cache()
        manifest['models'][name]={'output_sha256':digest(final),'weights_sha256':expected['weights_sha256']}
    # Fixed policies. Top=first supplied evidence; mean=unweighted average; rank=1/log2(j+2).
    # Evidence list order is supplied by dataset, NOT verified retrieval relevance.
    pred=rd[['record_id','gold','normalized_claim','evidence_count']].copy();summary=[]
    for name in ('frozen','weighted'):
        pf=pd.read_csv(out/f'{name}_evidence_probabilities.csv')
        merged=df[['record_id','evidence_index']].merge(pf,on=['record_id','evidence_index'],validate='one_to_one',sort=False)
        if len(merged)!=len(df):raise ValueError('Evidence mismatch')
        merged['margin']=merged.p_entailment-merged.p_contradiction
        merged['rank_weight']=1/np.log2(merged.evidence_index+2)
        grouped=merged.groupby('record_id',sort=True)
        scores={'first':grouped.margin.first(),'mean':grouped.margin.mean(),'rank_discounted':grouped.apply(lambda g:np.average(g.margin,weights=g.rank_weight),include_groups=False)}
        for policy,series in scores.items():
            col=f'{name}_{policy}';pred[col]=(series.reindex(pred.record_id).to_numpy()>0).astype(int)
            summary.append({'model':name,'policy':policy,**metrics(pred.gold.to_numpy(),pred[col].to_numpy())})
    # Cluster by normalized claim for uncertainty. No post-hoc policy selection.
    rng=np.random.default_rng(20261024);keys=pred.normalized_claim.unique();groups={k:pred.index[pred.normalized_claim==k].to_numpy() for k in keys};bootstrap=[]
    for policy in ('first','mean','rank_discounted'):
        a=pred[f'frozen_{policy}'].to_numpy();b=pred[f'weighted_{policy}'].to_numpy();y=pred.gold.to_numpy()
        delta=[]
        for _ in range(2000):
            sample=rng.choice(keys,size=len(keys),replace=True);idx=np.concatenate([groups[k] for k in sample]);delta.append(metrics(y[idx],b[idx])['macro_f1']-metrics(y[idx],a[idx])['macro_f1'])
        bootstrap.append({'policy':policy,'weighted_minus_frozen_macro_f1':metrics(y,b)['macro_f1']-metrics(y,a)['macro_f1'],'cluster_bootstrap_ci_low':float(np.quantile(delta,.025)),'cluster_bootstrap_ci_high':float(np.quantile(delta,.975))})
    pd.DataFrame(summary).to_csv(out/'evaluation_summary.csv',index=False);pd.DataFrame(bootstrap).to_csv(out/'paired_cluster_bootstrap.csv',index=False);pred.to_csv(out/'record_predictions.csv',index=False)
    unique=pred.drop_duplicates('normalized_claim',keep='first')
    pd.DataFrame([{'model':name,'policy':pol,**metrics(unique.gold.to_numpy(),unique[f'{name}_{pol}'].to_numpy())} for name in ('frozen','weighted') for pol in ('first','mean','rank_discounted')]).to_csv(out/'unique_claim_sensitivity.csv',index=False)
    manifest['files']={p.name:digest(p) for p in out.glob('*.csv')};(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('S129 COMPLETE; outputs:',out,flush=True)
    print(pd.DataFrame(summary)[['model','policy','n','accuracy','macro_f1']].to_string(index=False),flush=True)
    print('CAUTION: COVID-Fact truth labels are not guaranteed to be evidence-level NLI labels; first evidence is not a verified relevance ranking.',flush=True)

if __name__=='__main__':main()
