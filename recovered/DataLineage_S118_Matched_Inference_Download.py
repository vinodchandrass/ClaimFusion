"""S118: matched six-condition NLI inference; no training, no label-based evaluation.
Run: python DataLineage_S118_Matched_Inference.py --root D:\\DataLineage
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODELS = {
    'frozen': 'S7_NLI_Model',
    'weighted': 'CPU_Domain_Adaptation_Output/checkpoint_weighted',
    'unweighted': 'CPU_Domain_Adaptation_Output/checkpoint_unweighted',
}
LENGTHS = (192, 512)
KEYS = ['claim_id','doc_id','doc_rank','sentence_index']
LABELS = ('contradiction','entailment','neutral')

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',type=Path,default=Path(r'D:\DataLineage'))
    ap.add_argument('--input',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--batch-size',type=int,default=8)
    ap.add_argument('--device',default='cpu',choices=['cpu','cuda'])
    args=ap.parse_args()
    root=args.root
    source=args.input or root/'CrossDomain_Fusion_Output'/'adapted_scifact_candidates_DEV_ONLY.csv'
    out=args.output or root/'S118_Matched_Inference_Output'
    out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(source,low_memory=False)
    required=KEYS+['claim','sentence']
    missing=[x for x in required if x not in df]
    if missing:raise ValueError(f'Missing required columns: {missing}')
    if len(df)!=4500 or df.claim_id.nunique()!=300:raise ValueError('Unexpected input dimensions: inspect before proceeding')
    if df[KEYS].isna().any().any() or df.duplicated(KEYS).any():raise ValueError('Missing/duplicate composite keys')
    if df[['claim','sentence']].isna().any().any():raise ValueError('Missing text')
    for _,g in df.groupby('claim_id',sort=False):
        if len(g)!=15 or g.doc_id.nunique()!=5:raise ValueError('Expected 5 docs and 15 sentences per claim')
    input_sha=digest(source)
    metadata={'input_path':str(source),'input_sha256':input_sha,'rows':len(df),'claims':int(df.claim_id.nunique()),'conditions':{},'device':args.device,'batch_size':args.batch_size,'torch':torch.__version__}
    device=torch.device(args.device)
    if args.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
    for model_name,relpath in MODELS.items():
        directory=root/relpath
        if not directory.is_dir():raise FileNotFoundError(directory)
        config=json.loads((directory/'config.json').read_text(encoding='utf-8'))
        label_map={int(k):str(v).lower() for k,v in config['id2label'].items()}
        if set(label_map.values())!=set(LABELS):raise ValueError(f'Unexpected label mapping for {model_name}: {label_map}')
        weight=directory/'model.safetensors'
        if not weight.exists():raise FileNotFoundError(weight)
        weight_sha=digest(weight)
        tokenizer=AutoTokenizer.from_pretrained(str(directory),local_files_only=True)
        model=AutoModelForSequenceClassification.from_pretrained(str(directory),local_files_only=True,use_safetensors=True).to(device).eval()
        for maxlen in LENGTHS:
            name=f'{model_name}_L{maxlen}'
            final=out/f'{name}.csv'
            progress=out/f'{name}.partial.csv'
            statefile=out/f'{name}.state.json'
            expected={'input_sha256':input_sha,'weight_sha256':weight_sha,'max_length':maxlen,'model':model_name,'tokenizer_sha256':digest(directory/'tokenizer.json')}
            if final.exists():
                saved=pd.read_csv(final)
                if len(saved)!=len(df) or not saved[KEYS].equals(df[KEYS]):raise ValueError(f'Existing final output mismatch: {final}')
                old=json.loads(statefile.read_text()) if statefile.exists() else {}
                if any(old.get(k)!=v for k,v in expected.items()):raise ValueError(f'Existing output provenance mismatch: {final}')
                print('SKIP completed',name,flush=True)
                metadata['conditions'][name]={'status':'complete','output':str(final),'sha256':digest(final)}
                continue
            done=0
            if progress.exists():
                if not statefile.exists():raise ValueError(f'Missing resume state for {progress}')
                old=json.loads(statefile.read_text())
                if any(old.get(k)!=v for k,v in expected.items()):raise ValueError('Resume metadata mismatch')
                prev=pd.read_csv(progress)
                done=len(prev)
                if done>len(df) or not prev[KEYS].equals(df.iloc[:done][KEYS].reset_index(drop=True)):
                    raise ValueError('Resume keys mismatch')
            statefile.write_text(json.dumps({**expected,'status':'running'},indent=2))
            print(f'RUN {name}: {done}/{len(df)}',flush=True)
            start=time.time()
            for i in range(done,len(df),args.batch_size):
                part=df.iloc[i:i+args.batch_size]
                tokens=tokenizer(part.claim.astype(str).tolist(),part.sentence.astype(str).tolist(),
                    truncation=True,padding=True,max_length=maxlen,return_tensors='pt')
                tokens={k:v.to(device) for k,v in tokens.items()}
                with torch.inference_mode():
                    logits=model(**tokens).logits
                    probs=torch.softmax(logits.float(),dim=-1).cpu().numpy()
                record=part[KEYS].copy()
                for idx,label in label_map.items():record[f'p_{label}']=probs[:,idx]
                record.to_csv(progress,mode='a',header=(i==0),index=False,float_format='%.10g')
                if (i//args.batch_size)%25==0:print(name,min(i+len(part),len(df)),'/',len(df),flush=True)
            completed=pd.read_csv(progress)
            if len(completed)!=len(df) or not completed[KEYS].equals(df[KEYS]):raise ValueError('Final output alignment failed')
            p=completed[[f'p_{label}' for label in LABELS]].to_numpy(dtype=float)
            if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=2e-5):raise ValueError('Invalid probabilities')
            os.replace(progress,final)
            statefile.write_text(json.dumps({**expected,'status':'complete','output_sha256':digest(final),'elapsed_seconds':round(time.time()-start,2)},indent=2))
            metadata['conditions'][name]={'status':'complete','output':str(final),'sha256':digest(final)}
            (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
            print('DONE',name,flush=True)
        del model
        if args.device=='cuda':torch.cuda.empty_cache()
    (out/'manifest.json').write_text(json.dumps(metadata,indent=2))
    print('ALL SIX CONDITIONS COMPLETE; no claim-level evaluation performed.')

if __name__=='__main__':main()
