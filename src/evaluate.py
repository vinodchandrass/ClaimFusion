#!/usr/bin/env python3
"""Recompute binary claim-level metrics from supplied matched predictions; no model inference."""
import argparse,csv,json,random
LABELS=("SUPPORTS","REFUTES")
def metrics(gold,pred):
    if not gold: raise ValueError("No observations")
    cm={g:{p:0 for p in LABELS} for g in LABELS}
    for g,p in zip(gold,pred): cm[g][p]+=1
    fs=[];rec={}
    for c in LABELS:
        tp=cm[c][c];fp=sum(cm[g][c] for g in LABELS if g!=c);fn=sum(cm[c][p] for p in LABELS if p!=c)
        fs.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
        rec[c]=tp/(tp+fn) if tp+fn else None
    return {"n":len(gold),"accuracy":sum(g==p for g,p in zip(gold,pred))/len(gold),"macro_f1":sum(fs)/2,"recall":rec,"confusion":cm}
def quantile(xs,q):
    xs=sorted(xs);t=(len(xs)-1)*q;i=int(t);j=min(i+1,len(xs)-1)
    return xs[i]+(xs[j]-xs[i])*(t-i)
def evaluate(rows,reps,seed):
    if not rows: raise ValueError("No rows")
    if len(set(r["claim_id"] for r in rows))!=len(rows): raise ValueError("Duplicate claim_id")
    for r in rows:
        if any(r[k] not in LABELS for k in ("gold","frozen","adapted")):raise ValueError("Invalid label")
    g=[r["gold"] for r in rows];f=[r["frozen"] for r in rows];a=[r["adapted"] for r in rows]
    fm=metrics(g,f);am=metrics(g,a);rng=random.Random(seed);n=len(rows);ds=[]
    for _ in range(reps):
        ix=[rng.randrange(n) for j in range(n)]
        ds.append(metrics([g[i] for i in ix],[a[i] for i in ix])["macro_f1"]-metrics([g[i] for i in ix],[f[i] for i in ix])["macro_f1"])
    return {"status":"RECOMPUTED_FROM_SUPPLIED_ROWS","frozen":fm,"adapted":am,"macro_f1_difference":am["macro_f1"]-fm["macro_f1"],"bootstrap":{"unit":"claim","method":"paired percentile","resamples":reps,"seed":seed,"ci95":[quantile(ds,.025),quantile(ds,.975)]}}
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--predictions",required=True);p.add_argument("--output",required=True);p.add_argument("--resamples",type=int,default=2000);p.add_argument("--seed",type=int,default=2026);args=p.parse_args()
    if args.resamples<100:raise ValueError("Need at least 100 resamples")
    with open(args.predictions,newline="",encoding="utf-8-sig") as f:
        reader=csv.DictReader(f);required={"claim_id","gold","frozen","adapted"}
        if not required.issubset(reader.fieldnames or []):raise ValueError("Missing required columns")
        rows=list(reader)
    with open(args.output,"w",encoding="utf-8") as f:json.dump(evaluate(rows,args.resamples,args.seed),f,indent=2)
