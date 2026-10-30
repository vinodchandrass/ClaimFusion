import argparse, csv, hashlib, json, math, random, time
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ["Refutes", "Supports", "Neutral"]
MAP = {"refutes":0,"refute":0,"contradiction":0,"contradicts":0,
       "supports":1,"support":1,"entailment":1,"entails":1,
       "neutral":2,"nei":2,"not enough info":2}
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):h.update(block)
    return h.hexdigest()
def normalize(s): return " ".join(str(s).lower().split())
def load(path):
    with open(path,encoding="utf-8-sig",newline="") as f: rows=list(csv.DictReader(f))
    if not rows:raise ValueError("Empty CSV: "+str(path))
    for i,r in enumerate(rows):
        for k in ("claim","evidence","label"):
            if not r.get(k):raise ValueError(f"Missing {k} in row {i}")
        v=MAP.get(normalize(r["label"]))
        if v is None:raise ValueError(f"Unexpected label {r['label']!r}")
        r["_y"]=v
    return rows
def split_groups(rows, fraction, seed):
    groups=defaultdict(list)
    for r in rows:groups[normalize(r["claim"])].append(r)
    keys=list(groups)
    random.Random(seed).shuffle(keys)
    n=max(1,round(len(keys)*fraction))
    val_keys=set(keys[:n])
    tr=[r for k in keys if k not in val_keys for r in groups[k]]
    va=[r for k in keys if k in val_keys for r in groups[k]]
    if not tr or not va:raise ValueError("Empty split")
    assert not ({normalize(r["claim"]) for r in tr}&{normalize(r["claim"]) for r in va})
    return tr,va
def metric(y,p):
    cm=[[0]*3 for _ in range(3)]
    for a,b in zip(y,p):cm[a][b]+=1
    per={}
    for k,name in enumerate(LABELS):
        tp=cm[k][k]; support=sum(cm[k]); predicted=sum(row[k] for row in cm)
        precision=tp/predicted if predicted else 0
        recall=tp/support if support else 0
        f1=2*precision*recall/(precision+recall) if precision+recall else 0
        per[name]={"support":support,"precision":precision,"recall":recall,"f1":f1}
    return {"n":len(y),"accuracy":sum(cm[k][k] for k in range(3))/len(y),
            "macro_f1":sum(x["f1"] for x in per.values())/3,
            "per_class":per,"confusion_matrix":cm}
def ensure_model_labels(config):
    mapping={int(k):str(v).lower() for k,v in config.id2label.items()}
    if len(mapping)!=3:raise ValueError(f"Expected three labels, got {mapping}")
    expected=("contradiction","entailment","neutral")
    if any(expected[i] not in mapping[i] for i in range(3)):
        raise ValueError(f"Frozen model label order differs: {mapping}; stop rather than guess")
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--train",required=True)
    ap.add_argument("--dev",required=True)
    ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--epochs",type=int,default=1)
    ap.add_argument("--batch-size",type=int,default=4)
    ap.add_argument("--grad-accum",type=int,default=4)
    ap.add_argument("--threads",type=int,default=4)
    ap.add_argument("--max-length",type=int,default=192)
    ap.add_argument("--learning-rate",type=float,default=1.5e-5)
    ap.add_argument("--val-fraction",type=float,default=0.2)
    ap.add_argument("--seed",type=int,default=20261021)
    ap.add_argument("--variants",choices=["both","weighted","unweighted"],default="both")
    ap.add_argument("--dev-only-after-training",action="store_true",help="Required acknowledgment: development evaluation is not independent")
    args=ap.parse_args()
    if not args.dev_only_after_training:
        ap.error("Include --dev-only-after-training to acknowledge HEALTHVER dev is exploratory, not independent holdout")
    if args.epochs<1 or args.batch_size<1 or args.grad_accum<1:ap.error("epochs, batch-size, grad-accum must be positive")
    if not 0<args.val_fraction<0.5:ap.error("val-fraction must be between 0 and 0.5")
    train_path=Path(args.train).resolve(); dev_path=Path(args.dev).resolve()
    if train_path==dev_path:raise ValueError("Train and dev must differ")
    if "test" in train_path.name.lower() or "test" in dev_path.name.lower():
        raise ValueError("Test-like filenames prohibited in this development-only runner")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    train=load(train_path);dev=load(dev_path)
    internal_train,internal_val=split_groups(train,args.val_fraction,args.seed)
    train_claims={normalize(r["claim"]) for r in train}
    dev_claims={normalize(r["claim"]) for r in dev}
    overlap=train_claims&dev_claims
    # Exclude train claims that occur in dev from internal fitting, to prevent exact-claim leakage.
    internal_train=[r for r in internal_train if normalize(r["claim"]) not in overlap]
    internal_val=[r for r in internal_val if normalize(r["claim"]) not in overlap]
    if not internal_train or not internal_val:raise ValueError("No data after claim-overlap exclusion")
    import torch
    from torch.utils.data import DataLoader,Dataset
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch.set_num_threads(args.threads)
    random.seed(args.seed);torch.manual_seed(args.seed)
    model_path=Path(args.model)
    tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True)
    class PairDataset(Dataset):
        def __init__(self,rows):
            self.rows=rows
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return self.rows[i]
    def collate(batch):
        enc=tokenizer([r["evidence"] for r in batch],[r["claim"] for r in batch],
                      truncation=True,padding=True,max_length=args.max_length,return_tensors="pt")
        enc["labels"]=torch.tensor([r["_y"] for r in batch],dtype=torch.long)
        return enc
    def evaluate(model,rows,return_probs=False):
        loader=DataLoader(PairDataset(rows),batch_size=args.batch_size,shuffle=False,collate_fn=collate)
        preds=[];gold=[];probs=[]
        model.eval()
        with torch.no_grad():
            for b in loader:
                y=b.pop("labels"); logits=model(**b).logits
                prob=torch.softmax(logits,dim=-1)
                preds+=prob.argmax(-1).tolist();gold+=y.tolist()
                if return_probs:probs+=prob.tolist()
        result=metric(gold,preds)
        return (result,preds,probs) if return_probs else result
    variants=["unweighted","weighted"] if args.variants=="both" else [args.variants]
    report={"status":"DEVELOPMENT_ONLY_NO_INDEPENDENT_HOLDOUT","method":"CPU three-class NLI domain adaptation",
            "train_sha256":sha(train_path),"dev_sha256":sha(dev_path),
            "model_config_sha256":sha(model_path/"config.json"),
            "settings":vars(args),"train_pairs":len(train),"dev_pairs":len(dev),
            "internal_train_pairs":len(internal_train),"internal_val_pairs":len(internal_val),
            "internal_train_claims":len({normalize(r["claim"]) for r in internal_train}),
            "internal_val_claims":len({normalize(r["claim"]) for r in internal_val}),
            "exact_train_dev_claim_overlap":len(overlap),
            "dev_only":True,"sealed_test_accessed":False,"results":{},
            "limitations":["HealthVer dev is development data, not independent validation",
                           "Evidence reuse across train/dev is not eliminated by claim-grouped splitting",
                           "Only one internal validation partition; no cross-validation",
                           "This tests NLI evidence pairs, not document-level fusion"]}
    counts=Counter(r["_y"] for r in internal_train)
    weights=torch.tensor([len(internal_train)/(3*max(1,counts[i])) for i in range(3)],dtype=torch.float)
    for variant in variants:
        print(f"Training {variant} on CPU, {len(internal_train)} pairs",flush=True)
        torch.manual_seed(args.seed)
        model=AutoModelForSequenceClassification.from_pretrained(str(model_path),local_files_only=True)
        ensure_model_labels(model.config)
        model.to("cpu")
        optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate)
        criterion=torch.nn.CrossEntropyLoss(weight=weights if variant=="weighted" else None)
        loader=DataLoader(PairDataset(internal_train),batch_size=args.batch_size,shuffle=True,collate_fn=collate)
        best=-1;history=[]; best_dir=out/("checkpoint_"+variant)
        t0=time.time()
        for epoch in range(args.epochs):
            model.train();optimizer.zero_grad();loss_total=0.0
            for step,b in enumerate(loader):
                y=b.pop("labels")
                logits=model(**b).logits
                loss=criterion(logits,y)
                loss_total+=loss.item()
                (loss/args.grad_accum).backward()
                if (step+1)%args.grad_accum==0 or step+1==len(loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                    optimizer.step();optimizer.zero_grad()
                if (step+1)%200==0:print(f"  epoch {epoch+1}, batch {step+1}/{len(loader)}",flush=True)
            val=evaluate(model,internal_val)
            history.append({"epoch":epoch+1,"train_mean_batch_loss":loss_total/len(loader),"internal_val":val})
            print(f"  epoch {epoch+1} val macro-F1 {val['macro_f1']:.4f}",flush=True)
            if val["macro_f1"]>best:
                best=val["macro_f1"]
                model.save_pretrained(best_dir,safe_serialization=True)
                tokenizer.save_pretrained(best_dir)
        del model
        best_model=AutoModelForSequenceClassification.from_pretrained(str(best_dir),local_files_only=True)
        dev_result,pred,probs=evaluate(best_model,dev,return_probs=True)
        with (out/f"dev_predictions_{variant}_DEV_ONLY.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.writer(f);w.writerow(["row_index","claim_group_sha256","gold_label","predicted_label","p_refutes","p_supports","p_neutral"])
            for i,(r,p,pr) in enumerate(zip(dev,pred,probs)):
                group_hash=hashlib.sha256(normalize(r["claim"]).encode()).hexdigest()
                w.writerow([i,group_hash,LABELS[r["_y"]],LABELS[p],*[f"{v:.9f}" for v in pr]])
        report["results"][variant]={"history":history,"best_internal_val_macro_f1":best,
                                   "dev":dev_result,"elapsed_seconds":round(time.time()-t0,2),
                                   "checkpoint":str(best_dir)}
        with (out/"domain_adaptation_report_DEV_ONLY.json").open("w",encoding="utf-8") as f:
            json.dump(report,f,indent=2)
        del best_model
    print("DONE",out/"domain_adaptation_report_DEV_ONLY.json",flush=True)
if __name__=="__main__":main()
