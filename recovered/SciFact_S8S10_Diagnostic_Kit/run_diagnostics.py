import argparse,csv,json,zipfile,hashlib,random,statistics
from pathlib import Path
from collections import Counter,defaultdict

def macro(y,p):
    fs=[]
    for label in ('CONTRADICT','SUPPORT'):
        tp=sum(a==label and b==label for a,b in zip(y,p));fp=sum(a!=label and b==label for a,b in zip(y,p));fn=sum(a==label and b!=label for a,b in zip(y,p))
        fs.append(2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else 0)
    return sum(fs)/2

def metrics(rows):
    y=[r['gold'] for r in rows];p=[r['pred'] for r in rows]
    return {'n':len(rows),'macro_f1':macro(y,p),'accuracy':sum(a==b for a,b in zip(y,p))/len(rows) if rows else None,'confusion_matrix_labels':['CONTRADICT','SUPPORT'],'confusion_matrix':[[sum(a==x and b==z for a,b in zip(y,p)) for z in ('CONTRADICT','SUPPORT')] for x in ('CONTRADICT','SUPPORT')]}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--s7',required=True);ap.add_argument('--s5s6');ap.add_argument('--out',required=True);ap.add_argument('--bootstrap',type=int,default=10000);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(a.s7) as z:
        r=list(csv.DictReader(z.read('S7_predictions.csv').decode('utf-8-sig').splitlines()));lock=json.loads(z.read('semantic_model_lock.json'));reported=json.loads(z.read('S7_results.json'))
    assert len(r)==209 and all(x['gold'] in ('CONTRADICT','SUPPORT') for x in r)
    for x in r:
        for key in ('p_contradiction','p_entailment','p_neutral'):x[key]=float(x[key])
        x['sentence_count']=int(x['sentence_count'])
        assert abs(sum(x[k] for k in ('p_contradiction','p_entailment','p_neutral'))-1)<0.0001
    assert abs(metrics(r)['macro_f1']-reported['model_metrics']['macro_f1'])<1e-10
    assert len({x['claim_id'] for x in r})==188
    baseline=metrics(r)
    neutral_bins=[]
    for name,fn in [('neutral_below_0.25',lambda x:x['p_neutral']<.25),('neutral_0.25_to_0.5',lambda x:.25<=x['p_neutral']<.5),('neutral_0.5_to_0.75',lambda x:.5<=x['p_neutral']<.75),('neutral_at_least_0.75',lambda x:x['p_neutral']>=.75)]:
        group=[x for x in r if fn(x)];neutral_bins.append({'bin':name,'count':len(group),'errors':sum(x['gold']!=x['pred'] for x in group),'macro_f1':metrics(group)['macro_f1'] if group else None})
    # Descriptive only: thresholds are not selected from dev for any deployment claim.
    selective=[]
    for t in (0.25,0.50,0.75,0.90,0.95):
        accepted=[x for x in r if x['p_neutral']<t];rejected=[x for x in r if x['p_neutral']>=t]
        selective.append({'fixed_neutral_ceiling':t,'accepted':len(accepted),'abstained':len(rejected),'coverage':len(accepted)/len(r),'selective_accuracy':metrics(accepted)['accuracy'] if accepted else None,'selective_macro_f1':metrics(accepted)['macro_f1'] if accepted else None,'abstained_gold_distribution':dict(Counter(x['gold'] for x in rejected))})
    errors=sorted((x for x in r if x['gold']!=x['pred']),key=lambda x:-x['p_neutral'])
    with (out/'error_inventory.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['claim_id','doc_id','gold','pred','p_contradiction','p_entailment','p_neutral','sentence_count']);w.writeheader();w.writerows(errors)
    by_count=[]
    for key in ('1','2','3+'):
        group=[x for x in r if (str(x['sentence_count'])==key if key!='3+' else x['sentence_count']>=3)]
        by_count.append({'sentence_count':key,**metrics(group)})
    # Evaluate a frozen comparator only. Never fit or select thresholds on official dev.
    comparisons={}
    if a.s5s6:
        with zipfile.ZipFile(a.s5s6) as z:lex=list(csv.DictReader(z.read('S5S6_predictions.csv').decode('utf-8-sig').splitlines()))
        for name in sorted(set(x['model'] for x in lex if x['split']=='official_dev')):
            l={(x['claim_id'],x['doc_id']):x for x in lex if x['split']=='official_dev' and x['model']==name}
            if set(l)!={(x['claim_id'],x['doc_id']) for x in r}:raise ValueError('Prediction keys mismatch for '+name)
            comparisons[name]={'macro_f1':macro([x['gold'] for x in r],[l[x['claim_id'],x['doc_id']]['predicted'] for x in r]),'delta_s7_minus_lexical':baseline['macro_f1']-macro([x['gold'] for x in r],[l[x['claim_id'],x['doc_id']]['predicted'] for x in r])}
    # Cluster bootstrap for the pre-existing S7 vs majority comparison, not a new model.
    groups=defaultdict(list)
    for x in r:groups[x['claim_id']].append(x)
    ids=sorted(groups);rng=random.Random(2026);deltas=[]
    for _ in range(a.bootstrap):
        sample=[x for cid in rng.choices(ids,k=len(ids)) for x in groups[cid]]
        y=[x['gold'] for x in sample]
        deltas.append(macro(y,[x['pred'] for x in sample])-macro(y,['SUPPORT']*len(sample)))
    deltas.sort();ci=[deltas[int(.025*(len(deltas)-1))],deltas[int(.975*(len(deltas)-1))]]
    report={'status':'DESCRIPTIVE_AUDIT_ONLY_NOT_A_FUSION_EVALUATION','source_sha256':hashlib.sha256(Path(a.s7).read_bytes()).hexdigest(),'model_lock':lock,'s7_recomputed':baseline,'neutral_bins':neutral_bins,'fixed_threshold_selective_diagnostics':selective,'error_counts':dict(Counter(x['gold']+'->'+x['pred'] for x in errors)),'sentence_count_strata':by_count,'lexical_comparisons':comparisons,'cluster_bootstrap_vs_majority':{'resamples':a.bootstrap,'delta_macro_f1':baseline['macro_f1']-macro([x['gold'] for x in r],['SUPPORT']*len(r)),'ci_95':ci},'limitations':['All thresholds are post-hoc descriptive and must not be selected based on official dev for confirmatory evaluation','No source-level provenance or reliability ground truth in SciFact','No sentence-level NLI probabilities in S7 output; cannot validate alternative sentence aggregation','No provenance fusion implemented or evaluated','No test labels or retrieval evaluation','Cannot establish abstention correctness on NEI because no gold NEI labels']}
    (out/'S8S10_diagnostic_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('AUDIT_PASS',len(r),'pairs',len(ids),'claims','S7 macro-F1',round(baseline['macro_f1'],4));print('REPORT:',out/'S8S10_diagnostic_report.json')
if __name__=='__main__':main()
