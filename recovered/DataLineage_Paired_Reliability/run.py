import argparse,csv,hashlib,json,math,random
from pathlib import Path
from collections import defaultdict,Counter
LABELS=['Refutes','Supports','Neutral']
def read(p):
 with open(p,encoding='utf-8-sig',newline='') as f: rows=list(csv.DictReader(f))
 req={'row_index','claim_group_sha256','gold_label','predicted_label','p_refutes','p_supports','p_neutral'}
 if not rows or not req.issubset(rows[0]):raise ValueError(f'Invalid predictions schema: {p}')
 d={}
 for r in rows:
  i=int(r['row_index']); g=r['gold_label']; pred=r['predicted_label']; pr=[float(r['p_'+x.lower()]) for x in LABELS]
  if i in d or g not in LABELS or pred not in LABELS or not all(math.isfinite(v) and 0<=v<=1 for v in pr) or abs(sum(pr)-1)>0.0001:raise ValueError(f'Invalid prediction row {i}')
  if LABELS[max(range(3),key=lambda k:pr[k])]!=pred:raise ValueError(f'Argmax mismatch row {i}')
  d[i]=(r['claim_group_sha256'],LABELS.index(g),LABELS.index(pred),pr)
 return d
def metrics(rows):
 cm=[[0]*3 for _ in range(3)]
 for _,y,p,_ in rows:cm[y][p]+=1
 n=len(rows); f=[]
 for i in range(3):
  tp=cm[i][i]; fp=sum(cm[j][i] for j in range(3))-tp; fn=sum(cm[i])-tp
  f.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0)
 return {'n':n,'accuracy':sum(cm[i][i] for i in range(3))/n,'macro_f1':sum(f)/3,'class_f1':dict(zip(LABELS,f)), 'confusion_matrix':cm,'false_supports':sum(cm[i][1] for i in [0,2]),'false_supports_from_neutral':cm[2][1],'supports_precision':cm[1][1]/sum(cm[i][1] for i in range(3)) if sum(cm[i][1] for i in range(3)) else 0}
def reliability(rows):
 n=len(rows); brier=sum(sum((pr[k]-(k==y))**2 for k in range(3)) for _,y,p,pr in rows)/n
 bins=[[] for _ in range(10)]
 for _,y,p,pr in rows:bins[min(9,int(max(pr)*10))].append((max(pr),int(y==p)))
 ece=sum(len(b)/n*abs(sum(x[0] for x in b)/len(b)-sum(x[1] for x in b)/len(b)) for b in bins if b)
 ordered=sorted(rows,key=lambda r:max(r[3]),reverse=True)
 curve=[]
 for coverage in [0.5,0.6,0.7,0.8,0.9,1.0]:
  selected=ordered[:math.ceil(n*coverage)]; m=metrics(selected)
  curve.append({'coverage':len(selected)/n,'accuracy':m['accuracy'],'macro_f1':m['macro_f1'],'false_supports':m['false_supports'],'supports_precision':m['supports_precision']})
 return {'brier_multiclass':brier,'ece_10_bin':ece,'confidence_selective_curve':curve}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--weighted',required=True);ap.add_argument('--unweighted',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bootstrap',type=int,default=2000);ap.add_argument('--seed',type=int,default=20261021);a=ap.parse_args()
 w=read(a.weighted);u=read(a.unweighted)
 if w.keys()!=u.keys():raise ValueError('Prediction row IDs differ')
 for i in w:
  if w[i][:2]!=u[i][:2]:raise ValueError(f'Gold or claim group mismatch at {i}')
 ids=sorted(w); groups=defaultdict(list)
 for i in ids:groups[w[i][0]].append(i)
 if len(groups)<2:raise ValueError('Insufficient claim groups')
 wr=[w[i] for i in ids];ur=[u[i] for i in ids];wm=metrics(wr);um=metrics(ur)
 rng=random.Random(a.seed);keys=list(groups);diff_acc=[];diff_f1=[]
 for _ in range(a.bootstrap):
  sample=[i for key in rng.choices(keys,k=len(keys)) for i in groups[key]]
  mw=metrics([w[i] for i in sample]);mu=metrics([u[i] for i in sample]);diff_acc.append(mw['accuracy']-mu['accuracy']);diff_f1.append(mw['macro_f1']-mu['macro_f1'])
 def interval(vals):
  s=sorted(vals);return [s[int(.025*(len(s)-1))],s[int(.975*(len(s)-1))]]
 result={'status':'DEVELOPMENT_ONLY_NO_INDEPENDENT_HOLDOUT','n':len(ids),'unique_claim_groups':len(groups),'bootstrap_replicates':a.bootstrap,'weighted':{'metrics':wm,'reliability':reliability(wr)},'unweighted':{'metrics':um,'reliability':reliability(ur)},'paired_weighted_minus_unweighted':{'accuracy_delta':wm['accuracy']-um['accuracy'],'accuracy_cluster_bootstrap_ci95':interval(diff_acc),'macro_f1_delta':wm['macro_f1']-um['macro_f1'],'macro_f1_cluster_bootstrap_ci95':interval(diff_f1)},'interpretation_limits':['Exploratory HEALTHVER dev only; cannot claim independent generalization','Bootstrap groups by normalized claim, not evidence source','Confidence-based selective performance is descriptive and threshold choices are not validated','No SciFact evaluation or document fusion experiment performed']}
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 (out/'paired_reliability_report_DEV_ONLY.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({'weighted_macro_f1':wm['macro_f1'],'unweighted_macro_f1':um['macro_f1'],'macro_f1_delta_ci95':result['paired_weighted_minus_unweighted']['macro_f1_cluster_bootstrap_ci95'],'output':str(out/'paired_reliability_report_DEV_ONLY.json')},indent=2))
if __name__=='__main__':main()
