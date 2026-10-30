import argparse,csv,hashlib,json,sys,traceback,zipfile
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,confusion_matrix,classification_report,accuracy_score

SEEDS=[1103,2207,3301,4409,5501]
CLASSES=['CONTRADICT','SUPPORT']
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def readjsonl(p):
 with open(p,encoding='utf-8-sig') as f:
  return [json.loads(x) for x in f if x.strip()]
def norm(s): return ' '.join(s.casefold().split())
def ids(p):
 with open(p,encoding='utf-8-sig',newline='') as f:
  return {str(x['claim_id']) for x in csv.DictReader(f)}
def build(root, split, allowed=None):
 claims=readjsonl(root/('claims_'+split+'.jsonl'))
 corpus={str(x.get('doc_id')):x for x in readjsonl(root/'corpus.jsonl')}
 output=[]; issues=[]; excluded=Counter()
 for claim in claims:
  cid=str(claim['id'])
  if allowed is not None and cid not in allowed: continue
  evidence=claim.get('evidence',{})
  if not isinstance(evidence,dict): raise ValueError('Malformed evidence for '+cid)
  for docid,groups in evidence.items():
   if not groups: continue
   if docid not in corpus: raise ValueError('Missing corpus document '+docid)
   doc=corpus[docid]; abstract=doc.get('abstract',[])
   if not isinstance(abstract,list): raise ValueError('Abstract not sentence list')
   labels={g.get('label') for g in groups}
   if any(l not in CLASSES for l in labels): raise ValueError('Unexpected label '+str(labels))
   if len(labels)!=1:
    issues.append({'split':split,'claim_id':cid,'doc_id':docid,'reason':'MIXED_LABELS'});excluded['mixed_labels']+=1;continue
   sentids=sorted({i for g in groups for i in g.get('sentences',[])})
   if any(not isinstance(i,int) or i<0 or i>=len(abstract) for i in sentids):
    raise ValueError('Invalid sentence index '+cid+'/'+docid)
   if not sentids: issues.append({'split':split,'claim_id':cid,'doc_id':docid,'reason':'EMPTY_SENTENCES'});excluded['empty_sentences']+=1;continue
   output.append({'split':split,'claim_id':cid,'doc_id':docid,'claim':claim['claim'],
    'evidence':' '.join(str(abstract[i]) for i in sentids),'label':next(iter(labels)),
    'sentence_ids':sentids,'group_count':len(groups)})
 return output,issues,dict(excluded)
def writecsv(path,rows):
 if not rows: return
 with open(path,'w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def evaluate(tr,dev,seed,bootstrap_n):
 ytr=np.array([r['label'] for r in tr]); ydev=np.array([r['label'] for r in dev]);
 if len(set(ytr))!=2 or len(set(ydev))!=2: raise ValueError('Need both labels in train and dev')
 groups=np.array([r['claim_id'] for r in tr]);
 if len(set(groups))<3: raise ValueError('Too few training claims')
 # Predeclare C candidates; inner CV is grouped by claim ID. TF-IDF fit on each inner training partition only.
 Cs=[0.1,1.0,10.0]
 def matrix(train,valid,mode):
  def txt(r):
   if mode=='claim': return r['claim']
   if mode=='evidence': return r['evidence']
   return 'CLAIM '+r['claim']+' EVIDENCE '+r['evidence']
  vect=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=30000,sublinear_tf=True)
  a=vect.fit_transform([txt(r) for r in train]);b=vect.transform([txt(r) for r in valid]);return a,b
 def fit_predict(train,valid,mode,C):
  a,b=matrix(train,valid,mode)
  model=LogisticRegression(C=C,max_iter=1500,random_state=seed,solver='liblinear')
  model.fit(a,[r['label'] for r in train]);return model.predict(b)
 folds=list(GroupKFold(n_splits=3).split(np.zeros(len(tr)),ytr,groups))
 def tune(mode):
  scores=[]
  for C in Cs:
   fold_scores=[]
   for ti,vi in folds:
    t=[tr[i] for i in ti];v=[tr[i] for i in vi]
    if len({x['label'] for x in t})<2: continue
    pred=fit_predict(t,v,mode,C)
    fold_scores.append(f1_score([x['label'] for x in v],pred,labels=CLASSES,average='macro',zero_division=0))
   scores.append(float(np.mean(fold_scores)) if fold_scores else -1)
  return Cs[int(np.argmax(scores))],scores
 models={};predictions={};tuning={}
 maj=Counter(ytr).most_common(1)[0][0]
 predictions['majority']=np.array([maj]*len(dev))
 for mode in ['claim','evidence','pair']:
  C,scores=tune(mode);tuning[mode]={'selected_C':C,'cv_macro_f1':scores}
  predictions['tfidf_'+mode]=fit_predict(tr,dev,mode,C)
 # This is an explicitly exploratory feature-fusion control, NOT a novel method.
 # Combined text is already a standard linear baseline. Do not call it a proposed method.
 def summary(pred):
  return {'macro_f1':float(f1_score(ydev,pred,labels=CLASSES,average='macro',zero_division=0)),
          'accuracy':float(accuracy_score(ydev,pred)),
          'confusion_matrix':confusion_matrix(ydev,pred,labels=CLASSES).tolist(),
          'per_class':classification_report(ydev,pred,labels=CLASSES,output_dict=True,zero_division=0)}
 for k,p in predictions.items():models[k]=summary(p)
 # Paired claim-cluster bootstrap. Report intervals for scores and pair-minus-evidence contrast.
 claims=defaultdict(list)
 for i,r in enumerate(dev):claims[r['claim_id']].append(i)
 claim_ids=sorted(claims)
 rng=np.random.default_rng(seed)
 dist={k:[] for k in predictions}; diff=[]
 for _ in range(bootstrap_n):
  selected=rng.choice(claim_ids,size=len(claim_ids),replace=True)
  indices=np.array([i for c in selected for i in claims[c]],dtype=int)
  vals={k:float(f1_score(ydev[indices],p[indices],labels=CLASSES,average='macro',zero_division=0)) for k,p in predictions.items()}
  for k,v in vals.items():dist[k].append(v)
  diff.append(vals['tfidf_pair']-vals['tfidf_evidence'])
 for k in models:models[k]['claim_cluster_bootstrap_95ci']=list(map(float,np.quantile(dist[k],[0.025,0.975])))
 comparison={'pair_minus_evidence_macro_f1':models['tfidf_pair']['macro_f1']-models['tfidf_evidence']['macro_f1'],
  'paired_claim_cluster_bootstrap_95ci':list(map(float,np.quantile(diff,[0.025,0.975]))),
  'bootstrap_samples':bootstrap_n,'claim_clusters':len(claim_ids)}
 pred_rows=[]
 for i,r in enumerate(dev):
  pred_rows.append({'claim_id':r['claim_id'],'doc_id':r['doc_id'],'gold':r['label'],**{k:str(p[i]) for k,p in predictions.items()}})
 return models,tuning,comparison,pred_rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--s2',required=True);ap.add_argument('--out',required=True);ap.add_argument('--bootstrap',type=int,default=10000);args=ap.parse_args()
 root=Path(args.root);s2=Path(args.s2);out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 for fn in ['claims_train.jsonl','claims_dev.jsonl','corpus.jsonl']:
  if not (root/fn).exists():raise FileNotFoundError(root/fn)
 manifest=json.loads((s2/'s2_protocol_manifest.json').read_text(encoding='utf-8-sig'))
 for fn,expected in manifest['source_sha256'].items():
  p=root/Path(fn.replace('\\','/'))
  if not p.is_file() or sha(p)!=expected:raise ValueError('S2 source hash mismatch: '+str(p))
 allowed=ids(s2/'official_dev_training_ids.csv')
 train,itr,extrain=build(root,'train',allowed)
 dev,idev,exdev=build(root,'dev')
 if not train or not dev:raise ValueError('No labelled observations')
 # Critical guard: training claim text must not occur in evaluation.
 if {norm(r['claim']) for r in train}&{norm(r['claim']) for r in dev}:
  raise ValueError('TRAIN/DEV EXACT TEXT LEAKAGE AFTER EXCLUSIONS')
 if set(r['claim_id'] for r in train)&set(r['claim_id'] for r in dev):raise ValueError('TRAIN/DEV CLAIM ID OVERLAP')
 seed=SEEDS[0];models,tuning,comparison,pred_rows=evaluate(train,dev,seed,args.bootstrap)
 writecsv(out/'official_dev_predictions.csv',pred_rows)
 writecsv(out/'excluded_ambiguous_pairs.csv',itr+idev)
 report={'stage':'S3/S4 baseline execution, not a new claim-validity fusion model',
 'status':'BASELINES_COMPLETED_PROPOSED_METHOD_NOT_YET_DEFINED',
 'source_hashes_checked':True,'training_claim_ids_allowed':len(allowed),
 'train_labelled_pairs':len(train),'dev_labelled_pairs':len(dev),
 'train_unique_claims':len({r['claim_id'] for r in train}),
 'dev_unique_claims':len({r['claim_id'] for r in dev}),
 'train_labels':dict(Counter(r['label'] for r in train)),
 'dev_labels':dict(Counter(r['label'] for r in dev)),
 'excluded_train':extrain,'excluded_dev':exdev,
 'models':models,'inner_group_cv_tuning':tuning,'paired_comparison':comparison,
 'limitations':['Gold-evidence oracle setting: evidence sentences were selected from annotations, unavailable in realistic retrieval',
 'Test set not evaluated','No pretrained neural baseline','No novel fusion model was defined or evaluated',
 'Repeated document or shared corpus effects not removed by claim-cluster bootstrap',
 'Do not interpret bootstrap confidence intervals as proof of generalization'],
 'seed':seed,'other_prespecified_seeds_reserved_for_future_robustness':SEEDS[1:]}
 (out/'S3S4_results.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
 print(json.dumps({'status':report['status'],'train_pairs':len(train),'dev_pairs':len(dev),
  'macro_f1':{k:round(v['macro_f1'],4) for k,v in models.items()},'out':str(out)},indent=2))
if __name__=='__main__':
 try:main()
 except Exception as e:print('S3S4_FAILED:',str(e),file=sys.stderr);traceback.print_exc();sys.exit(2)
