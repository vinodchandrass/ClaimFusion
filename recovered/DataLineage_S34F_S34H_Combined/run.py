import argparse,csv,json,hashlib,collections,random,math
from pathlib import Path

def read(p):
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def decide(e,c,t=.5):return 'SUPPORTS' if e>=t and e>c else 'REFUTES' if c>=t and c>e else 'NOT ENOUGH INFO'
def metric(rs,k):
 n=len(rs); covered=[r for r in rs if r[k]!='NOT ENOUGH INFO']; correct=lambda a:sum(r[k]==r['gold'] for r in a)
 return {'n':n,'accuracy':correct(rs)/n if n else None,'coverage':len(covered)/n if n else None,'selective_accuracy':correct(covered)/len(covered) if covered else None,'correct':correct(rs),'abstentions':n-len(covered)}
def boot(rs,a,b,B,seed):
 d=[int(r[b]==r['gold'])-int(r[a]==r['gold']) for r in rs];n=len(d)
 if not n:return None
 rng=random.Random(seed); vals=sorted(sum(d[rng.randrange(n)] for _ in range(n))/n for _ in range(B));return {'point':sum(d)/n,'ci95':[vals[int(.025*(B-1))],vals[int(.975*(B-1))]],'n_boot':B,'improved':sum(v>0 for v in d),'regressed':sum(v<0 for v in d)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',default=r'D:\DataLineage');ap.add_argument('--source');ap.add_argument('--cache');ap.add_argument('--s34e');ap.add_argument('--out');ap.add_argument('--bootstrap',type=int,default=2000);a=ap.parse_args();root=Path(a.base)
 source=Path(a.source) if a.source else root/'S32_FEVER_Resolved/S32_resolved_evidence.csv'
 cache=Path(a.cache) if a.cache else root/'S33_S35_Validation_Output/s34_s35_dev/pair_probabilities.csv'
 prior=Path(a.s34e) if a.s34e else root/'S33_S35_Validation_Output/s34e_combined/s34e_claim_predictions.csv'
 out=Path(a.out) if a.out else root/'S33_S35_Validation_Output/s34f_s34h_combined';out.mkdir(parents=True,exist_ok=True)
 scores={}
 for r in read(cache):
  k=(r['claim_id'],r['page_id'],r['sentence_index']);assert k not in scores,('duplicate_cache_key',k)
  scores[k]=(float(r['p_entailment']),float(r['p_contradiction']))
 byclaim=collections.defaultdict(list);groups=collections.defaultdict(lambda:collections.defaultdict(list));gold={};unresolved=collections.Counter();allrows=collections.Counter()
 for r in read(source):
  if r['split']!='dev':continue
  cid=r['claim_id'];gold[cid]=r['gold_label'];allrows[cid]+=1
  if r['resolved']=='1' and r['evidence_text'].strip():
   k=(cid,r['page_id'],r['sentence_index']);assert k in scores,('missing_pair',k)
   byclaim[cid].append(k);groups[cid][r['group_index']].append(k)
  else:unresolved[cid]+=1
 old={r['claim_id']:r for r in read(prior)}
 assert len(scores)==11047 and len(byclaim)==6616 and len(old)==6616
 results=[];tax=collections.Counter();witness=[]
 for cid,raw in byclaim.items():
  keys=sorted(set(raw));pages=collections.defaultdict(list)
  for k in keys:pages[k[1]].append(k)
  E=max(scores[k][0] for k in keys);C=max(scores[k][1] for k in keys);base=decide(E,C)
  assert base==old[cid]['baseline'],('baseline_parity',cid)
  # Both high values may be from same pair or distinct pairs, possibly same or different page.
  ekeys=[k for k in keys if scores[k][0]>=.5];ckeys=[k for k in keys if scores[k][1]>=.5]
  conflict=bool(ekeys and ckeys)
  samepair=any(k in set(ckeys) for k in ekeys) if conflict else False
  crosspage=any(e[1]!=c[1] for e in ekeys for c in ckeys) if conflict else False
  samepage=any(e[1]==c[1] for e in ekeys for c in ckeys) if conflict else False
  conflict_type=('none' if not conflict else 'same_pair' if samepair else 'cross_page_only' if crosspage and not samepage else 'same_page_only' if samepage and not crosspage else 'mixed_pages')
  # Frozen, interpretable policies. Not a fitted model and not a claim of provenance trust.
  # Cross-page-only guard: abstain ONLY when all strong opposing signals are on different pages.
  cross_guard='NOT ENOUGH INFO' if conflict_type=='cross_page_only' else base
  # Single-page preference: choose strongest page-local label by largest class score;
  # tie broken by page_id lexicographic order, deterministic.
  pagechoices=[]
  for page,ks in pages.items():
   pe=max(scores[k][0] for k in ks);pc=max(scores[k][1] for k in ks)
   pagechoices.append((max(pe,pc),page,decide(pe,pc),pe,pc))
  pagechoices.sort(key=lambda z:(-z[0],z[1]));page_label=pagechoices[0][2]
  # Conservative agreement: if two or more pages each produce a decisive label and disagree, abstain.
  decisive={x[2] for x in pagechoices if x[2]!='NOT ENOUGH INFO'}
  page_agreement='NOT ENOUGH INFO' if len(decisive)>1 else base
  # Page-local strongest decision, irrespective of conflicting other pages (exploratory).
  page_winner=page_label
  # Evidence group composition is oracle metadata: diagnostics ONLY, never a feature in candidate rules.
  group_counts=[len(set(ks)) for ks in groups[cid].values()]
  row={'claim_id':cid,'gold':gold[cid],'baseline':base,'s34e_conflict_guard':old[cid]['conflict_guard'],'cross_page_only_guard':cross_guard,'page_disagreement_guard':page_agreement,'strongest_page':page_winner,'conflict_type':conflict_type,'conflict':int(conflict),'same_pair_conflict':int(samepair),'same_page_conflict':int(samepage),'cross_page_conflict':int(crosspage),'distinct_pages':len(pages),'unique_pairs':len(keys),'duplicate_rows':len(raw)-len(keys),'n_groups_oracle':len(groups[cid]),'max_group_size_oracle':max(group_counts),'unresolved_rows':unresolved[cid],'max_entailment':E,'max_contradiction':C,'max_probability_margin':abs(E-C),'top_page_id':pagechoices[0][1]}
  assert (row['duplicate_rows']>=0 and decide(max(scores[k][0] for k in raw),max(scores[k][1] for k in raw))==base)
  results.append(row);tax[(conflict_type,gold[cid],base)]+=1
  if conflict and len(witness)<100:witness.append({'claim_id':cid,'type':conflict_type,'evidence_entailment':sorted(ekeys),'evidence_contradiction':sorted(ckeys)})
 results.sort(key=lambda r:r['claim_id']);methods=['baseline','s34e_conflict_guard','cross_page_only_guard','page_disagreement_guard','strongest_page']
 report={'status':'EXPLORATORY_DEV_ONLY_NOT_INDEPENDENT_PROVENANCE','input_sha256':{str(p):sha(p) for p in (source,cache,prior)},'n':len(results),'frozen_threshold':.5,'baseline_parity':'PASS','duplicate_invariance':'PASS','policy_definitions':{'cross_page_only_guard':'abstain only when strong opposing labels occur exclusively on different pages','page_disagreement_guard':'abstain if at least two page-local decisive labels disagree','strongest_page':'label from page with highest maximum entailment or contradiction score; tie by page id'},'methods':{k:metric(results,k) for k in methods},'paired':{},'strata':{},'conflict_taxonomy':dict(collections.Counter(r['conflict_type'] for r in results)),'class_by_conflict':{},'limitations':['Development-only FEVER gold evidence; gold NEI absent','Page identifiers are not verified independent provenance','Gold group membership used only for descriptive audit, not policy','No held-out test, retrieval, independent SciFact, or calibrated threshold','Multiple exploratory policies compared: CIs are descriptive, not confirmatory','Page-based methods are not suitable for single-page independent-source claims']}
 for k in methods[1:]:report['paired'][k]=boot(results,'baseline',k,a.bootstrap,34060+methods.index(k))
 strata={'single_page':lambda r:r['distinct_pages']==1,'multi_page':lambda r:r['distinct_pages']>1,'conflict':lambda r:r['conflict']==1,'no_conflict':lambda r:r['conflict']==0,'duplicate':lambda r:r['duplicate_rows']>0,'no_duplicate':lambda r:r['duplicate_rows']==0,'supports':lambda r:r['gold']=='SUPPORTS','refutes':lambda r:r['gold']=='REFUTES'}
 for typ in ('same_pair','cross_page_only','same_page_only','mixed_pages'):strata[typ]=lambda r,t=typ:r['conflict_type']==t
 for name,fn in strata.items():
  sub=[r for r in results if fn(r)];report['strata'][name]={k:metric(sub,k) for k in methods}
 for typ in ('none','same_pair','cross_page_only','same_page_only','mixed_pages'):
  sub=[r for r in results if r['conflict_type']==typ];report['class_by_conflict'][typ]={label:sum(r['gold']==label for r in sub) for label in ('SUPPORTS','REFUTES')}
 # Coverage-risk curves: vary a single abstention margin, pre-specified descriptive grid, no selection.
 curves=[]
 for t in (0,.01,.025,.05,.1,.2,.3,.4,.5):
  chosen=[r for r in results if r['baseline']!='NOT ENOUGH INFO' and r['max_probability_margin']>=t]
  curves.append({'margin':t,'coverage':len(chosen)/len(results),'selective_accuracy':sum(r['baseline']==r['gold'] for r in chosen)/len(chosen) if chosen else None})
 report['descriptive_coverage_risk_grid']=curves
 with (out/'s34f_s34h_claim_predictions.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(results[0]));w.writeheader();w.writerows(results)
 with (out/'s34f_s34h_conflict_taxonomy.csv').open('w',encoding='utf-8',newline='') as f:
  w=csv.writer(f);w.writerow(['conflict_type','gold','baseline','count']);w.writerows([(*k,v) for k,v in sorted(tax.items())])
 (out/'s34f_s34h_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 (out/'s34f_s34h_witness_sample.json').write_text(json.dumps(witness,indent=2),encoding='utf-8')
 print('PASS',len(results),'claims; report:',out/'s34f_s34h_report.json')
 print(json.dumps({'methods':report['methods'],'conflict_taxonomy':report['conflict_taxonomy']},indent=2))
if __name__=='__main__':main()
