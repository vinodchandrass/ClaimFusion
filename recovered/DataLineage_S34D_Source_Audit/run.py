import argparse,csv,json,hashlib,collections,math
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',default=r'D:\DataLineage');z=a.parse_args();base=Path(z.base)
 source=base/'S32_FEVER_Resolved'/'S32_resolved_evidence.csv';cache=base/'S33_S35_Validation_Output'/'s34_s35_dev'/'pair_probabilities.csv';out=base/'S33_S35_Validation_Output'/'s34d_source_audit';out.mkdir(parents=True,exist_ok=True)
 scores={}
 with cache.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   k=(r['claim_id'],r['page_id'],r['sentence_index']);assert k not in scores,k
   v=[float(r['p_'+c]) for c in ('contradiction','entailment','neutral')];assert all(math.isfinite(t) and 0<=t<=1 for t in v) and abs(sum(v)-1)<1e-3
   scores[k]=v
 claims=collections.defaultdict(list);groups=collections.defaultdict(lambda:collections.defaultdict(list));gold={};unresolved=collections.Counter();rows=0
 with source.open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   if r['split']!='dev':continue
   rows+=1;cid=r['claim_id'];key=(cid,r['page_id'],r['sentence_index']);gold[cid]=r['gold_label']
   if r['resolved']=='1' and r['evidence_text'].strip():
    if key not in scores:raise ValueError('Missing cached pair '+str(key))
    claims[cid].append(key);groups[cid][r['group_index']].append(key)
   else:unresolved[cid]+=1
 assert len(scores)==11047, 'Unexpected cache size: investigate before interpreting'
 diagnostics=[];pair_reuse=collections.Counter();page_reuse=collections.Counter();multi_page=0
 for cid,keys in claims.items():
  unique=list(dict.fromkeys(keys));pages={k[1] for k in unique};pair_reuse.update(keys);page_reuse.update(k[1] for k in unique)
  gp=groups[cid];group_signatures=[tuple(sorted(set(v))) for v in gp.values()];dup_groups=len(group_signatures)-len(set(group_signatures))
  by_page=collections.Counter(k[1] for k in unique)
  diagnostics.append({'claim_id':cid,'gold_label':gold[cid],'unique_pairs':len(unique),'resolved_rows':len(keys),'distinct_pages':len(pages),'duplicate_pair_rows':len(keys)-len(unique),'evidence_groups':len(gp),'duplicate_group_signatures':dup_groups,'unresolved_rows':unresolved[cid],'largest_page_share':max(by_page.values())/len(unique),'max_entailment':max(scores[k][1] for k in unique),'max_contradiction':max(scores[k][0] for k in unique)})
 fields=list(diagnostics[0]);path=out/'s34d_claim_source_diagnostics.csv'
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(diagnostics)
 report={'status':'DESCRIPTIVE_SOURCE_STRUCTURE_AUDIT_NOT_PROVENANCE_VALIDATION','source_sha256':sha(source),'pair_cache_sha256':sha(cache),'development_rows':rows,'cached_unique_pairs':len(scores),'claims_with_resolved_evidence':len(diagnostics),'claims_with_multiple_pages':sum(int(int(r['distinct_pages'])>1) for r in diagnostics),'claims_with_duplicate_evidence_rows':sum(int(int(r['duplicate_pair_rows'])>0) for r in diagnostics),'claims_with_duplicate_group_signatures':sum(int(int(r['duplicate_group_signatures'])>0) for r in diagnostics),'claims_with_unresolved_rows':sum(int(int(r['unresolved_rows'])>0) for r in diagnostics),'distinct_pages_in_development':len(page_reuse),'repeated_pages_across_claims':sum(v>1 for v in page_reuse.values()),'available_attributes':['page_id','sentence_index','group_index','resolved','evidence_text','NLI probabilities'],'not_established':['source reliability','independent document authorship','document publication time','retrieval provenance','source independence','NEI gold performance','independent SciFact transfer'],'decision':'DO_NOT_LABEL_PAGE_DIVERSITY_AS_INDEPENDENT_PROVENANCE'}
 (out/'s34d_source_audit_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
