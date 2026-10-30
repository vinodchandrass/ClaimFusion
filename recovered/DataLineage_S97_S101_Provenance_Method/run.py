#!/usr/bin/env python3
"""Development-only provenance audit and prespecified candidate fusion; never reads test files."""
import argparse,csv,json,hashlib,collections,re,pathlib,math

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def readcsv(p):
 with open(p,encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields):
 with open(p,'w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 ap=argparse.ArgumentParser()
 ap.add_argument('--candidates',required=True);ap.add_argument('--corpus');ap.add_argument('--reviewed-metadata',help='Manually verified CSV: doc_id,doi,verification_status,source_reference; only VERIFIED entries used')
 ap.add_argument('--out',required=True)
 a=ap.parse_args();out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
 for path in (a.candidates,a.corpus,a.reviewed_metadata):
  if path and ('test' in pathlib.Path(path).name.lower() or 'holdout' in pathlib.Path(path).name.lower()):raise SystemExit('Refusing test/holdout-named input')
 rows=readcsv(a.candidates);needed={'claim_id','doc_id','doc_rank','sentence','p_contradiction','p_entailment'}
 if not rows or not needed.issubset(rows[0]):raise SystemExit('Missing candidate columns')
 corpus={};dups=0
 if a.corpus:
  with open(a.corpus,encoding='utf-8') as f:
   for line in f:
    x=json.loads(line);k=str(x.get('doc_id',''))
    if k in corpus:dups+=1
    corpus[k]=x
 metadata={};rejected=0
 if a.reviewed_metadata:
  for r in readcsv(a.reviewed_metadata):
   d=r.get('doc_id','').strip();doi=r.get('doi','').strip().lower().replace('https://doi.org/','').replace('doi:','')
   if r.get('verification_status','').strip().upper()=='VERIFIED' and doi and r.get('source_reference','').strip() and re.fullmatch(r'10\.\d{4,9}/\S+',doi):
    if d in metadata and metadata[d]!=doi:raise SystemExit('Conflicting verified DOI for '+d)
    metadata[d]=doi
   else:rejected+=1
 doc_ids=sorted({str(r['doc_id']) for r in rows});manifest=[]
 for d in doc_ids:
  x=corpus.get(d,{});manifest.append({'doc_id':d,'title':x.get('title',''),'corpus_match':bool(d in corpus),'verified_doi':metadata.get(d,''),'verification_status':'VERIFIED' if d in metadata else 'UNKNOWN','source_reference':'See independently reviewed metadata CSV' if d in metadata else ''})
 writecsv(out/'s97_source_manifest.csv',manifest,list(manifest[0]))
 by_claim=collections.defaultdict(lambda:collections.defaultdict(list))
 for r in rows:
  try: c=float(r['p_contradiction']);e=float(r['p_entailment']);rank=int(r['doc_rank'])
  except (ValueError,TypeError):raise SystemExit('Invalid numeric candidate values')
  if not (0<=c<=1 and 0<=e<=1):raise SystemExit('Invalid probability')
  by_claim[str(r['claim_id'])][(rank,str(r['doc_id']))].append((e,c))
 diagnostics=[];reused=0;fallback=0;unknown_claims=0
 for cid,docs in sorted(by_claim.items()):
  ordered=sorted(docs)
  if not ordered:continue
  summaries=[]
  for rank,d in ordered:
   vals=docs[(rank,d)];s=sum(z[0] for z in vals)/len(vals);c=sum(z[1] for z in vals)/len(vals)
   summaries.append((rank,d,s,c,metadata.get(d)))
  top=summaries[0];top_label='SUPPORTS' if top[2]>top[3] else 'REFUTES'
  known=[x for x in summaries if x[4]];unknown=len(summaries)-len(known)
  if unknown:unknown_claims+=1
  # Only DOI-verified documents may be deduplicated. Unknowns remain distinct *identities*, not independent sources.
  groups=collections.defaultdict(list)
  for rank,d,s,c,doi in summaries:groups['DOI:'+doi if doi else 'UNKNOWN_DOC:'+d].append((rank,s,c))
  collapsed=[]
  for key,v in groups.items():
   collapsed.append((min(z[0] for z in v),sum(z[1] for z in v)/len(v),sum(z[2] for z in v)/len(v)))
  collapsed.sort()
  if len(collapsed)<len(summaries):reused+=1
  # Prespecified, untuned rank weights 1/rank; exploratory only, not proven superior.
  weights=[1/max(1,x[0]) for x in collapsed];tot=sum(weights)
  fs=sum(w*x[1] for w,x in zip(weights,collapsed))/tot
  fc=sum(w*x[2] for w,x in zip(weights,collapsed))/tot
  fused='SUPPORTS' if fs>fc else 'REFUTES'
  # Safety: no provenance-confirmed cross-document relation => no claim of provenance-aware improvement.
  eligible=len(known)>=2
  if not eligible:fallback+=1
  diagnostics.append({'claim_id':cid,'documents':len(summaries),'verified_doi_documents':len(known),'unknown_provenance_documents':unknown,'verified_doi_collapsed_groups':len(collapsed),'top_doc_prediction':top_label,'exploratory_grouped_prediction':fused,'grouping_supported_by_verified_doi':eligible,'operational_prediction':fused if eligible else top_label,'operational_fallback':not eligible})
 writecsv(out/'s99_claim_predictions_UNLABELED.csv',diagnostics,list(diagnostics[0]))
 report={'stage':'S97-S101','candidate_rows':len(rows),'claim_count':len(by_claim),'document_ids':len(doc_ids),'corpus_provided':bool(a.corpus),'corpus_matched_doc_ids':sum(d in corpus for d in doc_ids) if a.corpus else None,'duplicate_corpus_ids':dups,'verified_doi_document_ids':sum(d in metadata for d in doc_ids),'rejected_unverified_metadata_rows':rejected,'claims_with_unknown_provenance':unknown_claims,'claims_with_confirmed_duplicate_doi_groups':reused,'operational_fallback_claims':fallback,'candidate_sha256':sha(a.candidates),'corpus_sha256':sha(a.corpus) if a.corpus else None,'reviewed_metadata_sha256':sha(a.reviewed_metadata) if a.reviewed_metadata else None,'test_accessed':False,'gold_labels_read':False,'accuracy_evaluated':False,'external_lookup_performed':False,'independence_established':False,'gate':'DEVELOPMENT_ONLY_PROVENANCE_INCOMPLETE' if fallback else 'REVIEWED_DOI_GROUPING_AVAILABLE_NOT_INDEPENDENCE_ATTESTED','limitations':['DOI equality is evidence of common publication identity, not independence across different DOIs.','The original SciFact corpus has no native DOI field; verified metadata must be supplied separately.','The exploratory fusion rule is not tuned or independently validated.','Candidate file may contain DEV_ONLY columns; this program ignores all annotation/gold fields.']}
 (out/'s101_provenance_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
