import argparse,csv,hashlib,json,re
from pathlib import Path
from collections import Counter

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def norm(s):return re.sub(r'\s+',' ',str(s or '')).strip().casefold()
def main():
 ap=argparse.ArgumentParser(description='SciFact development-only corpus metadata recovery; NEVER read test claims.')
 ap.add_argument('--candidates',required=True);ap.add_argument('--corpus',required=True);ap.add_argument('--out',required=True)
 a=ap.parse_args();cp=Path(a.candidates);corpus=Path(a.corpus);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 required={'claim_id','doc_id','doc_rank','sentence'}
 with cp.open(newline='',encoding='utf-8-sig') as f:
  rd=csv.DictReader(f);assert required.issubset(set(rd.fieldnames or [])), 'candidate schema mismatch'
  pairs={}; docs=set(); n=0
  for r in rd:
   n+=1; key=(r['claim_id'],str(r['doc_id']));docs.add(key[1]);pairs.setdefault(key,[]).append(r)
 assert n>0,'empty candidate file'
 meta={}; duplicate_ids=[]; malformed=0
 with corpus.open(encoding='utf-8') as f:
  for line in f:
   if not line.strip():continue
   obj=json.loads(line); idv=str(obj.get('doc_id',obj.get('id','')))
   if not idv:malformed+=1;continue
   if idv in meta:duplicate_ids.append(idv);continue
   if idv not in docs:continue
   abstract=obj.get('abstract',[])
   if isinstance(abstract,str):abstract=[abstract]
   if not isinstance(abstract,list):abstract=[]
   meta[idv]={'title':obj.get('title',''),'abstract':abstract,'keys':sorted(obj.keys()),'doi':obj.get('doi',''),'authors':obj.get('authors',''),'year':obj.get('year',''),'url':obj.get('url','')}
 rows=[]; source_fields=Counter()
 for idv in sorted(docs):
  m=meta.get(idv)
  if m:
   for k in ('doi','authors','year','url'):source_fields[k]+=bool(m.get(k))
  rows.append({'doc_id':idv,'corpus_found':int(m is not None),'title':m['title'] if m else '', 'abstract_sentence_count':len(m['abstract']) if m else 0,'doi_in_corpus':str(m['doi']) if m else '', 'authors_in_corpus':str(m['authors']) if m else '', 'year_in_corpus':str(m['year']) if m else '', 'url_in_corpus':str(m['url']) if m else '', 'available_keys':';'.join(m['keys']) if m else ''})
 with (out/'s92_document_metadata.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 audits=[]; unmatched=0
 for (cid,idv),rlist in sorted(pairs.items()):
  m=meta.get(idv); abstract=[norm(x) for x in m['abstract']] if m else []
  for r in rlist:
   idx=int(r['sentence_index']) if str(r.get('sentence_index','')).isdigit() else -1
   txt=norm(r['sentence']); indexed=bool(m and 0<=idx<len(abstract) and txt==abstract[idx]); anywhere=bool(m and txt in abstract)
   unmatched+=not anywhere
   audits.append({'claim_id':cid,'doc_id':idv,'doc_rank':r['doc_rank'],'sentence_index':r.get('sentence_index',''),'index_text_match':int(indexed),'any_abstract_sentence_match':int(anywhere),'corpus_found':int(m is not None)})
 with (out/'s93_sentence_alignment.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=audits[0].keys());w.writeheader();w.writerows(audits)
 missing=len(docs)-len(meta); found=len(meta)
 report={'stage':'S92-S96','candidate_sha256':digest(cp),'corpus_sha256':digest(corpus),'candidate_rows':n,'unique_retrieved_doc_ids':len(docs),'corpus_matched_doc_ids':found,'missing_doc_ids':missing,'duplicate_corpus_ids':len(duplicate_ids),'malformed_corpus_id_rows':malformed,'aligned_abstract_sentences_anywhere':n-unmatched,'unaligned_candidate_sentences':unmatched,'explicit_metadata_fields_in_corpus':dict(source_fields),'external_bibliographic_lookup_performed':False,'test_accessed':False,'new_model_inference':False,'claim_accuracy_evaluated':False,'independence_attested':False,'source_independence_identifiable':False,'gate': 'CORPUS_ALIGNMENT_ONLY_NOT_SOURCE_INDEPENDENCE' if missing==0 and unmatched==0 else 'CORPUS_ALIGNMENT_INCOMPLETE','interpretation':'Title and abstract alignment authenticate corpus linkage only; they do not establish independent publishers, citation edges, or evidence reliability.'}
 (out/'s96_metadata_gate.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
