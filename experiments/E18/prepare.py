"""Verify original candidates/qrels and preselect timing/reserve fixtures."""
import csv
import json
import time
import numpy as np
from transformers import AutoTokenizer
from .common import ROOT,HERE,ASSETS,DATA,rows,fixtures,sha,save


def main():
    data = rows()
    old = json.loads((ROOT/'experiments/E17/plan.json').read_text())
    for relative,digest in old['dataset_source_sha256'].items():
        assert sha(ROOT/relative)==digest, relative
    for dataset in ('scifact','nfcorpus'):
        folder = ROOT/'data/beir'/dataset
        corpus = {r['_id']:(r.get('title','')+' '+r['text']).strip() for r in map(json.loads,(folder/'corpus.jsonl').open())}
        queries = {r['_id']:r['text'] for r in map(json.loads,(folder/'queries.jsonl').open())}
        qrels = {}
        split = 'train' if dataset=='scifact' else 'dev'
        for r in csv.DictReader((folder/f'qrels/{split}.tsv').open(),delimiter='\t'):
            qrels.setdefault(r['query-id'],{})[r['corpus-id']]=int(r['score'])
        for r in data:
            if r['dataset']==dataset:
                assert r['query']==queries[r['query_id']] and r['qrels']==qrels[r['query_id']]
                assert r['texts']==[corpus[d] for d in r['document_ids']]
    selected,reserved = fixtures(),fixtures(True)
    assert len(selected)==64 and len(reserved)==421
    tokenizer = AutoTokenizer.from_pretrained(ASSETS,local_files_only=True,use_fast=True,trust_remote_code=False)
    distributions = {}
    for dataset in ('scifact','nfcorpus'):
        samples = [(r['query'],t) for r in selected if r['dataset']==dataset for t in r['texts']]
        lengths = []
        started = time.perf_counter()
        for offset in range(0,len(samples),256):
            chunk = samples[offset:offset+256]
            encoded = tokenizer([q for q,d in chunk],[d for q,d in chunk],padding=False,truncation=False,
                                return_length=True,verbose=False)
            lengths += encoded['length']
        seconds = time.perf_counter()-started
        distributions[dataset] = {'pairs':len(samples),'original_combined_token_length':{
            'min':min(lengths),'max':max(lengths),'median':float(np.median(lengths)),
            'p95':float(np.percentile(lengths,95))},
            'truncation_fraction':{str(cap):float(np.mean(np.array(lengths)>cap)) for cap in (128,256,512)},
            'untruncated_tokenizer_pairs_per_second':len(samples)/seconds}
    save(HERE/'data_manifest.json', {'status':'verified','candidate_sha256':sha(DATA),
         'development_queries':485,'top100_pairs':48500,'datasets':{'scifact':162,'nfcorpus':323},
         'source_sha256':old['dataset_source_sha256'],
         'preparation_provenance_sha256':sha(ROOT/'experiments/E16/data_plan.json'),
         'E17_input_audit_sha256':sha(ROOT/'experiments/E17/input_verification.json'),
         'query_candidate_text_full_qrels':'Every development record verified against source corpora/qrels',
         'selection':'Lowest SHA256(1801:dataset:query_id), 32 per dataset before timings',
         'timing_query_ids':[[r['dataset'],r['query_id']] for r in selected],
         'reserve_query_ids':[[r['dataset'],r['query_id']] for r in reserved],
         'timing_length_distributions':distributions,
         'pretrained_training_overlap':'Unknown; no leaderboard or zero-shot claim'})


if __name__=='__main__':
    main()
