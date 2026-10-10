"""Training distribution and query-dependence checks using validation only."""
import collections
import hashlib
import heapq
import json
import re
from pathlib import Path
import numpy as np
import torch
from transformers import BertTokenizer
from scripts.model import ScratchConfig, ScratchReranker
from experiments.E15.metrics import metrics
def sha256(path):
    with path.open("rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def features(text):
    words = re.findall(r'\w+', text.casefold())
    letters = [c for c in text if c.isalpha()]
    return {'characters': len(text), 'words': len(words),
            'uppercase_fraction': sum(c.isupper() for c in letters)/max(1,len(letters)),
            'all_lowercase': float(text == text.lower()),
            'starts_uppercase': float(bool(text) and text.lstrip()[:1].isupper())}

def main():
    source = ROOT/'data/processed/clean_pairs.jsonl'
    counters = {str(y): collections.Counter() for y in (0,1)}
    train_ids = set(); validation_ids = set(); examples = {str(y): [] for y in (0,1)}
    for line in source.open():
        r = json.loads(line)
        if r['split'] == 'train':
            train_ids.add(r['query_id']); bucket = counters[str(r['label'])]
            bucket['pairs'] += 1
            bucket.update(features(r['document']))
            if len(examples[str(r['label'])]) < 8: examples[str(r['label'])].append(r)
        elif r['split'] == 'validation': validation_ids.add(r['query_id'])
    if train_ids & validation_ids: raise RuntimeError('Query leakage')
    chosen = set(heapq.nsmallest(512, validation_ids)); rows = []
    for line in (ROOT/'data/processed/eval_validation_candidates.jsonl').open():
        r=json.loads(line)
        if r['query_id'] in chosen and r['positive_count'] and r['negative_count']: rows.append(r)
    rows.sort(key=lambda r:r['query_id'])
    torch.set_num_threads(4); torch.backends.mha.set_fastpath_enabled(False)
    model=ScratchReranker(ScratchConfig.from_file(ROOT/'experiments/E1/four_layer/config.json')).eval()
    checkpoint=ROOT/'runs/full_train/epoch_0003.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model_state_dict'])
    tokenizer=BertTokenizer.from_pretrained(ROOT/'artifacts/short_epoch_1024/tokenizer',local_files_only=True)
    audits=[]; totals={name:[] for name in ['normal','rotated_query','empty_query','lowercase_documents','lowercase_heuristic','shorter_document_heuristic']}
    with torch.inference_mode():
        for index,r in enumerate(rows):
            docs=[c['text'] for c in r['candidates']]; ids=[c['document_id'] for c in r['candidates']]
            qrels={c['document_id']:c['label'] for c in r['candidates']}; scores={}
            for name,query,documents in [('normal',r['query'],docs),('rotated_query',rows[(index+1)%len(rows)]['query'],docs),('empty_query','',docs),('lowercase_documents',r['query'],[d.lower() for d in docs])]:
                encoded=tokenizer([query]*len(docs),documents,padding=True,truncation='longest_first',max_length=128,return_tensors='pt')
                scores[name]=model(**encoded).tolist()
            scores['lowercase_heuristic']=[features(d)['all_lowercase'] for d in docs]
            scores['shorter_document_heuristic']=[-len(d) for d in docs]
            values={}
            for name,ss in scores.items():
                ranked=sorted(ids,key=lambda d:(-ss[ids.index(d)],d)); values[name]=metrics(ranked,qrels)
                totals[name].append(values[name])
            audits.append({'query_id':r['query_id'],'query':r['query'],'candidates':r['candidates'],'scores':scores,'metrics':values})
            if (index+1)%100==0: print('audited',index+1,len(rows),flush=True)
    result={'status':'complete','scope':'All train-pair text features; deterministic lowest-512-ID validation query sample with both labels. No test rankings used.',
        'clean_pairs_sha256':sha256(source),'checkpoint_sha256':sha256(checkpoint),'code_sha256':sha256(Path(__file__)),
        'train_queries':len(train_ids),'validation_queries':len(validation_ids),'sample_queries':len(rows),
        'train_label_features':{y:{k:v/c['pairs'] for k,v in c.items() if k!='pairs'}|{'pairs':c['pairs']} for y,c in counters.items()},
        'validation_ablation_metrics':{name:{k:float(np.mean([r[k] for r in vs])) for k in vs[0]} for name,vs in totals.items()},
        'examples':examples,'limitations':['Sample ablations diagnose query dependence, not full retrieval quality.','Heuristics are fixed diagnostics, not trained replacements.']}
    (HERE/'training_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    (HERE/'validation_examples.json').write_text(json.dumps(audits,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='examples'},indent=2))

if __name__=='__main__': main()
