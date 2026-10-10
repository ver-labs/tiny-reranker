"""Retrieve fixed BM25 top-100 candidates from complete BEIR corpora."""
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
from .metrics import metrics, assert_trec_parity

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def load(name):
    folder = ROOT / 'data/beir' / name
    expected = json.loads((HERE / 'datasets.json').read_text())[name]
    for relative, digest in expected['file_sha256'].items():
        if sha256(folder / relative) != digest: raise RuntimeError('Dataset hash mismatch')
    with (folder / 'corpus.jsonl').open() as f: corpus = {r['_id']: r for r in map(json.loads, f)}
    with (folder / 'queries.jsonl').open() as f: queries = {r['_id']: r['text'] for r in map(json.loads, f)}
    qrels = {}
    with (folder / 'qrels/test.tsv').open() as f:
        for row in csv.DictReader(f, delimiter='\t'):
            qrels.setdefault(row['query-id'], {})[row['corpus-id']] = int(row['score'])
    if not set(qrels).issubset(queries): raise RuntimeError('Missing test query')
    if any(not set(rels).issubset(corpus) for rels in qrels.values()): raise RuntimeError('Missing qrel document')
    return corpus, queries, qrels


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.casefold())


def main():
    summary = {'retriever': 'rank_bm25.BM25Okapi', 'k1': 1.5, 'b': .75, 'epsilon': .25,
        'preprocessing': 'casefold + ASCII alphanumeric tokens; no stemming or stopword removal',
        'top_k': 100, 'candidate_policy': 'full corpus; no relevance injection; exclude identical query/document IDs',
        'versions': {p: importlib.metadata.version(p) for p in ['rank-bm25', 'numpy', 'pytrec-eval-terrier']},
        'code_sha256': {p.name: sha256(p) for p in [Path(__file__), HERE / 'metrics.py']}, 'datasets': {}}
    for name in ('scifact', 'nfcorpus'):
        corpus, queries, qrels = load(name); doc_ids = sorted(corpus)
        texts = [corpus[doc].get('title', '') + ' ' + corpus[doc]['text'] for doc in doc_ids]
        bm25 = BM25Okapi([tokenize(t) for t in texts], k1=1.5, b=.75, epsilon=.25)
        ids_array = np.asarray(doc_ids); runs = {}; totals = {k: 0. for k in metrics([], next(iter(qrels.values())))}
        no_relevant = 0; path = ROOT / 'data/beir' / name / 'bm25_top100.jsonl'
        with path.open('w') as output:
            for qid in sorted(qrels):
                scores = bm25.get_scores(tokenize(queries[qid]))
                if not np.isfinite(scores).all(): raise RuntimeError('Nonfinite BM25 scores')
                indices = [int(i) for i in np.lexsort((ids_array, -scores)) if doc_ids[int(i)] != qid][:100]
                ids = [doc_ids[i] for i in indices]; runs[qid] = ids
                values = metrics(ids, qrels[qid])
                for key, value in values.items(): totals[key] += value
                no_relevant += not any(qrels[qid].get(doc, 0) > 0 for doc in ids)
                output.write(json.dumps({'query_id': qid, 'document_ids': ids,
                    'bm25_scores': [float(scores[i]) for i in indices]}) + '\n')
        assert_trec_parity(qrels, runs)
        summary['datasets'][name] = {'corpus_documents': len(corpus), 'test_queries': len(qrels),
            'candidate_pairs': sum(map(len, runs.values())), 'candidate_file_sha256': sha256(path),
            'queries_without_relevant_retrieved_candidate': no_relevant,
            'bm25_metrics': {k: v / len(qrels) for k, v in totals.items()}, 'trec_eval_parity_verified': True}
        print(name, summary['datasets'][name], flush=True)
    (HERE / 'retrieval_manifest.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__': main()
