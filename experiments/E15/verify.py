"""Audit all downloaded model scores against the exact saved candidates/qrels."""
import gzip
import json
import math
from pathlib import Path
from .prepare import ROOT, HERE, sha256, load
from .metrics import metrics, assert_trec_parity


def rows_for_dataset(name, report):
    manifest = report['datasets'][name]['manifest']
    if name == 'constructed_test':
        path = ROOT / 'data/processed/eval_test_candidates.jsonl'
        if sha256(path) != manifest['candidate_sha256']: raise RuntimeError('Test file changed')
        with path.open() as f: raw = [json.loads(line) for line in f]
        return [{'query_id': r['query_id'], 'document_ids': [c['document_id'] for c in r['candidates']],
                 'qrels': {c['document_id']: c['label'] for c in r['candidates']}} for r in raw]
    _, _, qrels = load(name)
    path = ROOT / 'data/beir' / name / 'bm25_top100.jsonl'
    if sha256(path) != manifest['candidate_sha256']: raise RuntimeError('BM25 candidates changed')
    with path.open() as f: raw = [json.loads(line) for line in f]
    return [{'query_id': r['query_id'], 'document_ids': r['document_ids'],
             'qrels': qrels[r['query_id']]} for r in raw]


def main():
    report = json.loads((HERE / 'quality.json').read_text())
    if report['status'] != 'complete': raise RuntimeError('Incomplete GPU evaluation')
    for relative, expected in report['code_sha256'].items():
        if sha256(ROOT / relative) != expected: raise RuntimeError(f'Measured source changed: {relative}')
    verified = {}
    for name, result in report['datasets'].items():
        rows = rows_for_dataset(name, report)
        verified[name] = {}
        for path, summary in result['paths'].items():
            artifact = HERE / summary['rankings_file']
            if sha256(artifact) != summary['rankings_sha256']: raise RuntimeError('Audit hash mismatch')
            count = pairs = 0; totals = {key: 0. for key in metrics([], rows[0]['qrels'])}; runs = {}
            with gzip.open(artifact, 'rt') as f:
                for expected in rows:
                    line = f.readline()
                    if not line: raise RuntimeError('Missing query audit')
                    r = json.loads(line)
                    if r['query_id'] != expected['query_id'] or r['candidate_ids'] != expected['document_ids'] or r['qrels'] != expected['qrels']:
                        raise RuntimeError('Candidate/query/qrel mismatch')
                    ids = r['candidate_ids']; scores = r['scores']
                    if len(scores) != len(ids) or not all(math.isfinite(v) for v in scores): raise RuntimeError('Invalid scores')
                    order = sorted(range(len(ids)), key=lambda i: (-scores[i], ids[i]))
                    ranked = [ids[i] for i in order]
                    if ranked != r['ranking']: raise RuntimeError('Stored ranking mismatch')
                    values = metrics(ranked, expected['qrels'])
                    if values != r['metrics']: raise RuntimeError('Stored per-query metric mismatch')
                    for key, value in values.items(): totals[key] += value
                    runs[r['query_id']] = ranked; count += 1; pairs += len(ids)
                if f.readline(): raise RuntimeError('Unexpected extra audit query')
            actual = {key: value / count for key, value in totals.items()}
            if actual != summary['metrics'] or count != summary['queries'] or pairs != summary['pairs']:
                raise RuntimeError('Recomputed model summary mismatch')
            assert_trec_parity({r['query_id']: r['qrels'] for r in rows}, runs)
            verified[name][path] = {'queries': count, 'pairs': pairs, 'metrics': actual,
                                    'trec_eval_parity_verified': True}
            print(f'Verified {name}/{path}: {count:,} queries', flush=True)
    output = {'status': 'verified', 'quality_sha256': sha256(HERE / 'quality.json'),
              'verifier_sha256': sha256(Path(__file__)), 'datasets': verified}
    (HERE / 'audit_verification.json').write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__': main()
