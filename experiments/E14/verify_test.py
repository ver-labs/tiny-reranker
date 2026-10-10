"""Recompute full-test quality/parity from retained scores and original candidates."""
import gzip
import hashlib
import itertools
import json
import math
from pathlib import Path

from scripts.evaluate_initial import ranking_metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha256(path):
    with path.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    result = json.loads((HERE / 'test_quality.json').read_text())
    plan = result['plan']; audit = HERE / result['audit_file']
    if result['status'] != 'complete': raise RuntimeError('Evaluation incomplete')
    if sha256(audit) != result['audit_sha256']: raise RuntimeError('Audit hash mismatch')
    if sha256(HERE / 'test_plan.json') != result['plan_sha256']: raise RuntimeError('Plan hash mismatch')
    if json.loads((HERE / 'test_plan.json').read_text()) != plan: raise RuntimeError('Plan contents mismatch')
    for relative, expected in plan['code_sha256'].items():
        if sha256(ROOT / relative) != expected: raise RuntimeError(f'Measured source changed: {relative}')
    candidates = ROOT / plan['test_manifest']['candidate_file']
    if sha256(candidates) != plan['candidate_sha256']: raise RuntimeError('Candidate hash mismatch')
    modes = ('original', 'cuda', 'triton')
    totals = {m: {'mrr': 0., 'ndcg': 0., 'changed': 0, 'max_error': 0., 'outside_tolerance': 0} for m in modes}
    count = pairs = 0; seen = set()
    with gzip.open(audit, 'rt') as source, candidates.open() as original:
        for saved, raw in itertools.zip_longest(source, original):
            if saved is None or raw is None: raise RuntimeError('Missing query record')
            row, candidate_row = json.loads(saved), json.loads(raw)
            qid = row['query_id']
            if qid != candidate_row['query_id'] or qid in seen or int(qid[:16], 16) % 10 != 9:
                raise RuntimeError('Query mismatch, duplicate or non-test query')
            seen.add(qid)
            ids, labels = row['candidate_ids'], row['labels']
            if ids != [c['document_id'] for c in candidate_row['candidates']] or labels != [c['label'] for c in candidate_row['candidates']]:
                raise RuntimeError('Candidate order or label mismatch')
            for mode in modes:
                values = row['scores'][mode]
                if len(values) != len(ids) or not all(math.isfinite(v) for v in values):
                    raise RuntimeError('Invalid retained scores')
                ordered = sorted(range(len(ids)), key=lambda i: (-values[i], ids[i]))
                ranking = [ids[i] for i in ordered]
                if ranking != row['rankings'][mode]: raise RuntimeError('Stored ranking mismatch')
                mrr, ndcg = ranking_metrics([labels[i] for i in ordered])
                if mrr != row['metrics'][mode]['mrr_at_10'] or ndcg != row['metrics'][mode]['ndcg_at_10']:
                    raise RuntimeError('Stored query metric mismatch')
                totals[mode]['mrr'] += mrr; totals[mode]['ndcg'] += ndcg
                totals[mode]['changed'] += ranking != row['rankings']['original']
                for actual, reference in zip(values, row['scores']['original']):
                    error = abs(actual - reference)
                    totals[mode]['max_error'] = max(totals[mode]['max_error'], error)
                    totals[mode]['outside_tolerance'] += error > 1e-5 + 1e-4 * abs(reference)
            count += 1; pairs += len(ids)
    if count != result['queries'] or pairs != result['pairs']: raise RuntimeError('Count mismatch')
    for mode, t in totals.items():
        expected = result['metrics'][mode]
        if (t['mrr'] / count != expected['mrr_at_10'] or t['ndcg'] / count != expected['ndcg_at_10']
            or t['changed'] != expected['changed_rankings_vs_pytorch']
            or t['max_error'] != expected['max_score_error_vs_pytorch']
            or t['outside_tolerance'] != expected['pairs_outside_score_tolerance']):
            raise RuntimeError(f'Recomputed summary mismatch: {mode}')
    report = {'status': 'verified', 'queries': count, 'pairs': pairs,
              'audit_sha256': sha256(audit), 'test_quality_sha256': sha256(HERE / 'test_quality.json'),
              'verifier_sha256': sha256(Path(__file__)), 'recomputed_totals': totals}
    (HERE / 'test_audit_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
