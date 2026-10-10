"""Three rotated full-test scoring passes per path at fixed pair batch 128."""
import gzip
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from scripts.evaluate_initial import ranking_metrics
from .benchmark import ROOT, HERE, environment, load_fixture, save, sha256

MODES = ('original', 'cuda', 'triton')
BATCH_SIZE = 128
REPEATS = 3


def main():
    previous = json.loads((HERE / 'test_quality.json').read_text())
    for relative, expected in previous['plan']['code_sha256'].items():
        if sha256(ROOT / relative) != expected: raise RuntimeError(f'Measured source changed: {relative}')
    candidates = ROOT / previous['plan']['test_manifest']['candidate_file']
    if sha256(candidates) != previous['plan']['candidate_sha256']: raise RuntimeError('Test candidates changed')
    audit = HERE / previous['audit_file']
    if sha256(audit) != previous['audit_sha256']: raise RuntimeError('Score audit changed')
    with candidates.open() as source: rows = [json.loads(line) for line in source]
    with gzip.open(audit, 'rt') as source: original_rankings = [json.loads(line)['rankings']['original'] for line in source]
    flat = [(row['query'], c['text']) for row in rows for c in row['candidates']]
    if len(flat) != previous['pairs'] or len(rows) != previous['queries']: raise RuntimeError('Test count mismatch')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    model, tokenizer, _, checkpoint, _ = load_fixture()
    def prepare(chunk):
        inputs = tokenizer([q for q, _ in chunk], [d for _, d in chunk], padding=True,
            truncation='longest_first', max_length=model.config.max_length,
            return_attention_mask=True, return_token_type_ids=True, return_tensors='pt')
        return {k: v.cuda() for k, v in inputs.items()}
    report = {'environment': environment(), 'checkpoint_sha256': sha256(checkpoint),
        'candidate_sha256': sha256(candidates), 'dataset_split': 'test',
        'queries': len(rows), 'pairs': len(flat), 'pair_batch_size': BATCH_SIZE,
        'repeats': REPEATS, 'precision': 'FP32; TF32 disabled', 'attention_fastpath': False,
        'triton_num_warps': 4, 'max_length': model.config.max_length,
        'scope': 'Warm complete scoring: tokenization + H2D + model + CPU float scores. No HTTP, ranking sort, file reads, model loading or warm-up.',
        'measured_source_sha256': previous['plan']['code_sha256'],
        'benchmark_source_sha256': sha256(Path(__file__)), 'full_test_quality_sha256': sha256(HERE / 'test_quality.json'),
        'rounds': [], 'limitations': ['Three full-dataset passes per path, rotated path order; fixed batch size 128.',
            'Batch-latency percentiles span the full test data, not repeated timings on one query.',
            'Throughput is offline pair scoring, not HTTP request throughput.',
            'No checkpoint or launch configuration tuning from test results.']}
    with torch.inference_mode():
        inputs = prepare(flat[:BATCH_SIZE])
        for mode in MODES:
            for _ in range(10): model(**inputs, kernel_mode=mode, num_warps=4).cpu()
        torch.cuda.synchronize()
        for repeat in range(REPEATS):
            for mode in MODES[repeat:] + MODES[:repeat]:
                timings = []; scores = []; lengths = []; counts = []
                torch.cuda.synchronize(); pass_start = time.perf_counter()
                for start in range(0, len(flat), BATCH_SIZE):
                    chunk = flat[start:start + BATCH_SIZE]
                    torch.cuda.synchronize(); begin = time.perf_counter()
                    inputs = prepare(chunk)
                    values = model(**inputs, kernel_mode=mode, num_warps=4).cpu().tolist()
                    elapsed = time.perf_counter() - begin
                    if len(values) != len(chunk) or not all(np.isfinite(values)): raise RuntimeError('Invalid output')
                    timings.append(elapsed * 1000); scores.extend(values)
                    lengths.append(inputs['input_ids'].shape[1]); counts.append(len(chunk))
                    if (start + len(chunk)) % 16384 == 0 or start + len(chunk) == len(flat):
                        print(f'Full-test latency round={repeat+1} path={mode} pairs={start+len(chunk):,}/{len(flat):,}', flush=True)
                pass_seconds = time.perf_counter() - pass_start
                # Quality auditing deliberately happens outside the timed pass.
                offset = 0; changed = 0; mrr = ndcg = 0.
                for row, reference in zip(rows, original_rankings):
                    n = len(row['candidates'])
                    ranked = sorted(zip(row['candidates'], scores[offset:offset+n]), key=lambda v: (-v[1], v[0]['document_id']))
                    changed += [c['document_id'] for c, _ in ranked] != reference
                    m, d = ranking_metrics([c['label'] for c, _ in ranked]); mrr += m; ndcg += d; offset += n
                if offset != len(flat): raise RuntimeError('Missing scores')
                mrr /= len(rows); ndcg /= len(rows)
                if changed or mrr != previous['metrics']['original']['mrr_at_10'] or ndcg != previous['metrics']['original']['ndcg_at_10']:
                    raise RuntimeError('Full-pass ranking quality changed')
                scoring_seconds = sum(timings) / 1000
                report['rounds'].append({'round': repeat+1, 'mode': mode,
                    'scoring_seconds': scoring_seconds, 'pass_wall_seconds': pass_seconds,
                    'pairs_per_second': len(flat) / scoring_seconds,
                    'batch_p50_ms': float(np.percentile(timings, 50)), 'batch_p95_ms': float(np.percentile(timings, 95)),
                    'batch_p99_ms': float(np.percentile(timings, 99)), 'batch_ms': timings,
                    'batch_pair_counts': counts, 'batch_padded_lengths': lengths,
                    'mrr_at_10': mrr, 'ndcg_at_10': ndcg, 'changed_rankings': changed})
                save(HERE / 'full_test_latency_progress.json', report)
                print(f'Completed round={repeat+1} path={mode}: scoring={scoring_seconds:.3f}s, '
                      f'p50={np.median(timings):.3f}ms, p95={np.percentile(timings,95):.3f}ms', flush=True)
    report['summary'] = {}
    for mode in MODES:
        runs = [r for r in report['rounds'] if r['mode'] == mode]
        report['summary'][mode] = {field: statistics.median(r[field] for r in runs)
            for field in ('scoring_seconds', 'pass_wall_seconds', 'pairs_per_second', 'batch_p50_ms', 'batch_p95_ms', 'batch_p99_ms')}
        report['summary'][mode]['scoring_seconds_range'] = [min(r['scoring_seconds'] for r in runs), max(r['scoring_seconds'] for r in runs)]
    report['status'] = 'complete'
    save(HERE / 'full_test_latency.json', report)
    print(json.dumps(report['summary'], indent=2), flush=True)


if __name__ == '__main__': main()
