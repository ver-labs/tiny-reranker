"""Matched direct inference benchmark: pretokenized forward and complete pair scoring."""
import argparse
import importlib.metadata
import itertools
import json
import platform
import statistics
import time

from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST
from .backend import Backend, HERE, CHECKPOINT, sha256, write_json


def stats(values):
    return {'median_ms': statistics.median(values)*1000,
            'p95_ms': sorted(values)[min(len(values)-1, int(.95 * len(values))) ]*1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['mlx', 'pytorch'], required=True)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1,4,8,16,64])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output-name')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--split', choices=['validation','test'], default='validation')
    args = parser.parse_args()
    if args.repeats < 1 or any(s < 1 for s in args.batch_sizes):
        parser.error('sizes and repeats must be positive')
    backend = Backend(args.backend)
    candidates = CANDIDATES if args.split=='validation' else CANDIDATES.with_name('eval_test_candidates.jsonl')
    expected = json.loads(CANDIDATE_MANIFEST.read_text())['splits'][args.split]
    if sha256(candidates)!=expected['sha256']:
        raise RuntimeError('Candidate data differs from its manifest')
    with candidates.open() as source:
        rows = [json.loads(s) for s in itertools.islice(source, 128)]
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    sample_hash = __import__('hashlib').sha256(json.dumps(flat).encode()).hexdigest()
    results = []
    for size in (list(reversed(args.batch_sizes)) if args.reverse else args.batch_sizes):
        batches = [flat[i:i+size] for i in range(0, len(flat), size)]
        encoded = [backend.prepare(backend.encode([q for q,d in b], [d for q,d in b])) for b in batches]
        # Warm every actual shape, then ten extra calls at the nominal batch size.
        for inputs in encoded:
            backend.forward(inputs)
        for _ in range(10):
            backend.forward(encoded[0])
        forward_times, scoring_times = [], []
        forward_calls, scoring_calls = [], []
        forward_medians, scoring_medians = [], []
        memory_samples = []
        for repeat in range(args.repeats):
            for mode in (('forward', 'scoring') if repeat % 2 == 0 else ('scoring', 'forward')):
                backend.synchronize()
                start = time.perf_counter()
                calls = []
                for b, inputs in zip(batches, encoded):
                    begin = time.perf_counter()
                    if mode == 'forward':
                        backend.forward(inputs)
                    else:
                        backend.score([q for q,d in b], [d for q,d in b])
                    calls.append(time.perf_counter()-begin)
                backend.synchronize()
                elapsed = time.perf_counter()-start
                (forward_times if mode == 'forward' else scoring_times).append(elapsed)
                (forward_calls if mode == 'forward' else scoring_calls).extend(calls)
                (forward_medians if mode == 'forward' else scoring_medians).append(stats(calls)['median_ms'])
                memory_samples.append(backend.memory())
        results.append({'pair_batch_size': size, 'pairs': len(flat),
            'forward': {**stats(forward_calls), 'repeat_seconds': forward_times,
                        'repeat_median_ms': forward_medians,
                        'median_pairs_per_second': len(flat)/statistics.median(forward_times)},
            'scoring': {**stats(scoring_calls), 'repeat_seconds': scoring_times,
                        'repeat_median_ms': scoring_medians,
                        'median_pairs_per_second': len(flat)/statistics.median(scoring_times)},
            'sampled_memory': {k: max(s[k] for s in memory_samples) for k in memory_samples[0]}})
        print(args.backend, size, results[-1]['scoring']['median_pairs_per_second'], flush=True)
    write_json(HERE / (args.output_name or f'{args.backend}_benchmark.json'), {
        'backend': args.backend, 'loading_seconds': backend.loading_seconds,
        'dataset_split': args.split, 'candidate_sha256': expected['sha256'],
        'checkpoint_sha256': sha256(CHECKPOINT),
        'sample_sha256': sample_hash, 'queries': len(rows), 'repeats': args.repeats, 'warmups': 10,
        'precision': 'float32', 'max_length': 128, 'python': platform.python_version(),
        'platform': platform.platform(),
        'versions': {p: importlib.metadata.version(p) for p in ['torch', 'mlx', 'transformers']},
        'forward_scope': 'pretokenized resident device inputs through CPU-readable scores',
        'scoring_scope': 'tokenization, input conversion/transfer, forward, CPU-readable scores',
        'latency_unit': 'pair batch; not an individual query',
        'memory_scope': 'sampled current allocation after scoring; includes retained input batches; framework counters differ, not process memory',
        'results': results})


if __name__ == '__main__':
    main()
