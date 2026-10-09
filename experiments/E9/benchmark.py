"""Matched direct inference benchmark: pretokenized forward and complete pair scoring."""
import argparse
import importlib.metadata
import itertools
import json
import platform
import statistics
import time

from scripts.evaluate_initial import CANDIDATES
from .backend import Backend, HERE, sha256, write_json


def stats(values):
    return {'median_ms': statistics.median(values)*1000,
            'p95_ms': sorted(values)[min(len(values)-1, int(.95 * len(values))) ]*1000}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=['mlx', 'pytorch'], required=True)
    args = parser.parse_args()
    backend = Backend(args.backend)
    with CANDIDATES.open() as source:
        rows = [json.loads(s) for s in itertools.islice(source, 128)]
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    sample_hash = __import__('hashlib').sha256(json.dumps(flat).encode()).hexdigest()
    results = []
    for size in (1, 4, 8, 16, 64):
        batches = [flat[i:i+size] for i in range(0, len(flat), size)]
        encoded = [backend.prepare(backend.encode([q for q,d in b], [d for q,d in b])) for b in batches]
        # Warm every actual shape, then ten extra calls at the nominal batch size.
        for inputs in encoded:
            backend.forward(inputs)
        for _ in range(10):
            backend.forward(encoded[0])
        forward_times, scoring_times = [], []
        forward_calls, scoring_calls = [], []
        memory_samples = []
        for repeat in range(5):
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
                memory_samples.append(backend.memory())
        results.append({'pair_batch_size': size, 'pairs': len(flat),
            'forward': {**stats(forward_calls), 'repeat_seconds': forward_times,
                        'median_pairs_per_second': len(flat)/statistics.median(forward_times)},
            'scoring': {**stats(scoring_calls), 'repeat_seconds': scoring_times,
                        'median_pairs_per_second': len(flat)/statistics.median(scoring_times)},
            'sampled_memory': {k: max(s[k] for s in memory_samples) for k in memory_samples[0]}})
        print(args.backend, size, results[-1]['scoring']['median_pairs_per_second'], flush=True)
    write_json(HERE / f'{args.backend}_benchmark.json', {
        'backend': args.backend, 'loading_seconds': backend.loading_seconds,
        'checkpoint_sha256': sha256(HERE.parent / 'E1/four_layer/head_only.pt'),
        'sample_sha256': sample_hash, 'queries': len(rows), 'repeats': 5, 'warmups': 10,
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
