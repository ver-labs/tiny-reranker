"""Summarize matched direct and serving measurements without mixing timing scopes."""
import json
import statistics
from .backend import HERE, write_json


def main():
    quality = json.loads((HERE/'quality.json').read_text())
    direct = {n: json.loads((HERE/f'{n}_benchmark.json').read_text()) for n in ('pytorch','mlx')}
    assert direct['mlx']['sample_sha256'] == direct['pytorch']['sample_sha256']
    assert direct['mlx']['checkpoint_sha256'] == direct['pytorch']['checkpoint_sha256'] == quality['checkpoint_sha256']
    rows = []
    for pt, ml in zip(direct['pytorch']['results'], direct['mlx']['results']):
        assert pt['pair_batch_size'] == ml['pair_batch_size']
        row = {'pair_batch_size': pt['pair_batch_size']}
        for mode in ('forward','scoring'):
            p = pt[mode]['median_pairs_per_second']
            m = ml[mode]['median_pairs_per_second']
            row[mode] = {'pytorch_pairs_per_second': p, 'mlx_pairs_per_second': m,
                         'mlx_speedup': m/p,
                         'pytorch_batch_median_ms': pt[mode]['median_ms'],
                         'mlx_batch_median_ms': ml[mode]['median_ms']}
        rows.append(row)
    serving = json.loads((HERE/'serving_benchmark.json').read_text())
    requests = []
    for concurrency in ('1','4','8'):
        row = {'concurrent_requests': int(concurrency)}
        for name in ('pytorch','mlx'):
            repeats = [r for round in serving['rounds'] if round['backend']==name
                       for r in round['results'][concurrency]]
            row[name] = {key: statistics.median([r[key] for r in repeats])
                         for key in ('requests_per_second','median_ms','p95_ms',
                                     'median_queue_ms','median_processing_ms')}
            row[name]['successful_requests'] = sum(r['successful_requests'] for r in repeats)
            row[name]['failures'] = sum(r['failures'] for r in repeats)
        row['mlx_throughput_speedup'] = row['mlx']['requests_per_second']/row['pytorch']['requests_per_second']
        requests.append(row)
    result = {'selected_checkpoint': 'experiments/E1/four_layer/head_only.pt',
              'execution': {'hardware': 'Apple M5 GPU; same Mac for both backends',
                            'pytorch': 'GPU inference via MPS/Metal',
                            'mlx': 'GPU inference via MLX/Metal',
                            'shared_cpu_work': 'tokenization and HTTP handling'},
              'precision': 'FP32; MLX_ENABLE_TF32=0', 'quality': quality,
              'direct': rows, 'serving': requests,
              'limitations': ['Constructed MS MARCO validation, not leaderboard results.',
                'Direct: 128 queries/256 pairs, five timed repetitions per size, one process per backend.',
                'Serving: 128 queries with two documents each, five repeats in each of two rounds with reversed backend order.',
                'One worker per server; concurrent requests queue; no cross-request batching.',
                'Memory counters measure different framework allocators and are not comparable process peaks.',
                'No test split evaluation; results are local workload measurements, not a general backend ranking.']}
    write_json(HERE/'comparison.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
