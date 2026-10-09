"""Real loopback HTTP benchmark, sequential isolated servers and repeated concurrency."""
import argparse
import asyncio
import itertools
import json
import socket
import statistics
import subprocess
import sys
import time

import httpx

from scripts.evaluate_initial import CANDIDATES
from .backend import HERE, write_json
from .benchmark import stats


async def run_requests(client, payloads, concurrency, expected):
    semaphore = asyncio.Semaphore(concurrency)
    times, queue_times, processing_times = [], [], []
    async def one(payload):
        async with semaphore:
            start = time.perf_counter()
            response = await client.post('/rerank', json=payload)
            times.append(time.perf_counter()-start)
            response.raise_for_status()
            data = response.json()
            assert data['request_id'] == payload['request_id']
            assert [d['document_id'] for d in data['ranking']] == expected[payload['request_id']]
            queue_times.append(data['queue_ms'])
            processing_times.append(data['processing_ms'])
    start = time.perf_counter()
    await asyncio.gather(*(one(p) for p in payloads))
    elapsed = time.perf_counter()-start
    return {'seconds': elapsed, 'requests_per_second': len(payloads)/elapsed,
            **stats(times), 'median_queue_ms': statistics.median(queue_times),
            'median_processing_ms': statistics.median(processing_times),
            'successful_requests': len(payloads), 'failures': 0}


async def benchmark(name, payloads, expected, batch_size):
    # Choose an available loopback port, rather than assuming a fixed one is free.
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        port = s.getsockname()[1]
    log_path = HERE / f'{name}_server.log'
    with log_path.open('w') as log:
        proc = subprocess.Popen([sys.executable, '-m', 'experiments.E9.serve',
            '--backend', name, '--port', str(port), '--pair-batch-size', str(batch_size)],
            stdout=log, stderr=subprocess.STDOUT)
        try:
            async with httpx.AsyncClient(base_url=f'http://127.0.0.1:{port}', timeout=30,
                    trust_env=False, limits=httpx.Limits(max_connections=16)) as client:
                ready_start = time.perf_counter()
                for _ in range(300):
                    if proc.poll() is not None:
                        raise RuntimeError(f'{name} server failed: {log_path.read_text()}')
                    try:
                        health = await client.get('/health')
                        if health.status_code == 200 and health.json()['ready']:
                            break
                    except httpx.ConnectError:
                        pass
                    await asyncio.sleep(.1)
                else:
                    raise RuntimeError('Server did not become ready')
                ready_seconds = time.perf_counter()-ready_start
                # Semantic HTTP checks on the real server.
                invalid = await client.post('/rerank', json={'request_id':'bad','query':'q','documents':[]})
                assert invalid.status_code == 422
                top = dict(payloads[0], top_k=1)
                response = await client.post('/rerank', json=top)
                response.raise_for_status()
                assert len(response.json()['ranking']) == 1
                assert response.json()['ranking'][0]['document_id'] == expected[top['request_id']][0]
                for p in payloads[:16]:
                    response = await client.post('/rerank', json=p)
                    response.raise_for_status()
                results = {str(c): [] for c in (1, 4, 8)}
                for repeat in range(5):
                    for c in ((1,4,8) if repeat % 2 == 0 else (8,4,1)):
                        result = await run_requests(client, payloads, c, expected)
                        results[str(c)].append(result)
                        print(name, 'repeat', repeat+1, 'concurrency', c,
                              round(result['requests_per_second'], 1), 'requests/s', flush=True)
                return {'backend': name, 'server_ready_seconds': ready_seconds,
                        'pair_batch_size': batch_size, 'warmup_requests': 16,
                        'requests_per_repeat': len(payloads), 'repeats': 5,
                        'queue_capacity': 32, 'model_workers': 1,
                        'scope': 'loopback HTTP with persistent client connections; tokenization, queue, model, sorting, JSON included',
                        'concurrency_method': 'closed loop, bounded in-flight requests; latency begins after acquiring client slot',
                        'top_k_and_invalid_request_checks': True, 'results': results}
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair-batch-size', type=int, default=8)
    args = parser.parse_args()
    with CANDIDATES.open() as source:
        rows = [json.loads(line) for line in itertools.islice(source, 128)]
    with (HERE.parent / 'E1/four_layer/four_layer_head_only_rankings.jsonl').open() as source:
        references = [json.loads(line) for line in itertools.islice(source, len(rows))]
    expected = {r['query_id']: [d['document_id'] for d in r['ranking']] for r in references}
    payloads = [{'request_id': r['query_id'], 'query': r['query'], 'top_k': len(r['candidates']),
                 'documents': [{'document_id': c['document_id'], 'text': c['text']}
                               for c in r['candidates']]} for r in rows]
    # Alternate backend order over two isolated rounds to reduce order/thermal bias.
    results = []
    for order in (('pytorch', 'mlx'), ('mlx', 'pytorch')):
        for name in order:
            result = await benchmark(name, payloads, expected, args.pair_batch_size)
            results.append(result)
            write_json(HERE / 'serving_benchmark.json', {'rounds': results,
                'sample': 'first 128 fixed validation queries; two candidate documents per query',
                'test_split_touched': False})


if __name__ == '__main__':
    asyncio.run(main())
