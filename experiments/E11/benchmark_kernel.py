"""Validate and benchmark epoch-3 inference with the fused path on and off."""
import argparse
import itertools
import json
from pathlib import Path
import statistics
import time

import numpy as np

from experiments.S1.backend import Backend, CHECKPOINT, ROOT, WEIGHTS, sha256, write_json
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, ranking_metrics
from .kernel import reference, residual_layer_norm, SOURCE
from .mlx_model import FusedMLXReranker
import mlx.core as mx

HERE = Path(__file__).resolve().parent
BATCHES = [1, 2, 8, 16, 64, 128]


def summarize(values):
    return {'median': statistics.median(values), 'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)), 'values': values}


def eval_sync(values):
    mx.eval(values)
    mx.synchronize()


def forward(model, inputs, fused):
    scores = model(**inputs, use_fused=fused)
    eval_sync(scores)
    return np.array(scores)


def probes(backend):
    errors = {}
    fixtures = [(['short query'], ['short document']),
                (['short query', 'long query ' * 90], ['short document', 'long passage ' * 150]),
                (['apple silicon'] * 8, ['document ' * (i + 1) for i in range(8)])]
    for queries, docs in fixtures:
        inputs = backend.prepare(backend.encode(queries, docs))
        standard = backend.model(**inputs, trace=True)
        fused = backend.model(**inputs, trace=True, use_fused=True)
        eval_sync((standard, fused))
        for name in standard:
            a, b = np.array(standard[name]), np.array(fused[name])
            np.testing.assert_allclose(b, a, atol=3e-5, rtol=3e-5)
            errors[name] = max(errors.get(name, 0), float(np.max(np.abs(a - b))))
        # Padding content must remain irrelevant in the custom path as in the reference.
        changed = backend.encode(queries, docs)
        mask = changed['attention_mask'] == 0
        changed['input_ids'][mask] = backend.tokenizer.unk_token_id
        changed['token_type_ids'][mask] = 1
        np.testing.assert_allclose(forward(backend.model, inputs, True),
                                   forward(backend.model, backend.prepare(changed), True), atol=1e-5, rtol=0)
    return {'layer_max_absolute_errors': errors, 'padding_ignored': True,
            'singleton_and_truncation_checked': True, 'atol': 3e-5, 'rtol': 3e-5}


def validate_all(backend, expected):
    count = changed = 0
    max_error = 0.
    metrics = {'standard': [0., 0.], 'fused': [0., 0.]}
    start = time.perf_counter()
    temporary = HERE / 'fused_validation_rankings.jsonl.tmp'
    with CANDIDATES.open() as source, temporary.open('w') as output:
        while True:
            rows = [json.loads(line) for line in itertools.islice(source, 32)]
            if not rows:
                break
            flat = [(row, c) for row in rows for c in row['candidates']]
            scores = {'standard': [], 'fused': []}
            for offset in range(0, len(flat), 64):
                chunk = flat[offset:offset + 64]
                inputs = backend.prepare(backend.encode([r['query'] for r, _ in chunk], [c['text'] for _, c in chunk]))
                # Alternate path order for numerical validation; these are not timing samples.
                for mode in (('standard', 'fused') if count % 64 == 0 else ('fused', 'standard')):
                    scores[mode].extend(forward(backend.model, inputs, mode == 'fused').tolist())
            np.testing.assert_allclose(scores['fused'], scores['standard'], atol=1e-5, rtol=1e-4)
            max_error = max(max_error, float(np.max(np.abs(np.array(scores['fused']) - scores['standard']))))
            offset = 0
            for row in rows:
                rankings = {}
                for mode in scores:
                    values = scores[mode][offset:offset + len(row['candidates'])]
                    ranking = sorted([{'document_id': c['document_id'], 'label': c['label'], 'score': s}
                                      for c, s in zip(row['candidates'], values)], key=lambda x: (-x['score'], x['document_id']))
                    assert all(np.isfinite(x['score']) for x in ranking)
                    rankings[mode] = ranking
                    mrr, ndcg = ranking_metrics([x['label'] for x in ranking])
                    metrics[mode][0] += mrr; metrics[mode][1] += ndcg
                changed += [x['document_id'] for x in rankings['standard']] != [x['document_id'] for x in rankings['fused']]
                output.write(json.dumps({'query_id': row['query_id'], 'standard': rankings['standard'], 'fused': rankings['fused']}) + '\n')
                offset += len(row['candidates']); count += 1
            if count % 4096 == 0:
                print(f'Fused parity: {count:,}/{expected["evaluable_queries"]:,} validation queries', flush=True)
                write_json(HERE / 'kernel_progress.json', {'phase': 'validation', 'queries_scored': count})
    assert count == expected['evaluable_queries']
    assert abs(metrics['standard'][1] - metrics['fused'][1]) / count <= 1e-4
    path = HERE / 'fused_validation_rankings.jsonl'
    temporary.replace(path)
    result = {'queries': count, 'pairs': expected['candidates'], 'changed_rankings': changed,
              'max_score_absolute_error': max_error, 'pair_batch_limit': 64,
              'seconds': time.perf_counter() - start, 'rankings_sha256': sha256(path),
              **{mode: {'mrr_at_10': values[0] / count, 'ndcg_at_10': values[1] / count} for mode, values in metrics.items()}}
    write_json(HERE / 'kernel_quality.json', result)
    print('Full validation:', result, flush=True)
    return result


def microbench(backend, flat, repeats):
    results = []
    for size in BATCHES:
        pairs = flat[:size]
        inputs = backend.prepare(backend.encode([q for q, _ in pairs], [d for _, d in pairs]))
        model = backend.model
        ids, types = inputs['input_ids'], inputs['token_type_ids']
        residual = model.embedding_norm(model.token_embeddings(ids) +
                   model.position_embeddings(mx.arange(ids.shape[1]))[None] + model.segment_embeddings(types))
        layer = model.encoder_layers[0]
        branch = layer.self_attn(layer.norm1(residual), (inputs['attention_mask'] != 0)[:, None, None, :])
        eval_sync((residual, branch))
        args = (residual, branch, layer.norm2.weight, layer.norm2.bias, layer.norm2.eps)
        functions = {'standard': lambda: reference(*args),
                     'fused': lambda: residual_layer_norm(*args, fallback=False)}
        for function in functions.values():
            for _ in range(10): eval_sync(function())
        timings = {'standard': [], 'fused': []}
        for repeat in range(repeats):
            for mode in (('standard', 'fused') if repeat % 2 == 0 else ('fused', 'standard')):
                mx.synchronize(); begin = time.perf_counter()
                eval_sync(functions[mode]())
                timings[mode].append((time.perf_counter() - begin) * 1000)
        row = {'batch_size': size, 'sequence_length': ids.shape[1],
               **{mode: summarize(values) for mode, values in timings.items()}}
        row['speedup'] = row['standard']['median'] / row['fused']['median']
        results.append(row)
        print(f'Operation batch={size}: {row["speedup"]:.3f}x', flush=True)
    return results


def model_bench(backend, flat, repeats_per_round):
    raw = []
    for round_index in range(2):
        for size in (BATCHES if round_index == 0 else list(reversed(BATCHES))):
            chunks = [flat[offset:offset + size] for offset in range(0, len(flat), size)]
            resident = [backend.prepare(backend.encode([q for q, _ in chunk], [d for _, d in chunk])) for chunk in chunks]
            eval_sync(resident)
            for fused in (False, True):
                for inputs in resident: forward(backend.model, inputs, fused)
                for _ in range(10): forward(backend.model, resident[0], fused)
            for repeat in range(repeats_per_round):
                modes = ('standard', 'fused') if (repeat + round_index) % 2 == 0 else ('fused', 'standard')
                for mode in modes:
                    fused = mode == 'fused'
                    scopes = ('forward', 'scoring') if repeat % 2 == 0 else ('scoring', 'forward')
                    for scope in scopes:
                        mx.synchronize(); begin = time.perf_counter()
                        call_times = []
                        for chunk, inputs in zip(chunks, resident):
                            call_begin = time.perf_counter()
                            if scope == 'scoring':
                                inputs = backend.prepare(backend.encode([q for q, _ in chunk], [d for _, d in chunk]))
                            forward(backend.model, inputs, fused)
                            call_times.append((time.perf_counter() - call_begin) * 1000)
                        mx.synchronize(); elapsed = time.perf_counter() - begin
                        raw.append({'round': round_index + 1, 'repeat': repeat + 1, 'batch_size': size,
                                    'mode': mode, 'scope': scope, 'pairs': len(flat), 'seconds': elapsed,
                                    'pairs_per_second': len(flat) / elapsed,
                                    'median_batch_ms': statistics.median(call_times)})
            print(f'Model benchmark round={round_index + 1} batch={size} complete', flush=True)
            write_json(HERE / 'kernel_progress.json', {'phase': 'benchmark', 'round': round_index + 1, 'batch_size': size})
            del resident
            mx.clear_cache()
    rows = []
    for size in BATCHES:
        row = {'batch_size': size}
        for scope in ('forward', 'scoring'):
            row[scope] = {}
            for mode in ('standard', 'fused'):
                points = [r for r in raw if r['batch_size'] == size and r['mode'] == mode and r['scope'] == scope]
                row[scope][mode] = {'pairs_per_second': summarize([r['pairs_per_second'] for r in points]),
                                     'median_batch_ms': summarize([r['median_batch_ms'] for r in points])}
            row[scope]['speedup'] = row[scope]['fused']['pairs_per_second']['median'] / row[scope]['standard']['pairs_per_second']['median']
        rows.append(row)
    return rows, raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats-per-round', type=int, default=10)
    args = parser.parse_args()
    if args.repeats_per_round < 3: parser.error('Use at least three repeats per round')
    expected = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']
    assert sha256(CANDIDATES) == expected['sha256']
    with CANDIDATES.open() as source:
        rows = [json.loads(line) for line in itertools.islice(source, 128)]
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    backend = Backend('mlx')
    backend.model = FusedMLXReranker(backend.config)
    backend.model.load_weights(str(WEIGHTS), strict=True)
    backend.model.eval()
    eval_sync(backend.model.parameters())
    # Measure actual first use separately, including library/JIT work when needed.
    initial_args = (mx.ones((1, 128)), mx.zeros((1, 128)), mx.ones(128), mx.zeros(128))
    eval_sync(initial_args)
    begin = time.perf_counter(); eval_sync(residual_layer_norm(*initial_args, fallback=False))
    first_use_ms = (time.perf_counter() - begin) * 1000
    result = {'checkpoint': str(CHECKPOINT.relative_to(ROOT)), 'checkpoint_sha256': sha256(CHECKPOINT),
              'dataset_split': 'validation', 'candidate_sha256': expected['sha256'],
              'precision': 'FP32; MLX_ENABLE_TF32=0', 'device': mx.device_info(),
              'sample_queries': 128, 'sample_pairs': len(flat),
              'sample_sha256': __import__('hashlib').sha256(json.dumps(flat).encode()).hexdigest(),
              'batch_grid': BATCHES, 'repeats_per_point': 2 * args.repeats_per_round,
              'first_kernel_use_ms': first_use_ms,
              'first_use_scope': 'first invocation including library/JIT setup when needed, not pure compiler time',
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                              [Path(__file__), SOURCE, HERE / 'kernel.py', HERE / 'mlx_model.py']},
              'policy': 'Inference-only. Ordinary path stays default regardless of local speed results.',
              'limitations': ['Synchronized wall times include Python dispatch and CPU-readable scores.',
                             'Direct inference only; no HTTP coordinator speedup is measured.',
                             'Timing uses first 128 fixed validation queries; quality uses all validation queries.',
                             'No test data accessed. Shapes/dtypes outside the kernel contract use ordinary MLX.',
                             'Warmups are excluded; both paths share weights and inputs; path order alternates.']}
    result['trained_model_probes'] = probes(backend)
    result['operation_microbench'] = microbench(backend, flat, 30)
    result['full_validation_quality'] = validate_all(backend, expected)
    result['model_benchmark'], result['raw_model_repeats'] = model_bench(backend, flat, args.repeats_per_round)
    write_json(HERE / 'kernel_benchmark.json', result)
    write_json(HERE / 'kernel_progress.json', {'phase': 'complete'})
    print('Kernel benchmark saved.', flush=True)


if __name__ == '__main__': main()
