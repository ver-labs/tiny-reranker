"""Epoch-3 validation-input profiling; synchronized wall times, not GPU timestamps."""
import argparse
from collections import defaultdict
import cProfile
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import time

import numpy as np

# Import S1 first: preserve its FP32/TF32 policy before loading MLX.
from experiments.S1.backend import Backend, CHECKPOINT, ROOT, sha256, write_json
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST
import mlx.core as mx

HERE = Path(__file__).resolve().parent


def summary(values):
    return {'median_ms': statistics.median(values),
            'q25_ms': float(np.percentile(values, 25)),
            'q75_ms': float(np.percentile(values, 75)), 'repeat_ms': values}


def materialize(value):
    mx.eval(value)
    mx.synchronize()


def operation_fixture(model, inputs):
    """Same equations as S1; save each actual resident input for isolated replay."""
    ops = []

    def stage(name, category, function):
        value = function()
        materialize(value)
        ops.append((name, category, function))
        return value

    ids, types = inputs['input_ids'], inputs['token_type_ids']
    x = stage('embedding_additions', 'embedding', lambda:
              model.token_embeddings(ids) +
              model.position_embeddings(mx.arange(ids.shape[1]))[None] +
              model.segment_embeddings(types))
    x = stage('embedding_norm', 'layer_norm', lambda x=x: model.embedding_norm(x))
    mask = (inputs['attention_mask'] != 0)[:, None, None, :]
    materialize(mask)
    for i, layer in enumerate(model.encoder_layers):
        prefix = f'layer_{i + 1}.'
        residual = x
        norm = stage(prefix + 'norm1', 'layer_norm', lambda x=x, layer=layer: layer.norm1(x))
        batch, length, hidden = norm.shape
        def qkv(norm=norm, layer=layer, batch=batch, length=length, hidden=hidden):
            values = mx.split(layer.self_attn.in_proj(norm), 3, axis=-1)
            return tuple(a.reshape(batch, length, layer.self_attn.heads,
                                   hidden // layer.self_attn.heads).transpose(0, 2, 1, 3)
                         for a in values)
        q, k, v = stage(prefix + 'qkv', 'attention_projections', qkv)
        attention = stage(prefix + 'sdpa', 'attention_core',
                          lambda q=q, k=k, v=v, layer=layer:
                          mx.fast.scaled_dot_product_attention(q, k, v,
                              scale=(hidden // layer.self_attn.heads) ** -.5, mask=mask))
        projected = stage(prefix + 'out_projection', 'attention_projections',
                          lambda attention=attention, layer=layer, batch=batch, length=length, hidden=hidden:
                          layer.self_attn.out_proj(attention.transpose(0, 2, 1, 3).reshape(batch, length, hidden)))
        x = stage(prefix + 'residual1', 'residual_add', lambda residual=residual, projected=projected: residual + projected)
        residual = x
        norm = stage(prefix + 'norm2', 'layer_norm', lambda x=x, layer=layer: layer.norm2(x))
        z = stage(prefix + 'linear1', 'feed_forward_linear', lambda norm=norm, layer=layer: layer.linear1(norm))
        z = stage(prefix + 'gelu', 'gelu', lambda z=z: .5 * z * (1. + mx.erf(z / (2. ** .5))))
        z = stage(prefix + 'linear2', 'feed_forward_linear', lambda z=z, layer=layer: layer.linear2(z))
        x = stage(prefix + 'residual2', 'residual_add', lambda residual=residual, z=z: residual + z)
    x = stage('final_norm', 'layer_norm', lambda x=x: model.final_norm(x))
    scores = stage('scoring_head', 'scoring_head', lambda x=x: model.scoring_head(x[:, 0])[:, 0])
    return ops, np.array(scores)


def isolated_operations(ops, repeats):
    timings = defaultdict(list)
    for _, _, function in ops:
        for _ in range(3):
            materialize(function())
    # Reverse the operation order on alternate rounds to reduce ordering bias.
    for repeat in range(repeats):
        for name, _, function in (ops if repeat % 2 == 0 else list(reversed(ops))):
            mx.synchronize()
            begin = time.perf_counter()
            materialize(function())
            timings[name].append((time.perf_counter() - begin) * 1000)
    details = [{'operation': name, 'category': category, **summary(timings[name])}
               for name, category, _ in ops]
    categories = defaultdict(list)
    for entry in details:
        categories[entry['category']].append(entry['median_ms'])
    total = sum(entry['median_ms'] for entry in details)
    grouped = [{'category': name, 'isolated_sum_ms': sum(values),
                'share_of_isolated_sum_percent': sum(values) / total * 100}
               for name, values in categories.items()]
    return details, sorted(grouped, key=lambda x: -x['isolated_sum_ms'])


def phases(backend, queries, documents, repeats):
    samples = defaultdict(list)
    inputs = backend.prepare(backend.encode(queries, documents))
    materialize(inputs)
    for _ in range(10):
        backend.score(queries, documents)
        backend.forward(inputs)
    # Interleave modes instead of comparing separate time windows on a laptop.
    for repeat in range(repeats):
        modes = ['instrumented', 'normal_scoring', 'resident_forward']
        rotation = repeat % len(modes)
        for mode in modes[rotation:] + modes[:rotation]:
            mx.synchronize()
            start = time.perf_counter()
            if mode == 'normal_scoring':
                backend.score(queries, documents)
                mx.synchronize()
                samples[mode].append((time.perf_counter() - start) * 1000)
            elif mode == 'resident_forward':
                backend.forward(inputs)
                mx.synchronize()
                samples[mode].append((time.perf_counter() - start) * 1000)
            else:
                encoded = backend.encode(queries, documents)
                encoded_at = time.perf_counter()
                prepared = backend.prepare(encoded)
                materialize(prepared)
                prepared_at = time.perf_counter()
                scores = backend.model(**prepared)
                materialize(scores)
                inferred_at = time.perf_counter()
                values = np.array(scores).tolist()
                finished_at = time.perf_counter()
                assert len(values) == len(queries) and np.isfinite(values).all()
                for name, value in [('tokenization', encoded_at - start),
                                    ('input_preparation', prepared_at - encoded_at),
                                    ('model_forward', inferred_at - prepared_at),
                                    ('output_conversion', finished_at - inferred_at),
                                    ('instrumented_total', finished_at - start)]:
                    samples[name].append(value * 1000)
    return {name: summary(values) for name, values in samples.items()}, inputs


def capture(backend, inputs, size):
    path = HERE / f'epoch3_batch_{size}.gputrace'
    if path.exists():
        return {'batch_size': size, 'status': 'existing_capture_preserved', 'path': str(path.relative_to(ROOT))}
    active = False
    try:
        mx.synchronize()
        mx.metal.start_capture(str(path))
        active = True
        for _ in range(3):
            materialize(backend.model(**inputs))
        return {'batch_size': size, 'status': 'captured', 'path': str(path.relative_to(ROOT)),
                'forward_calls': 3, 'scope': 'unchanged model; resident inputs; timing runs exclude capture'}
    except Exception as error:
        return {'batch_size': size, 'status': 'unavailable', 'error': str(error)}
    finally:
        if active:
            mx.metal.stop_capture()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 2, 8, 16, 64, 128])
    parser.add_argument('--capture', action='store_true')
    args = parser.parse_args()
    if args.repeats < 5 or any(size < 1 or size > 128 for size in args.batch_sizes):
        parser.error('Use >=5 repeats and batch sizes in [1,128]')
    expected = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']
    assert sha256(CANDIDATES) == expected['sha256']
    rows = [json.loads(line) for line in itertools.islice(CANDIDATES.open(), 128)]
    flat = [(row['query'], c['text']) for row in rows for c in row['candidates']]
    backend = Backend('mlx')
    result = {'checkpoint': str(CHECKPOINT.relative_to(ROOT)), 'checkpoint_sha256': sha256(CHECKPOINT),
              'dataset_split': 'validation', 'candidate_sha256': expected['sha256'],
              'sample': 'first 128 fixed validation queries; first B pairs for each batch size',
              'sample_sha256': __import__('hashlib').sha256(json.dumps(flat).encode()).hexdigest(),
              'precision': 'float32; MLX_ENABLE_TF32=0', 'device': mx.device_info(),
              'platform': platform.platform(), 'mlx_version': importlib.metadata.version('mlx'),
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in
                              [Path(__file__), ROOT / 'experiments/S1/mlx_model.py', ROOT / 'experiments/S1/backend.py']},
              'repeats': args.repeats, 'warmups': 10, 'loading_seconds': backend.loading_seconds,
              'scope': 'steady-state MLX inference without HTTP; weights unchanged; no test-set tuning',
              'phase_order': 'instrumented, normal scoring and resident forward interleaved with rotating order',
              'limitations': [
                  'Synchronized wall times include Python dispatch and device synchronization, not pure GPU kernel durations.',
                  'Operation microbenchmarks materialize every stage and replay resident inputs; their sum is not whole-model latency.',
                  'Instrumented preparation adds a synchronization boundary; normal scoring is timed separately.',
                  'Different batch sizes contain prefixes of the fixed sample and can have different padded lengths.',
                  'Metal captures contain unchanged full graphs; Xcode is required to inspect GPU timelines.'],
              'results': [], 'metal_captures': []}
    for size in args.batch_sizes:
        queries, documents = map(list, zip(*flat[:size]))
        phase_times, inputs = phases(backend, queries, documents, args.repeats)
        ops, scores = operation_fixture(backend.model, inputs)
        reference = backend.forward(inputs)
        error = float(np.max(np.abs(reference - scores)))
        assert np.allclose(reference, scores, atol=1e-5, rtol=1e-5)
        assert np.array_equal(np.argsort(reference, kind='stable'), np.argsort(scores, kind='stable'))
        operations, categories = isolated_operations(ops, args.repeats)
        result['results'].append({'batch_size': size, 'padded_sequence_length': inputs['input_ids'].shape[1],
                                  'phases': phase_times, 'operations': operations, 'categories': categories,
                                  'instrumented_max_score_error': error, 'score_order_matches': True})
        print(f'batch={size} length={inputs["input_ids"].shape[1]} '
              f'normal={phase_times["normal_scoring"]["median_ms"]:.3f}ms '
              f'resident={phase_times["resident_forward"]["median_ms"]:.3f}ms '
              f'top={categories[0]["category"]} parity_error={error:.2g}', flush=True)
        if args.capture and size in (2, 64):
            result['metal_captures'].append(capture(backend, inputs, size))
            print(result['metal_captures'][-1], flush=True)
        write_json(HERE / 'profiling.json', result)
        del ops, inputs
        mx.clear_cache()
    # CPU call attribution is useful for dispatch/tokenization; it is not a GPU profiler.
    queries, documents = map(list, zip(*flat[:2]))
    cpu = cProfile.Profile()
    cpu.enable()
    for _ in range(100):
        backend.score(queries, documents)
    cpu.disable()
    cpu.dump_stats(str(HERE / 'cpu_dispatch.prof'))
    with (HERE / 'cpu_dispatch.txt').open('w') as stream:
        pstats.Stats(cpu, stream=stream).strip_dirs().sort_stats('cumulative').print_stats(35)
    print('MLX epoch-3 inference profiling complete.', flush=True)


if __name__ == '__main__':
    main()
