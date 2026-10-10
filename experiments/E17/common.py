"""Frozen inputs, deterministic requests, accuracy gates and clustered statistics."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from collections import Counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'runs/full_train/epoch_0003.pt'
EXPECTED = '2ea2608a9ec98083b89606d13bed1d7c6e7a90dc6160bbe44732166137538d52'
DATA = ROOT / 'data/e16/development.jsonl'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, data):
    from scripts.checkpoint_io import atomic_write_text
    atomic_write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', Path(path), replace_existing=True)


def rows():
    data = [json.loads(line) for line in DATA.open()]
    assert Counter(r['dataset'] for r in data) == {'scifact': 162, 'nfcorpus': 323}
    assert len({(r['dataset'], r['query_id']) for r in data}) == len(data)
    for r in data:
        assert len(r['document_ids']) == len(r['texts']) == len(set(r['document_ids'])) == 100
        assert r['qrels'] and any(v > 0 for v in r['qrels'].values())
        assert hashlib.sha256(' '.join(r['query'].casefold().split()).encode()).hexdigest() == r['query_group']
    return data


def key(row, k):
    return f"{row['dataset']}:{row['query_id']}:{k}"


def payload(row, k):
    if k not in (10, 50, 100):
        raise ValueError('Unsupported document count')
    return {'request_id': key(row, k), 'query': row['query'], 'top_k': k,
            'documents': [{'document_id': d, 'text': t}
                          for d, t in zip(row['document_ids'][:k], row['texts'][:k])]}


def ranked(ids, scores):
    if len(ids) != len(scores) or len(set(ids)) != len(ids) or not np.isfinite(scores).all():
        raise ValueError('Invalid candidate scores')
    return sorted(ids, key=lambda d: (-scores[ids.index(d)], d))


def interval(values, seed=1701):
    a = np.asarray(values, dtype=float)
    if not len(a) or not np.isfinite(a).all():
        raise ValueError('Invalid paired observations')
    rng = np.random.default_rng(seed)
    draws = np.mean(rng.choice(a, (10000, len(a)), replace=True), axis=1)
    return {'mean': float(a.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist(), 'n': len(a)}


def describe(values):
    a = np.asarray(values, dtype=float)
    return {'median': float(np.median(a)), 'range': [float(a.min()), float(a.max())],
            'iqr': np.quantile(a, [.25, .75]).tolist()}


def errors(reference, actual):
    b, a = np.asarray(reference), np.asarray(actual)
    if b.shape != a.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite/missing scores')
    e = abs(a-b)
    # Report stabilized relative error, plus the excluded near-zero population.
    relative = e / np.maximum(abs(b), 1e-3)
    return {'max': float(e.max()), 'median': float(np.median(e)),
            'p95': float(np.percentile(e, 95)), 'p99': float(np.percentile(e, 99)),
            'relative_max': float(relative.max()), 'relative_p99': float(np.percentile(relative, 99)),
            'relative_denominator_floor': 1e-3, 'reference_near_zero_count': int((abs(b)<1e-3).sum()),
            'strict_failures': int((e > 1e-5 + 1e-4*abs(b)).sum()), 'count': int(e.size)}


def acceptance(error, changes, metric_deltas, tolerance):
    strict = error['strict_failures'] == 0
    app = (error['max'] <= tolerance['application_max_absolute_error'] and
           changes['top1_fraction'] <= tolerance['application_top1_change_fraction'] and
           changes['top10_membership_fraction'] <= tolerance['application_top10_membership_change_fraction'] and
           all(d['ci95'][0] >= -tolerance['application_metric_loss'] for d in metric_deltas.values()))
    return {'strict_numerical_equivalence': strict, 'application_accepted': app,
            'accuracy_qualified': app, 'diagnostic_only': not app}


def environment():
    import importlib.metadata
    result = {'platform': platform.platform(), 'python': sys.version,
              'packages': {p: importlib.metadata.version(p) for p in ('torch','numpy','transformers','httpx')},
              'isolation': 'No exclusive hardware reservation; visible processes sampled, not proof of isolation.'}
    for name, cmd in {'hardware': ['system_profiler','SPHardwareDataType'],
                      'power': ['pmset','-g','batt'], 'thermal': ['pmset','-g','therm'],
                      'processes': ['ps','-axo','pid,comm,%cpu,rss'],
                      'gpu': ['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,utilization.gpu,memory.used','--format=csv']}.items():
        try:
            value = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=10)
            # Hardware UUID and serial number have no benchmark value.
            result[name] = '\n'.join(line for line in value.splitlines() if not any(s in line for s in ('Serial Number','UUID','Provisioning UDID')))
        except (OSError, subprocess.SubprocessError) as e:
            result[name] = f'unavailable: {type(e).__name__}'
    return result


def verify_plan():
    plan = json.loads((HERE/'plan.json').read_text())
    for path, digest in plan['input_sha256'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError(f'Frozen input changed: {path}')
    for path, digest in plan['code_sha256'].items():
        if sha(ROOT/path) != digest:
            raise RuntimeError(f'Frozen measured source changed: {path}')
    assert sha(CHECKPOINT) == EXPECTED
    return plan


def make_plan():
    if (HERE/'plan.json').exists():
        raise RuntimeError('Plan already frozen; use a new experiment for contract changes')
    data = rows()
    old = json.loads((ROOT/'experiments/E16/data_plan.json').read_text())
    assert sha(DATA) == old['files']['development.jsonl']
    source = {p: h for p,h in old['source_sha256'].items() if p.startswith('data/beir/')}
    for p,h in source.items():
        assert sha(ROOT/p) == h
    train_groups = {json.loads(line)['query_group'] for line in (ROOT/'data/e16/train.jsonl').open()}
    assert not train_groups.intersection(r['query_group'] for r in data)
    # Independently check development query text against all official test query texts.
    import csv
    blocked = set()
    for dataset in ('scifact','nfcorpus'):
        folder = ROOT/'data/beir'/dataset
        queries = {r['_id']:r['text'] for r in map(json.loads,(folder/'queries.jsonl').open())}
        for r in csv.DictReader((folder/'qrels/test.tsv').open(), delimiter='\t'):
            blocked.add(hashlib.sha256(' '.join(queries[r['query-id']].casefold().split()).encode()).hexdigest())
    assert not blocked.intersection(r['query_group'] for r in data)
    chosen = []
    for dataset in ('scifact','nfcorpus'):
        candidates = sorted((r for r in data if r['dataset']==dataset),
                            key=lambda r: hashlib.sha256(f"1701:{dataset}:{r['query_id']}".encode()).hexdigest())
        chosen += candidates[:8]
    inputs = [CHECKPOINT, DATA, ROOT/'data/e16/train.jsonl', ROOT/'experiments/E16/data_plan.json',
              ROOT/'experiments/E1/four_layer/config.json', ROOT/'runs/full_train/config.json',
              ROOT/'runs/full_train/epoch_0003_summary.json', ROOT/'artifacts/short_epoch_1024/manifest.json',
              *sorted((ROOT/'artifacts/short_epoch_1024/tokenizer').glob('*')),
              ROOT/'experiments/S1/conversion.json', ROOT/'experiments/S1/epoch_0003_mlx.safetensors']
    inputs = [p for p in inputs if p.is_file()]
    code = [*HERE.glob('*.py'), ROOT/'scripts/model.py', ROOT/'experiments/S1/backend.py',
            ROOT/'experiments/S1/mlx_model.py', ROOT/'experiments/E12/worker.py',
            ROOT/'experiments/E12/python_server.py', ROOT/'experiments/E15/metrics.py',
            ROOT/'experiments/E14/model.py', ROOT/'experiments/E14/kernel.py',
            ROOT/'experiments/E13/kernel.py', ROOT/'experiments/E13/triton_impl.py',
            *sorted((ROOT/'rust-server/src').glob('*.rs')), ROOT/'rust-server/Cargo.lock']
    plan = {'experiment': 'E17', 'seed':1701, 'checkpoint':str(CHECKPOINT.relative_to(ROOT)),
            'checkpoint_sha256':EXPECTED, 'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in inputs},
            'dataset_source_sha256':source, 'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},
            'datasets': {'scifact':162,'nfcorpus':323}, 'top100_pairs':48500,
            'overlap_verified': 'No E16 train/development normalized-text overlap; no official BEIR test/development overlap. E16 original-training exclusion provenance retained.',
            'document_counts':[10,50,100], 'max_length':128, 'pair_microbatch':32,
            'candidate_policy':'First K fixed BM25 candidates, no positive injection or duplicates; request preserves BM25 order; shared worker scores in document-ID order and ranks by (-score, document_id).',
            'http_subset':[[r['dataset'],r['query_id']] for r in chosen],
            'http_selection':'8 lowest SHA256(1701:dataset:query_id) per dataset, chosen before timings; 16 requests per repetition/configuration.',
            'accuracy_queries':485, 'accuracy_pairs_per_path':77600,
            'concurrency':[1,4,8], 'sessions':4, 'repeats_per_session':3, 'paired_repetitions':12,
            'ordering':'Seeded permutation of four HTTP paths; cyclic rotation across four sessions balances startup position. Seeded permutation of nine K/concurrency cells; cyclic rotation per repeat. Direct NVIDIA uses cyclic rotation across four primary paths; compiled path additional separate comparison.',
            'precision':'FP32, CUDA TF32 disabled, MLX_ENABLE_TF32=0, no autocast',
            'paths': {'strict':'Stock PyTorch, eval/inference_mode, MHA fastpath disabled',
                      'stock':'Stock PyTorch, eval/inference_mode, MHA fastpath enabled; profiler dispatch evidence required, label unsupported fusion honestly',
                      'cuda':'E14 custom CUDA, strict attention dispatch, fallback=False',
                      'triton':'E14 custom Triton, strict attention dispatch, four warps, fallback=False',
                      'compiled':'Additional stock torch.compile(backend=inductor, fullgraph=True, dynamic=False); bounded compatibility probe; first invocation not called cold compilation'},
            'accuracy_tolerance': {'strict_atol':1e-5,'strict_rtol':1e-4,
                                   'application_max_absolute_error':1e-2,'application_top1_change_fraction':.01,
                                   'application_top10_membership_change_fraction':.01,'application_metric_loss':.001,
                                   'application_rule':'Every dataset/K cell must pass max error, top1/membership fractions and lower paired 95% CI for nDCG/MRR >= -0.001; failed paths diagnostic only.',
                                   'http_atol':2e-5,'http_rtol':1e-4,'http_ranking':'Exact match to the same backend direct audit; validate every measured response.'},
            'direct_policy':'Flatten selected queries into pair microbatches (boundaries can span queries); HTTP never batches across requests.',
            'warmup':'Direct: 10 forwards per path on the first prepared 32 pairs; HTTP: worker warmup plus six requests (first selected query from each dataset at each K), all excluded.',
            'server': {'workers':1,'queue_capacity':32,'cross_request_batching':False,'request_timeout_seconds':30,'worker_timeout_seconds':60},
            'load':'Fixed-concurrency closed-loop, local loopback; next request submitted after completion; no internet or open-loop overload model.',
            'timing_scopes': {'operation':'Reuse E11/E14 historical operation results only; no new operation claims',
                              'forward':'Prepared device inputs, synchronize before and after model call; no tokenization, transfers or CPU conversion',
                              'direct':'CPU tokenization, transfer, forward, CPU-readable scores; excludes ranking, setup, file reads and warmup',
                              'http':'Client request/response including queue, IPC, tokenization, GPU, sorting and serialization; validation outside each request latency timer; startup reported separately'},
            'statistics':'Median/range/IQR across 12 repetitions. Matched repetition log speedups averaged within each session, bootstrap four session means (10000 draws, seed 1701); HTTP requests are not independent repetitions. Quality paired query bootstrap by dataset/K; no multiplicity correction, exploratory intervals.',
            'limits':'Four sessions give coarse uncertainty; short 16-request waves give noisy P99 and limited saturation coverage. Statistical intervals conditional on this machine/time, not hardware isolation.',
            'nvidia_status':'Pending: single BatchMode SSH attempt rejected authentication; user requested Apple only.'}
    assert sha(CHECKPOINT) == EXPECTED
    save(HERE/'plan.json', plan)


if __name__ == '__main__':
    make_plan()
