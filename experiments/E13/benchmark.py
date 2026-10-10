"""Run on Linux/NVIDIA; record pending hardware rather than inventing Mac timings."""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time

import torch

from scripts.checkpoint_io import atomic_write_text
from scripts.model import ScratchConfig
from scripts.evaluate_initial import ranking_metrics
from .kernel import reference, residual_layer_norm
from .model import TritonReranker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODES = ('original', 'reference', 'triton', 'optimized')


def sha256(path):
    with path.open('rb') as source: return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, data):
    atomic_write_text(json.dumps(data, indent=2) + '\n', path, replace_existing=True)


def environment():
    info = {'platform': platform.platform(), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
            'cuda_available': torch.cuda.is_available(), 'triton_installed': importlib.util.find_spec('triton') is not None}
    if info['cuda_available']:
        info.update(gpu=torch.cuda.get_device_name(), compute_capability=torch.cuda.get_device_capability())
        try:
            info['nvidia_driver'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True, timeout=5).strip()
        except (OSError, subprocess.SubprocessError): info['nvidia_driver'] = 'unavailable'
    if info['triton_installed']: info['triton'] = importlib.metadata.version('triton')
    return info


def summary(values):
    return {'median': statistics.median(values), 'minimum': min(values), 'maximum': max(values), 'values': values}


def load_fixture():
    checkpoint = ROOT / 'runs/full_train/epoch_0003.pt'
    checkpoint_summary = json.loads(checkpoint.with_name('epoch_0003_summary.json').read_text())
    config_path = ROOT / 'experiments/E1/four_layer/config.json'
    if sha256(checkpoint) != checkpoint_summary['checkpoint_sha256']: raise RuntimeError('Checkpoint hash mismatch')
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['provenance_sha256'] != sha256(checkpoint.with_name('config.json')): raise RuntimeError('Training provenance mismatch')
    provenance = json.loads(checkpoint.with_name('config.json').read_text())
    if sha256(config_path) != provenance['config_sha256']: raise RuntimeError('Model configuration mismatch')
    model = TritonReranker(ScratchConfig.from_file(config_path)).cuda().eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    assets = ROOT / 'artifacts/short_epoch_1024'
    manifest = json.loads((assets / 'manifest.json').read_text())
    config_data = json.loads(config_path.read_text())
    if manifest['tokenizer_source_id'] != config_data['tokenizer_id'] or manifest['tokenizer_source_revision'] != config_data['tokenizer_revision']:
        raise RuntimeError('Tokenizer provenance mismatch')
    for relative, expected in manifest['file_sha256'].items():
        if relative.startswith('tokenizer/') and sha256(assets / relative) != expected: raise RuntimeError('Tokenizer hash mismatch')
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(assets / 'tokenizer', local_files_only=True)
    if tokenizer.vocab_size != model.config.vocab_size or tokenizer.pad_token_id != model.config.pad_token_id:
        raise RuntimeError('Tokenizer configuration mismatch')
    candidates = ROOT / 'data/processed/eval_validation_candidates.jsonl'
    expected = json.loads((ROOT / 'results/evaluation_candidates.json').read_text())['splits']['validation']
    if sha256(candidates) != expected['sha256']: raise RuntimeError('Validation data hash mismatch')
    with candidates.open() as source: rows = [json.loads(line) for line in itertools.islice(source, 128)]
    return model, tokenizer, rows, checkpoint, expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--num-warps', type=int, choices=[1, 2, 4], default=4)
    args = parser.parse_args()
    if args.repeats < 4: parser.error('Use at least four repeats')
    info = environment()
    compatible = (platform.system() == 'Linux' and info['cuda_available'] and torch.version.hip is None and
                  info['triton_installed'] and tuple(info.get('compute_capability', (0, 0))) >= (8, 0))
    if not compatible:
        report = {'status': 'pending_nvidia_execution', 'environment': info,
                  'reason': 'Requires Linux + NVIDIA CUDA (compute capability >=8.0) + compatible Triton',
                  'gpu_correctness_verified': False, 'performance_measured': False}
        save(HERE / 'hardware_status.json', report)
        print(json.dumps(report, indent=2))
        if args.require_gpu: raise SystemExit(2)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, tokenizer, rows, checkpoint, manifest = load_fixture()
    flat = [(r['query'], c['text']) for r in rows for c in r['candidates']]
    def prepare(pairs):
        encoded = tokenizer([q for q, _ in pairs], [d for _, d in pairs], padding=True,
                            truncation='longest_first', max_length=model.config.max_length,
                            return_attention_mask=True, return_token_type_ids=True, return_tensors='pt')
        return {k: v.cuda() for k, v in encoded.items()}
    def scores(inputs, mode):
        # Exact original/reference/Triton share attention dispatch. CUDA's fused
        # stock encoder can change trained-checkpoint scores, so report that
        # optimized implementation as a separate fourth path.
        previous = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(mode == 'optimized')
        try: return model(**inputs, kernel_mode='original' if mode == 'optimized' else mode,
                          num_warps=args.num_warps).cpu()
        finally: torch.backends.mha.set_fastpath_enabled(previous)
    result = {'status': 'complete', 'environment': info, 'checkpoint_sha256': sha256(checkpoint),
              'candidate_sha256': manifest['sha256'], 'dataset_split': 'validation',
              'sample_queries': len(rows), 'sample_pairs': len(flat), 'repeats': args.repeats,
              'sample_sha256': hashlib.sha256(json.dumps(flat).encode()).hexdigest(),
              'precision': 'FP32; TF32 disabled', 'num_warps': args.num_warps,
              'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), HERE / 'model.py', HERE / 'kernel.py', HERE / 'triton_impl.py', ROOT / 'scripts/model.py']},
              'scope': 'direct synchronized wall timings; no HTTP; CPU-readable model scores',
              'limitations': ['Quality is a fixed 128-query parity fixture, not a full validation/test evaluation.',
                             'Compare reference and Triton for isolated kernel effect; original disables fused fast paths for exact score parity.',
                             'The optimized stock path is timed separately and its score/ranking differences are reported, not treated as exact parity.',
                             'Compilation is excluded from warm timing; no training or backward support.'],
              'operation': [], 'model': []}
    with torch.inference_mode():
        values = (torch.randn((2, 128, 128), device='cuda'), torch.randn((2, 128, 128), device='cuda'),
                  model.encoder_layers[0].norm2.weight, model.encoder_layers[0].norm2.bias)
        torch.cuda.synchronize(); start = time.perf_counter()
        residual_layer_norm(*values, fallback=False, num_warps=args.num_warps); torch.cuda.synchronize()
        result['first_kernel_use_ms'] = (time.perf_counter() - start) * 1000
        # Match E11's first-layer operation fixtures and 30 paired repeats.
        for size in (1, 2, 8, 16, 64, 128):
            fixture = prepare(flat[:size])
            positions = torch.arange(fixture['input_ids'].shape[1], device='cuda')
            residual = model.embedding_norm(model.token_embeddings(fixture['input_ids']) +
                model.position_embeddings(positions)[None] + model.segment_embeddings(fixture['token_type_ids']))
            layer = model.encoder_layers[0]
            norm = layer.norm1(residual)
            previous = torch.backends.mha.get_fastpath_enabled()
            torch.backends.mha.set_fastpath_enabled(False)
            try:
                branch = layer.self_attn(norm, norm, norm, key_padding_mask=fixture['attention_mask'] == 0,
                                         need_weights=False)[0].contiguous()
            finally: torch.backends.mha.set_fastpath_enabled(previous)
            values = (residual.contiguous(), branch, layer.norm2.weight, layer.norm2.bias)
            torch.cuda.synchronize()
            for warps in (1, 2, 4):
                actual = residual_layer_norm(*values, fallback=False, num_warps=warps)
                expected = reference(*values)
                for a, b in zip(actual, expected): torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
                functions = {'reference': lambda: reference(*values),
                             'triton': lambda: residual_layer_norm(*values, fallback=False, num_warps=warps)}
                for function in functions.values():
                    for _ in range(10): function()
                timings = {'reference': [], 'triton': []}
                for repeat in range(30):
                    for mode in (('reference', 'triton') if repeat % 2 == 0 else ('triton', 'reference')):
                        torch.cuda.synchronize(); begin = time.perf_counter()
                        functions[mode](); torch.cuda.synchronize()
                        timings[mode].append((time.perf_counter() - begin) * 1000)
                result['operation'].append({'batch_size': size, 'num_warps': warps,
                    'shape': list(residual.shape), 'repeats': 30,
                    'reference_ms': summary(timings['reference']), 'triton_ms': summary(timings['triton'])})
        inputs = prepare(flat)
        outputs = {mode: scores(inputs, mode) for mode in MODES}
        torch.testing.assert_close(outputs['reference'], outputs['original'], atol=1e-5, rtol=1e-4)
        torch.testing.assert_close(outputs['triton'], outputs['reference'], atol=1e-5, rtol=1e-4)
        metrics = {}; orders = {}; offset = 0
        for mode, values_out in outputs.items():
            mrr = ndcg = 0.; orders[mode] = []
            for row in rows:
                n = len(row['candidates'])
                candidates = sorted(zip(row['candidates'], values_out[offset:offset+n].tolist()),
                                    key=lambda x: (-x[1], x[0]['document_id']))
                orders[mode].append([c['document_id'] for c, _ in candidates])
                m, d = ranking_metrics([c['label'] for c, _ in candidates]); mrr += m; ndcg += d
                offset += n
            metrics[mode] = {'mrr_at_10': mrr / len(rows), 'ndcg_at_10': ndcg / len(rows)}
            offset = 0
        result['sample_parity'] = {'metrics': metrics,
            'changed_rankings': sum(a != b for a, b in zip(orders['reference'], orders['triton'])),
            'max_score_error': float((outputs['reference'] - outputs['triton']).abs().max()),
            'optimized_max_score_error': float((outputs['reference'] - outputs['optimized']).abs().max()),
            'optimized_changed_rankings': sum(a != b for a, b in zip(orders['reference'], orders['optimized']))}
        for size in (1, 2, 8, 16, 64, 128):
            chunks = [flat[i:i+size] for i in range(0, len(flat), size)]
            resident = [prepare(c) for c in chunks]
            timings = {scope: {mode: [] for mode in MODES} for scope in ('forward', 'scoring')}
            latencies = {scope: {mode: [] for mode in MODES} for scope in ('forward', 'scoring')}
            for mode in MODES:
                for item in resident: scores(item, mode)
                for _ in range(10): scores(resident[0], mode)
            for repeat in range(args.repeats):
                modes = list(MODES); shift = repeat % len(MODES)
                for mode in modes[shift:] + modes[:shift]:
                    for scope in (('forward', 'scoring') if repeat % 2 == 0 else ('scoring', 'forward')):
                        torch.cuda.synchronize(); begin = time.perf_counter()
                        calls = []
                        for chunk, item in zip(chunks, resident):
                            call_start = time.perf_counter()
                            scores(prepare(chunk) if scope == 'scoring' else item, mode)
                            calls.append((time.perf_counter() - call_start) * 1000)
                        torch.cuda.synchronize(); elapsed = time.perf_counter() - begin
                        timings[scope][mode].append(len(flat) / elapsed)
                        latencies[scope][mode].append(statistics.median(calls))
            result['model'].append({'pair_batch_size': size, 'pairs_per_second':
                {scope: {mode: summary(vals) for mode, vals in modes.items()} for scope, modes in timings.items()},
                'median_batch_ms': {scope: {mode: summary(vals) for mode, vals in modes.items()} for scope, modes in latencies.items()}})
            print(f'CUDA/Triton batch={size} complete', flush=True)
    save(HERE / 'benchmark.json', result)
    save(HERE / 'hardware_status.json', {'status': 'gpu_benchmark_complete', 'environment': info,
        'sample_parity_verified': True, 'full_quality_verified': False, 'performance_measured': True})


if __name__ == '__main__': main()
