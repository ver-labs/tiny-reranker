"""Train all parameters on 10K pairs, profile a replay, and evaluate validation."""
import argparse
from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import statistics
import subprocess
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import BertTokenizer

from scripts.checkpoint_io import atomic_torch_save, atomic_write_text
from scripts.collate import PairCollator
from scripts.data import CLEAN_PAIRS, CleanPairDataset, split_for_query
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, check_metrics
from scripts.evaluate_validation import evaluate_model, score_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.short_epoch import capture_rng_state, split_manifest_record

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/E1/four_layer'
COUNT = 10000
BATCH = 16
SEED = 123
LR = 3e-4


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(name, data):
    atomic_write_text(json.dumps(data, indent=2) + '\n', HERE / name, replace_existing=True)


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def memory():
    return {'mps_current_tensor_bytes': torch.mps.current_allocated_memory(),
            'mps_current_driver_bytes': torch.mps.driver_allocated_memory(),
            'process_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


class Setup:
    def __init__(self):
        if not torch.backends.mps.is_available():
            raise RuntimeError('P1 requires the Apple GPU via MPS; CPU fallback refused')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('Disable MPS CPU fallback for measured GPU training')
        start = time.perf_counter()
        self.initial_manifest = json.loads((SOURCE / 'initial.json').read_text())
        if sha(SOURCE / 'initial.pt') != self.initial_manifest['initial_checkpoint_sha256'] or \
                sha(SOURCE / 'config.json') != self.initial_manifest['config_sha256']:
            raise RuntimeError('Source initialization/config hash mismatch')
        self.config = ScratchConfig.from_file(SOURCE / 'config.json')
        self.initial = torch.load(SOURCE / 'initial.pt', map_location='cpu', weights_only=True)
        if self.initial['config'] != asdict(self.config):
            raise RuntimeError('Checkpoint configuration mismatch')
        bundle = ROOT / 'artifacts/short_epoch_1024'
        manifest = json.loads((bundle / 'manifest.json').read_text())
        config = json.loads((SOURCE / 'config.json').read_text())
        if manifest['tokenizer_source_id'] != config['tokenizer_id'] or \
                manifest['tokenizer_source_revision'] != config['tokenizer_revision']:
            raise RuntimeError('Tokenizer provenance mismatch')
        for path, digest in manifest['file_sha256'].items():
            if path.startswith('tokenizer/') and sha(bundle / path) != digest:
                raise RuntimeError('Tokenizer asset hash mismatch')
        self.tokenizer = BertTokenizer.from_pretrained(bundle / 'tokenizer', local_files_only=True)
        self.collator = PairCollator(self.tokenizer, self.config.max_length)
        self.source_hash = sha(CLEAN_PAIRS)
        self.dataset = CleanPairDataset('train')
        expected = json.loads((ROOT / 'results/pair_cleaning.json').read_text())['clean_pairs_by_split']['train']
        if len(self.dataset) != expected:
            raise RuntimeError('Training split count mismatch')
        self.indices = random.Random(SEED).sample(range(len(self.dataset)), COUNT)
        rows = [self.dataset[i] for i in self.indices]
        if len({r['pair_id'] for r in rows}) != COUNT or \
                any(split_for_query(r['query']) != 'train' for r in rows):
            raise RuntimeError('Duplicate selection or query-split leakage')
        self.ids = [r['pair_id'] for r in rows]
        self.ids_hash = hashlib.sha256('\n'.join(self.ids).encode()).hexdigest()
        self.setup_seconds = time.perf_counter()-start
        write('sample.json', {'selection': 'seeded sample without replacement; ordered sample is the epoch order',
            'seed': SEED, 'pair_count': COUNT, 'training_split_pair_count': len(self.dataset),
            'clean_file_sha256': self.source_hash, 'split_manifest': split_manifest_record(),
            'pair_ids_sha256': self.ids_hash, 'dataset_indices': self.indices, 'pair_ids': self.ids})

    def fresh(self):
        seed()
        model = ScratchReranker(self.config)
        model.load_state_dict(self.initial['model_state_dict'], strict=True)
        model.to('mps').train()
        seed()  # Reset dropout RNG after module construction.
        if not all(p.requires_grad and p.dtype == torch.float32 for p in model.parameters()):
            raise RuntimeError('Every parameter must be trainable FP32')
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        scaler = torch.amp.GradScaler('mps')
        if not scaler.is_enabled():
            raise RuntimeError('MPS GradScaler is disabled')
        return model, optimizer, scaler


class Stages:
    def __init__(self, synchronize, trace=False):
        self.synchronize = synchronize
        self.trace = trace
        self.values = {}

    @contextmanager
    def measure(self, name):
        if self.synchronize:
            torch.mps.synchronize()
        start = time.perf_counter()
        with torch.profiler.record_function(name) if self.trace else nullcontext():
            yield
        if self.synchronize:
            torch.mps.synchronize()
        self.values.setdefault(name, []).append(time.perf_counter()-start)

    def summary(self):
        total = sum(sum(v) for v in self.values.values())
        return {k: {'total_seconds': sum(v), 'median_batch_ms': statistics.median(v)*1000,
                    'percent_of_measured_stage_time': 100*sum(v)/total, 'batches': len(v)}
                for k,v in self.values.items()}


def step(setup, model, optimizer, scaler, indices, stages):
    with stages.measure('data_reading'):
        rows = [setup.dataset[i] for i in indices]
    with stages.measure('tokenization_and_collation'):
        batch = setup.collator(rows)
    with stages.measure('input_transfer'):
        inputs = batch['inputs'].to('mps')
        labels = batch['labels'].to('mps')
    with stages.measure('zero_grad'):
        optimizer.zero_grad(set_to_none=True)
    with stages.measure('forward_and_loss'):
        with torch.autocast('mps', dtype=torch.float16):
            logits = model(**inputs)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    with stages.measure('forward_checks'):
        if logits.shape != labels.shape or logits.dtype != torch.float16 or loss.dtype != torch.float32 or \
                not bool(torch.isfinite(logits).all() & torch.isfinite(loss)):
            raise RuntimeError('Invalid/nonfinite forward output or mixed precision dtype')
    live_memory = memory()
    with stages.measure('backward_and_unscale'):
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    with stages.measure('gradient_checks'):
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or g.dtype != torch.float32 for g in gradients):
            raise RuntimeError('Missing gradient or parameter gradient left FP32')
        # One scalar read rather than synchronizing separately for every parameter.
        finite = bool(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
    with stages.measure('optimizer_and_scaler'):
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        after = scaler.get_scale()
        skipped = after < before
        if skipped == finite:
            raise RuntimeError('Scaler update/skip is inconsistent with gradient checks')
    with stages.measure('logging_values'):
        loss_value = float(loss.detach().item())
    return {'loss': loss_value, 'skipped': skipped, 'scale': after,
            'memory': live_memory, 'pair_ids': batch['pair_ids']}


def normal_run(setup):
    model, optimizer, scaler = setup.fresh()
    stages = Stages(False)
    losses, batch_times, memory_samples, seen_ids, skipped = [], [], [], [], 0
    scale_start = scaler.get_scale()
    torch.mps.synchronize()
    start = time.perf_counter()
    for n, offset in enumerate(range(0, COUNT, BATCH), 1):
        begin = time.perf_counter()
        result = step(setup, model, optimizer, scaler, setup.indices[offset:offset+BATCH], stages)
        batch_times.append(time.perf_counter()-begin)
        losses.append(result['loss'])
        seen_ids.extend(result['pair_ids'])
        skipped += result['skipped']
        memory_samples.append(result['memory'])
        if n % 125 == 0:
            print(f'TRAIN batches={n}/625 pairs={len(seen_ids)} loss={statistics.mean(losses[-125:]):.4f} '
                  f'skips={skipped} elapsed={time.perf_counter()-start:.1f}s', flush=True)
    torch.mps.synchronize()
    elapsed = time.perf_counter()-start
    if seen_ids != setup.ids or len(set(seen_ids)) != COUNT:
        raise RuntimeError('Training sample/order mismatch')
    checkpoint = HERE / 'train_10000.pt'
    save_start = time.perf_counter()
    atomic_torch_save({'model_state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(), 'gradient_scaler_state_dict': scaler.state_dict(),
        'completed_epochs': 1, 'update_count': len(losses)-skipped,
        'config': asdict(setup.config), 'rng_state': capture_rng_state(torch.device('mps')),
        'data_split_manifest': split_manifest_record(),
        'sampler_state': {'order': 'saved sample indices', 'consumed': COUNT, 'next_sample_index': 0,
                          'next_epoch_index': 1, 'mid_epoch_resume_supported': False},
        'training_config': {'learning_rate': LR, 'batch_size': BATCH, 'precision': 'fp16_autocast',
            'seed': SEED, 'train_pairs': COUNT, 'pair_ids_sha256': setup.ids_hash,
            'clean_file_sha256': setup.source_hash, 'sample_file': 'sample.json', 'shuffle': 'seeded sample order',
            'optimizer': 'AdamW', 'optimizer_defaults': optimizer.defaults,
            'scheduler': None, 'gradient_accumulation': False,
            'activation_checkpointing': False, 'num_workers': 0}}, checkpoint)
    digest = sha(checkpoint)
    loaded = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if loaded['update_count'] != len(losses)-skipped or \
            len(loaded['optimizer_state_dict']['state']) != len(list(model.parameters())) or \
            loaded['gradient_scaler_state_dict'] != scaler.state_dict() or \
            not all(torch.equal(v.detach().cpu(), loaded['model_state_dict'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('Checkpoint reload verification failed')
    save_seconds = time.perf_counter()-save_start
    windows = [sum(batch_times[i:i+125]) for i in range(0,len(batch_times),125)]
    report = {'purpose': 'one normal 10K training epoch; no operator profiler or per-stage GPU synchronization',
        'training_pairs': COUNT, 'pair_batch_size': BATCH, 'batches': len(losses),
        'optimizer_updates': len(losses)-skipped, 'skipped_updates': skipped,
        'mean_training_bce': statistics.mean(losses), 'training_seconds': elapsed,
        'pairs_per_second': COUNT/elapsed, 'window_2000_pairs_seconds': windows,
        'per_batch_seconds': batch_times, 'per_batch_bce': losses,
        'instrumentation': 'CPU wall timers, required finite checks/scaler checks, loss reads, and allocator snapshots; no operator trace',
        'setup_seconds': setup.setup_seconds, 'checkpoint_save_and_reload_seconds': save_seconds,
        'checkpoint': str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256': digest,
        'source_initial_checkpoint_sha256': setup.initial_manifest['initial_checkpoint_sha256'],
        'config_sha256': setup.initial_manifest['config_sha256'], 'sample_pair_ids_sha256': setup.ids_hash,
        'backend': 'Apple GPU via MPS', 'precision': 'FP16 autocast; FP32 weights, gradients and BCE loss',
        'trainable_parameters': sum(p.numel() for p in model.parameters()), 'all_parameters_trained': True,
        'grad_scaler': {'initial_scale': scale_start, 'final_scale': scaler.get_scale()},
        'memory': {k:max(s[k] for s in memory_samples) for k in memory_samples[0]},
        'memory_measurement': 'MPS current-allocation snapshots with forward activations live; not true peak. macOS RSS is process high-water mark.',
        'environment': {'python':platform.python_version(), 'platform':platform.platform(),
            'versions': {p:importlib.metadata.version(p) for p in ['torch','transformers','numpy']},
            'code_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'code_has_uncommitted_changes': True},
        'test_split_scored': False}
    changed = [k for k,v in loaded['model_state_dict'].items()
               if not torch.equal(v, setup.initial['model_state_dict'][k])]
    steps = {int(v['step'].item()) for v in loaded['optimizer_state_dict']['state'].values()}
    if len(changed) != len(loaded['model_state_dict']) or steps != {len(losses)-skipped}:
        raise RuntimeError('Model/optimizer training coverage audit failed')
    write('checkpoint_audit.json', {
        'all_state_tensors_changed_from_random_initialization': len(changed),
        'total_state_tensors': len(loaded['model_state_dict']),
        'optimizer_states': len(loaded['optimizer_state_dict']['state']),
        'optimizer_steps': sorted(steps), 'scaler_state': loaded['gradient_scaler_state_dict'],
        'sample_unique_pairs': len(set(seen_ids))})
    write('training.json', report)
    print(f'TRAIN COMPLETE: {elapsed:.2f}s; {COUNT/elapsed:.1f} pairs/s; saved and reloaded {checkpoint}', flush=True)
    return model, report


def profile_replay(setup):
    model, optimizer, scaler = setup.fresh()
    # Warm up in the diagnostic model only; these updates never affect the real checkpoint.
    for offset in range(0, 5*BATCH, BATCH):
        step(setup, model, optimizer, scaler, setup.indices[offset:offset+BATCH], Stages(False))
    stages = Stages(True)
    start = time.perf_counter()
    for offset in range(5*BATCH, 55*BATCH, BATCH):
        step(setup, model, optimizer, scaler, setup.indices[offset:offset+BATCH], stages)
    torch.mps.synchronize()
    synchronized_seconds = time.perf_counter()-start
    profile = {'profile_batches': 50, 'profile_pairs': 800, 'warmup_batches': 5,
        'synchronized_replay_seconds': synchronized_seconds,
        'timing_label': 'wall time with GPU synchronization around every stage; includes dispatch and synchronization overhead, not GPU kernel time',
        'sample_scope': 'early training replay, 50 batches following five warmup batches from the 10K sample',
        'stages': stages.summary(), 'replay_weights_discarded': True}
    # CPU operator tracing is separate again, so tracing does not skew the stage breakdown.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                               record_shapes=True) as trace:
        for offset in range(55*BATCH, 65*BATCH, BATCH):
            step(setup, model, optimizer, scaler, setup.indices[offset:offset+BATCH], Stages(False, trace=True))
        torch.mps.synchronize()
    trace.export_chrome_trace(str(HERE / 'cpu_trace.json'))
    table = trace.key_averages().table(sort_by='self_cpu_time_total', row_limit=30)
    atomic_write_text(table + '\n', HERE / 'cpu_operators.txt', replace_existing=True)
    profile['cpu_trace'] = {'batches': 10, 'activity': 'CPU only; no GPU kernel trace claimed',
                            'file': 'cpu_trace.json', 'operators_file': 'cpu_operators.txt'}
    write('profiling.json', profile)
    print('PROFILE COMPLETE:', json.dumps(profile['stages'], indent=2), flush=True)


def validate(setup, model, report):
    expected = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']
    digest = sha(CANDIDATES)
    if digest != expected['sha256']:
        raise RuntimeError('Validation candidates mismatch')
    check_metrics()
    reranker = SimpleNamespace(model=model.eval(), tokenizer=setup.tokenizer, device=torch.device('mps'))
    for _ in range(10):
        score_pairs(reranker, ['warmup'], ['warmup document'])
    torch.mps.synchronize()
    start = time.perf_counter()
    summary = evaluate_model('train_10000', reranker, expected, digest,
        'Four-layer scratch encoder; all parameters trained from E1 seed-42 random weights on 10,000 seeded training pairs, one epoch, FP16 autocast with GradScaler.',
        extra_metadata={'model_id': 'scratch_encoder_p1_10000', 'checkpoint':report['checkpoint'], 'checkpoint_sha256':report['checkpoint_sha256'],
                        'sample_pair_ids_sha256':setup.ids_hash, 'test_split_scored':False}, result_dir=HERE)
    torch.mps.synchronize()
    summary['full_validation_seconds'] = time.perf_counter()-start
    write('train_10000_summary.json', summary)
    return summary


def summarize(setup, training, validation):
    profile = json.loads((HERE / 'profiling.json').read_text())
    stages = profile['stages']
    biggest = max(stages, key=lambda k: stages[k]['total_seconds'])
    full_count = len(setup.dataset)
    estimates = [s/2000*full_count for s in training['window_2000_pairs_seconds']]
    e1 = json.loads((SOURCE / 'four_layer_epoch1_summary.json').read_text())
    previous_best = json.loads((SOURCE / 'four_layer_head_only_summary.json').read_text())
    report = {'training': {'pairs': COUNT, 'updates': training['optimizer_updates'],
        'skips':training['skipped_updates'], 'seconds':training['training_seconds'],
        'pairs_per_second': training['pairs_per_second']},
        'validation': {'queries':validation['queries'], 'ndcg_at_10':validation['ndcg_at_10'],
            'mrr_at_10':validation['mrr_at_10'], 'e1_1024_pair_ndcg':e1['ndcg_at_10'],
            'ndcg_change':validation['ndcg_at_10']-e1['ndcg_at_10'],
            'previous_best_scratch_ndcg': previous_best['ndcg_at_10'],
            'ndcg_change_from_previous_best': validation['ndcg_at_10']-previous_best['ndcg_at_10'],
            'comparison_note': 'Different training exposure, sample/order and precision; not an isolated-method experiment'},
        'profiling': {'largest_measured_stage':biggest, 'stages':stages},
        'full_training_estimate': {'training_pairs':full_count,
            'one_epoch_seconds_from_total': training['training_seconds']/COUNT*full_count,
            'one_epoch_seconds_from_fastest_2k_window':min(estimates),
            'one_epoch_seconds_from_slowest_2k_window':max(estimates),
            'excludes': 'startup, validation, checkpoint saves; longer-run thermal/load and token-length variation may change speed'},
        'test_split_scored':False}
    write('report.json', report)
    print('FINAL REPORT:',json.dumps(report,indent=2),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluate-only', action='store_true', help='Evaluate an existing P1 checkpoint and regenerate report')
    args = parser.parse_args()
    if not args.evaluate_only and ((HERE/'train_10000.pt').exists() or (HERE/'training.json').exists()):
        parser.error('P1 training already exists; use --evaluate-only to recover evaluation')
    setup = Setup()
    print(f'SETUP: {len(setup.dataset):,} training pairs available; sampled {COUNT:,}; setup {setup.setup_seconds:.2f}s',flush=True)
    if args.evaluate_only:
        training = json.loads((HERE/'training.json').read_text())
        if sha(HERE/'train_10000.pt') != training['checkpoint_sha256'] or setup.ids_hash != training['sample_pair_ids_sha256']:
            raise RuntimeError('Existing P1 checkpoint/sample differs')
        if not (HERE/'profiling.json').exists():
            profile_replay(setup)
            torch.mps.empty_cache()
        model = ScratchReranker(setup.config)
        model.load_state_dict(torch.load(HERE/'train_10000.pt', map_location='cpu', weights_only=True)['model_state_dict'])
        model.to('mps')
    else:
        model, training = normal_run(setup)
        # Release the real GPU model during replay; reload its saved weights afterward.
        model.cpu()
        del model
        torch.mps.empty_cache()
        profile_replay(setup)
        torch.mps.empty_cache()
        model = ScratchReranker(setup.config)
        model.load_state_dict(torch.load(HERE/'train_10000.pt', map_location='cpu', weights_only=True)['model_state_dict'])
        model.to('mps')
    validation = validate(setup, model, training)
    summarize(setup, training, validation)


if __name__ == '__main__':
    main()
