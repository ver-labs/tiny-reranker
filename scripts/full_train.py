"""Full cleaned training split, with mid-epoch resume and epoch validation."""
import argparse
import fcntl
from contextlib import nullcontext
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
from transformers import BertTokenizer

from experiments.P1.run import Setup as PilotSetup, SOURCE, Stages, sha, step
from scripts.checkpoint_io import atomic_torch_save, atomic_write_text
from scripts.collate import PairCollator
from scripts.data import CLEAN_PAIRS, CleanPairDataset, split_for_query
from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, check_metrics
from scripts.evaluate_validation import evaluate_model, score_pairs
from scripts.model import ScratchConfig, ScratchReranker
from scripts.short_epoch import capture_rng_state, split_manifest_record

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUN = ROOT / 'runs/full_train'
SETTINGS = {
    'model': 'four-layer scratch encoder; all parameters trainable',
    'initialization': 'saved E1 seed-42 random weights; not the 10K checkpoint',
    'batch_size': 16, 'max_length': 128, 'padding': 'dynamic',
    'precision': 'FP16 autocast with GradScaler; FP32 parameters',
    'optimizer': 'AdamW', 'learning_rate': 3e-4, 'dropout': 0.1,
    'loss': 'BCEWithLogits', 'training_seed': 123,
    'gradient_accumulation': False, 'activation_checkpointing': False,
    'num_workers': 0, 'test_split_scored': False,
    'order': 'NumPy permutation of every training index; seed = 123 + epoch index',
}


def json_write(path, data):
    atomic_write_text(json.dumps(data, indent=2) + '\n', path, replace_existing=True)


def order_for_epoch(count, epoch):
    return np.random.default_rng(SETTINGS['training_seed'] + epoch).permutation(count)


def validate_sampler(state, count):
    offset = state['next_sample_offset']
    if not 0 <= offset <= count or (offset != count and offset % SETTINGS['batch_size']):
        raise RuntimeError('Resume offset is not a valid consumed batch boundary')
    order = state['order']
    if order is None:
        if offset != 0:
            raise RuntimeError('Missing sampler order at a nonzero offset')
        return
    order = order.numpy() if isinstance(order, torch.Tensor) else np.asarray(order)
    if order.shape != (count,) or not np.array_equal(np.sort(order), np.arange(count)):
        raise RuntimeError('Saved sampler order is not a permutation of the training split')


def restore_rng(rng, device):
    random.setstate(rng['python'])
    n = rng['numpy']
    np.random.set_state((n['algorithm'], np.array(n['keys'], dtype=np.uint32),
                        n['position'], n['has_gauss'], n['cached_gaussian']))
    torch.set_rng_state(rng['torch_cpu'])
    if device.type == 'mps':
        if rng['torch_mps'] is None:
            raise RuntimeError('Missing MPS RNG state')
        torch.mps.set_rng_state(rng['torch_mps'])


class NoProfile:
    def measure(self, name):
        return nullcontext()


class VerifiedTrainingDataset(CleanPairDataset):
    def __getitem__(self, index):
        row = super().__getitem__(index)
        if split_for_query(row['query']) != 'train':
            raise RuntimeError('Training query belongs to a held-out split')
        return row


def profiling_windows(total_batches):
    return {'early': 33, 'middle': total_batches // 2, 'late': total_batches - 200}


class EpochProfile:
    """Bounded stage and CPU trace windows on the actual training trajectory."""
    def __init__(self, run_dir, epoch, total_batches, saved):
        self.run_dir, self.epoch = run_dir, epoch
        self.starts = profiling_windows(total_batches)
        self.saved = saved
        self.trace = None
        self.trace_name = None
        self.trace_start = None

    def stages(self, batch):
        for name, start in self.starts.items():
            window = self.saved.setdefault(name, {'start_batch': start, 'stages': {}, 'cpu_traces': []})
            if start <= batch < start + 50:
                stage = Stages(True)
                stage.values = window['stages']
                return stage
            if start + 50 <= batch < start + 60:
                if self.trace is None:
                    self.trace = torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True)
                    self.trace.__enter__()
                    self.trace_name, self.trace_start = name, batch
                return Stages(False, trace=True)
        return NoProfile()

    def after_batch(self, batch):
        if self.trace is not None and batch >= self.starts[self.trace_name] + 59:
            self.close_trace(batch)

    def close_trace(self, batch):
        if self.trace is None:
            return
        torch.mps.synchronize()
        self.trace.__exit__(None, None, None)
        prefix = f'epoch_{self.epoch:04d}_{self.trace_name}_from_batch_{self.trace_start:05d}'
        self.trace.export_chrome_trace(str(self.run_dir / f'{prefix}_cpu_trace.json'))
        atomic_write_text(self.trace.key_averages().table(sort_by='self_cpu_time_total',row_limit=30)+'\n',
                          self.run_dir / f'{prefix}_cpu_operators.txt', replace_existing=True)
        self.saved[self.trace_name]['cpu_traces'].append({
            'first_batch':self.trace_start,'last_batch':batch,
            'trace_file':f'{prefix}_cpu_trace.json','operator_file':f'{prefix}_cpu_operators.txt',
            'activity':'CPU only; no GPU kernel timing claimed'})
        self.trace = None

    def flush(self):
        windows = {}
        for name, window in self.saved.items():
            stage = Stages(True)
            stage.values = window['stages']
            windows[name] = {'start_batch':window['start_batch'],
                'synchronized_batches_measured':len(next(iter(window['stages'].values()), [])),
                'stages':stage.summary() if window['stages'] else {},'cpu_traces':window['cpu_traces']}
        json_write(self.run_dir / f'epoch_{self.epoch:04d}_profiling.json', {
            'scope':'Actual full-data training; 50 synchronized batches plus 10 CPU-traced batches at each early/middle/late window',
            'timing_label':'stage wall time includes GPU dispatch and synchronization overhead',
            'epoch_timing_includes_bounded_profiling_overhead':True,'windows':windows})


class FullSetup:
    # Reuse the model/optimizer/scaler construction proven in the 10K pilot.
    fresh = PilotSetup.fresh

    def __init__(self, run_dir):
        if not torch.backends.mps.is_available():
            raise RuntimeError('Full training requires MPS GPU access')
        if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
            raise RuntimeError('CPU fallback is enabled; refusing measured MPS training')
        self.initial_manifest = json.loads((SOURCE / 'initial.json').read_text())
        if sha(SOURCE/'initial.pt') != self.initial_manifest['initial_checkpoint_sha256'] or \
                sha(SOURCE/'config.json') != self.initial_manifest['config_sha256']:
            raise RuntimeError('Initial checkpoint/config differs from E1')
        self.config = ScratchConfig.from_file(SOURCE/'config.json')
        self.initial = torch.load(SOURCE/'initial.pt', map_location='cpu', weights_only=True)
        if self.initial['config'] != asdict(self.config):
            raise RuntimeError('Initial model configuration mismatch')
        bundle = ROOT/'artifacts/short_epoch_1024'
        tokenizer_manifest = json.loads((bundle/'manifest.json').read_text())
        source_config = json.loads((SOURCE/'config.json').read_text())
        if tokenizer_manifest['tokenizer_source_id'] != source_config['tokenizer_id'] or \
                tokenizer_manifest['tokenizer_source_revision'] != source_config['tokenizer_revision']:
            raise RuntimeError('Local tokenizer provenance mismatch')
        for path, digest in tokenizer_manifest['file_sha256'].items():
            if path.startswith('tokenizer/') and sha(bundle/path) != digest:
                raise RuntimeError('Local tokenizer asset hash mismatch')
        self.tokenizer = BertTokenizer.from_pretrained(bundle/'tokenizer', local_files_only=True)
        self.collator = PairCollator(self.tokenizer, self.config.max_length)
        self.source_hash = sha(CLEAN_PAIRS)
        pilot = json.loads((ROOT/'experiments/P1/sample.json').read_text())
        if self.source_hash != pilot['clean_file_sha256']:
            raise RuntimeError('Clean data differs from the profiled pilot')
        self.dataset = VerifiedTrainingDataset('train')
        expected = json.loads((ROOT/'results/pair_cleaning.json').read_text())['clean_pairs_by_split']['train']
        if len(self.dataset) != expected:
            raise RuntimeError('Clean training split count mismatch')
        self.provenance = {
            'settings': SETTINGS, 'model_config': asdict(self.config),
            'training_pairs': len(self.dataset),
            'initial_checkpoint_sha256': self.initial_manifest['initial_checkpoint_sha256'],
            'config_sha256': self.initial_manifest['config_sha256'],
            'clean_file_sha256': self.source_hash, 'split_manifest': split_manifest_record(),
            'validation_candidate_sha256': sha(CANDIDATES),
            'torch_version': torch.__version__, 'numpy_version': np.__version__,
            'code_sha256': {path:sha(ROOT/path) for path in [
                'scripts/full_train.py', 'experiments/P1/run.py', 'scripts/model.py',
                'scripts/data.py', 'scripts/collate.py', 'scripts/checkpoint_io.py']},
        }
        expected_candidates = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']['sha256']
        if self.provenance['validation_candidate_sha256'] != expected_candidates:
            raise RuntimeError('Validation candidates differ from manifest')
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir/'config.json'
        if path.exists():
            if json.loads(path.read_text()) != self.provenance:
                raise RuntimeError('Prepared run config/data/environment has changed; use a new run directory')
        else:
            json_write(path, self.provenance)
        self.provenance_hash = sha(path)
        self.run_dir = run_dir


def new_state():
    return {'epoch_index': 0, 'next_sample_offset': 0, 'order': None,
            'optimizer_updates': 0, 'attempted_batches': 0, 'skipped_updates': 0,
            'epoch_loss_sum': 0.0, 'epoch_seconds': 0.0, 'epoch_memory': {},
            'validation_history': [], 'best_checkpoint': None, 'best_ndcg': None,
            'profiling': {}}


def save_state(path, model, optimizer, scaler, state, provenance_hash, device):
    logical_path = path
    if path.name == 'latest.pt':
        pointer = path.with_suffix('.json')
        previous_name = json.loads(pointer.read_text())['checkpoint'] if pointer.exists() else None
        next_name = 'latest_1.pt' if previous_name == 'latest_0.pt' else 'latest_0.pt'
        path = path.with_name(next_name)
    payload = {'model_state_dict': {k:v.detach().cpu() for k,v in model.state_dict().items()},
        'optimizer_state_dict': optimizer.state_dict(), 'gradient_scaler_state_dict': scaler.state_dict(),
        'rng_state': capture_rng_state(device), 'training_state': state,
        'provenance_sha256': provenance_hash}
    atomic_torch_save(payload, path, replace_existing=True)
    loaded = torch.load(path, map_location='cpu', weights_only=True)
    if loaded['training_state']['optimizer_updates'] != state['optimizer_updates'] or \
            loaded['gradient_scaler_state_dict'] != scaler.state_dict() or \
            not all(torch.equal(v.detach().cpu(), loaded['model_state_dict'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('Checkpoint read-back validation failed')
    sidecar = {'checkpoint': path.name, 'sha256': sha(path),
               'provenance_sha256': provenance_hash, 'epoch_index':state['epoch_index'],
               'next_sample_offset':state['next_sample_offset'],
               'optimizer_updates':state['optimizer_updates']}
    json_write(path.with_suffix('.json'), sidecar)
    if logical_path != path:
        # Publish only after the new slot and its sidecar are complete.
        # An interrupted save leaves the old pointer/slot usable.
        json_write(logical_path.with_suffix('.json'), sidecar)
    return sidecar['sha256']


def load_state(path, model, optimizer, scaler, provenance_hash, count, device):
    manifest = json.loads(path.with_suffix('.json').read_text())
    if path.name == 'latest.pt':
        if manifest['checkpoint'] not in ('latest_0.pt', 'latest_1.pt'):
            raise RuntimeError('Invalid latest checkpoint pointer')
        path = path.with_name(manifest['checkpoint'])
    if sha(path) != manifest['sha256'] or manifest['provenance_sha256'] != provenance_hash:
        raise RuntimeError('Resume checkpoint hash/provenance mismatch')
    saved = torch.load(path, map_location='cpu', weights_only=True)
    if saved['provenance_sha256'] != provenance_hash:
        raise RuntimeError('Resume payload provenance mismatch')
    state = saved['training_state']
    validate_sampler(state, count)
    if state['attempted_batches'] != state['optimizer_updates'] + state['skipped_updates']:
        raise RuntimeError('Resume counters are inconsistent')
    model.load_state_dict(saved['model_state_dict'], strict=True)
    optimizer.load_state_dict(saved['optimizer_state_dict'])
    scaler.load_state_dict(saved['gradient_scaler_state_dict'])
    steps = {int(v['step'].item()) for v in optimizer.state.values()}
    if state['optimizer_updates'] and (len(optimizer.state) != len(list(model.parameters())) or
                                      steps != {state['optimizer_updates']}):
        raise RuntimeError('Resume optimizer step/state mismatch')
    restore_rng(saved['rng_state'], device)
    return state


def evaluate_epoch(setup, model, checkpoint, checkpoint_hash, epoch):
    name = f'epoch_{epoch:04d}'
    summary_path = setup.run_dir/f'{name}_summary.json'
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary['checkpoint_sha256'] != checkpoint_hash or \
                summary['candidate_sha256'] != setup.provenance['validation_candidate_sha256']:
            raise RuntimeError('Existing evaluation belongs to another checkpoint or candidate set')
        return summary
    check_metrics()
    expected = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']
    reranker = SimpleNamespace(model=model.eval(), tokenizer=setup.tokenizer, device=torch.device('mps'))
    for _ in range(10):
        score_pairs(reranker, ['warmup'], ['warmup document'])
    torch.mps.synchronize()
    start = time.perf_counter()
    summary = evaluate_model(name, reranker, expected, setup.provenance['validation_candidate_sha256'],
        f'Four-layer scratch model; all parameters trained from E1 seed-42 random initialization on the complete cleaned training split; completed epoch {epoch}; FP16 autocast training.',
        extra_metadata={'model_id':'scratch_encoder_full_train',
            'checkpoint':str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256':checkpoint_hash,
            'test_split_scored':False}, result_dir=setup.run_dir)
    torch.mps.synchronize()
    summary['evaluation_seconds'] = time.perf_counter()-start
    json_write(summary_path, summary)
    return summary


def progress(setup, state, status, epochs, epoch_start, offset_start):
    done = state['next_sample_offset']
    elapsed_segment = time.perf_counter()-epoch_start
    rate = (done-offset_start)/elapsed_segment if done > offset_start else 0
    result = {'status':status, 'target_epochs':epochs, 'epoch':state['epoch_index']+1,
        'pairs_consumed_in_epoch':done, 'pairs_in_epoch':len(setup.dataset),
        'optimizer_updates':state['optimizer_updates'], 'skipped_updates':state['skipped_updates'],
        'mean_epoch_bce':state['epoch_loss_sum']/done if done else None,
        'segment_pairs_per_second':rate,
        'estimated_remaining_epoch_seconds':(len(setup.dataset)-done)/rate if rate else None,
        'scoring_split':'validation; test reserved', 'best_checkpoint':state['best_checkpoint'],
        'best_ndcg_at_10':state['best_ndcg']}
    json_write(setup.run_dir/'progress.json', result)
    with (setup.run_dir/'progress.jsonl').open('a') as log:
        log.write(json.dumps(result)+'\n')
    return result


def run(setup, args):
    device = torch.device('mps')
    model, optimizer, scaler = setup.fresh()
    latest = setup.run_dir/'latest.pt'
    if args.resume:
        state = load_state(latest, model, optimizer, scaler, setup.provenance_hash, len(setup.dataset), device)
    else:
        if latest.with_suffix('.json').exists() or any(setup.run_dir.glob('latest_*.pt')) or any(setup.run_dir.glob('epoch_*.pt')):
            raise RuntimeError('Training output exists; resume it or choose a new run directory')
        state = new_state()
    if args.resume and state.get('profile_enabled',False) != args.profile:
        raise RuntimeError('Resume with the same --profile setting as the original run')
    state['profile_enabled'] = args.profile
    stopped = False
    def request_stop(signum, frame):
        nonlocal stopped
        stopped = True
        print('Stop requested; saving after the current batch finishes.', flush=True)
    old_handlers = {s:signal.signal(s, request_stop) for s in (signal.SIGINT, signal.SIGTERM)}
    try:
        while state['epoch_index'] < args.epochs:
            count = len(setup.dataset)
            if state['order'] is None:
                state['order'] = torch.from_numpy(order_for_epoch(count, state['epoch_index']))
            validate_sampler(state, count)
            order = state['order'].numpy()
            offset_start = state['next_sample_offset']
            epoch_start = time.perf_counter()
            segment_base_seconds = state['epoch_seconds']
            profiler = EpochProfile(setup.run_dir,state['epoch_index']+1,math.ceil(count/SETTINGS['batch_size']),
                                    state['profiling'].setdefault(str(state['epoch_index']+1), {})) if args.profile else None
            model.train()
            print(f'epoch={state["epoch_index"]+1}/{args.epochs} start_pair={offset_start:,}/{count:,}', flush=True)
            while state['next_sample_offset'] < count:
                offset = state['next_sample_offset']
                indices = order[offset:offset+SETTINGS['batch_size']]
                batch_number = offset // SETTINGS['batch_size'] + 1
                stages = profiler.stages(batch_number) if profiler else NoProfile()
                result = step(setup, model, optimizer, scaler, indices, stages)
                if profiler:
                    profiler.after_batch(batch_number)
                state['next_sample_offset'] += len(indices)
                state['attempted_batches'] += 1
                state['skipped_updates'] += int(result['skipped'])
                state['optimizer_updates'] += int(not result['skipped'])
                state['epoch_loss_sum'] += result['loss']*len(indices)
                state['epoch_seconds'] = segment_base_seconds + time.perf_counter()-epoch_start
                for key,value in result['memory'].items():
                    state['epoch_memory'][key] = max(state['epoch_memory'].get(key,0),value)
                last = state['next_sample_offset'] == count
                if stopped and profiler:
                    profiler.close_trace(batch_number)
                if state['attempted_batches'] % args.log_every == 0 or last:
                    if profiler:
                        profiler.flush()
                    p = progress(setup,state,'training',args.epochs,epoch_start,offset_start)
                    eta = p['estimated_remaining_epoch_seconds']
                    print(f'epoch={state["epoch_index"]+1} pairs={state["next_sample_offset"]:,}/{count:,} '
                          f'BCE={p["mean_epoch_bce"]:.4f} updates={state["optimizer_updates"]} '
                          f'skips={state["skipped_updates"]} ETA={eta/60:.1f}min' if eta is not None else
                          f'epoch={state["epoch_index"]+1} pairs={state["next_sample_offset"]:,}/{count:,}', flush=True)
                if state['attempted_batches'] % args.checkpoint_every == 0 or last or stopped:
                    save_state(latest,model,optimizer,scaler,state,setup.provenance_hash,device)
                if stopped:
                    if profiler:
                        profiler.flush()
                    progress(setup,state,'stopped_resumable',args.epochs,epoch_start,offset_start)
                    return
            if profiler:
                profiler.flush()
            epoch = state['epoch_index']+1
            checkpoint = setup.run_dir/f'epoch_{epoch:04d}.pt'
            if checkpoint.exists():
                checkpoint_hash = json.loads(checkpoint.with_suffix('.json').read_text())['sha256']
                if sha(checkpoint) != checkpoint_hash:
                    raise RuntimeError('Epoch checkpoint differs from its hash')
                saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
                if saved['training_state']['epoch_index'] != state['epoch_index'] or \
                        saved['training_state']['next_sample_offset'] != count or \
                        not all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) for k,v in model.state_dict().items()):
                    raise RuntimeError('Existing epoch checkpoint differs from resumed weights/state')
            else:
                checkpoint_hash = save_state(checkpoint,model,optimizer,scaler,state,setup.provenance_hash,device)
            progress(setup,state,'validating',args.epochs,epoch_start,offset_start)
            summary = evaluate_epoch(setup,model,checkpoint,checkpoint_hash,epoch)
            result = {'epoch':epoch,'training_seconds':state['epoch_seconds'],
                'training_pairs':count,'mean_bce':state['epoch_loss_sum']/count,
                'optimizer_updates_total':state['optimizer_updates'], 'skipped_updates_total':state['skipped_updates'],
                'ndcg_at_10':summary['ndcg_at_10'],'mrr_at_10':summary['mrr_at_10'],
                'checkpoint':str(checkpoint.relative_to(ROOT)), 'checkpoint_sha256':checkpoint_hash,
                'memory':state['epoch_memory'],
                'bounded_profiling_enabled':args.profile,
                'timing_note':'includes periodic checkpoints, logging, and bounded profiling if enabled; excludes validation'}
            json_write(setup.run_dir/f'epoch_{epoch:04d}_training.json', result)
            state['validation_history'].append(result)
            if state['best_ndcg'] is None or summary['ndcg_at_10'] > state['best_ndcg']:
                state['best_ndcg'] = summary['ndcg_at_10']
                state['best_checkpoint'] = str(checkpoint.relative_to(ROOT))
                json_write(setup.run_dir/'best.json', result)
            state.update(epoch_index=epoch,next_sample_offset=0,order=None,epoch_loss_sum=0.0,
                         epoch_seconds=0.0,epoch_memory={})
            save_state(latest,model,optimizer,scaler,state,setup.provenance_hash,device)
            json_write(setup.run_dir/'history.json',state['validation_history'])
            if stopped:
                json_write(setup.run_dir/'progress.json',{'status':'stopped_resumable','completed_epochs':epoch,
                    'best_checkpoint':state['best_checkpoint'],'best_ndcg_at_10':state['best_ndcg']})
                return
        json_write(setup.run_dir/'progress.json',{'status':'complete','completed_epochs':state['epoch_index'],
            'best_checkpoint':state['best_checkpoint'],'best_ndcg_at_10':state['best_ndcg'],
            'optimizer_updates':state['optimizer_updates'],'skipped_updates':state['skipped_updates']})
        print(f'Training complete: best nDCG={state["best_ndcg"]}; checkpoint={state["best_checkpoint"]}',flush=True)
    finally:
        for s, handler in old_handlers.items():
            signal.signal(s, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--epochs', type=int, default=1, help='Total desired epochs, including completed epochs when resuming')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Collect early/middle/late stage and CPU profiling windows')
    parser.add_argument('--launch', action='store_true', help='Launch a detached training process and write its PID/log path')
    parser.add_argument('--preflight', action='store_true', help='Verify full data/model/tokenizer and prepare config; no training')
    parser.add_argument('--checkpoint-every', type=int, default=1000, help='Attempted batches between resumable saves')
    parser.add_argument('--log-every', type=int, default=100)
    args = parser.parse_args()
    if min(args.epochs,args.checkpoint_every,args.log_every) < 1:
        parser.error('epochs and intervals must be positive')
    args.run_dir = args.run_dir.resolve()
    if not args.run_dir.is_relative_to(ROOT):
        parser.error('run-dir must be inside this repository')
    if args.launch:
        if args.preflight:
            parser.error('--launch and --preflight cannot be combined')
        args.run_dir.mkdir(parents=True,exist_ok=True)
        command = [sys.executable,'-u','-m','scripts.full_train','--run-dir',str(args.run_dir),
                   '--epochs',str(args.epochs),'--checkpoint-every',str(args.checkpoint_every),
                   '--log-every',str(args.log_every)]
        if args.profile:
            command.append('--profile')
        if args.resume:
            command.append('--resume')
        log_path = args.run_dir/'training.log'
        with log_path.open('a') as log:
            child = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        record = {'pid':child.pid,'command':command,'log':str(log_path),
                  'status':'launched; inspect progress.json and training.log for actual state'}
        caffeinate = shutil.which('caffeinate')
        if caffeinate:
            awake = subprocess.Popen([caffeinate,'-i','-w',str(child.pid)],
                stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
            record['caffeinate_pid'] = awake.pid
        json_write(args.run_dir/'launch.json',record)
        print(json.dumps(record,indent=2))
        return
    setup = FullSetup(args.run_dir)
    if args.preflight:
        batches = math.ceil(len(setup.dataset)/SETTINGS['batch_size'])
        model, optimizer, scaler = setup.fresh()
        # Match the training attention path: MPS SDPA dropout requires autograd.
        # This forward-only check performs no backward or optimizer update.
        batch = setup.collator([setup.dataset[i] for i in order_for_epoch(len(setup.dataset),0)[:16]])
        model.train()
        with torch.autocast('mps',dtype=torch.float16):
            logits = model(**batch['inputs'].to('mps'))
        if logits.shape != (16,) or not bool(torch.isfinite(logits).all()):
            raise RuntimeError('Full-data forward preflight failed')
        record = {'status':'ready; training not started','training_pairs':len(setup.dataset),
            'batches_per_epoch':batches,'last_batch_pairs':len(setup.dataset)%SETTINGS['batch_size'],
            'all_parameters_trainable':all(p.requires_grad for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters()),
            'mps_grad_scaler_enabled':scaler.is_enabled(),'sample_forward_finite':True,
            'checkpoint_every_batches':args.checkpoint_every,'target_epochs':args.epochs,
            'bounded_profiling_enabled':args.profile,
            'estimated_epoch_minutes_from_pilot':51.1}
        json_write(args.run_dir/'preflight.json',record)
        print(json.dumps(record,indent=2))
        return
    with (args.run_dir/'train.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another trainer is already running in this directory')
        lock.seek(0)
        lock.truncate()
        lock.write(str(os.getpid()) + '\n')
        lock.flush()
        run(setup,args)


if __name__ == '__main__':
    main()
