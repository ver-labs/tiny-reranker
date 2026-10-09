"""Batch-size and HTTP concurrency sweeps of epoch 3; validation data only."""
import asyncio
import itertools
import json
import statistics
import subprocess
import sys

import numpy as np

from scripts.evaluate_initial import CANDIDATES
from .backend import HERE, ROOT, REFERENCE, CHECKPOINT, sha256, write_json
from .benchmark_serving import benchmark

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128]
CONCURRENCIES = [1, 2, 4, 8, 16, 32]
REPEATS = 3


def summarize(values):
    return {'median': statistics.median(values),
            'q25': float(np.percentile(values,25)),
            'q75': float(np.percentile(values,75)), 'values': values}


async def run():
    quality = json.loads((HERE/'quality.json').read_text())
    if quality['checkpoint_sha256'] != sha256(CHECKPOINT) or quality['changed_query_rankings'] != 0:
        raise RuntimeError('Verified epoch-3 quality report required before timing')
    direct = []
    # Separate processes ensure only one backend has a live GPU model.
    # Reverse both backend and size order in round two to reduce order bias.
    for round_index, order in enumerate((('pytorch','mlx'), ('mlx','pytorch')),1):
        for name in order:
            output = f'sweep_direct_round_{round_index}_{name}.json'
            command = [sys.executable,'-u','-m','experiments.S1.benchmark',
                       '--backend',name,'--batch-sizes',*[str(s) for s in BATCHES],
                       '--repeats',str(REPEATS),'--output-name',output]
            if round_index == 2:
                command.append('--reverse')
            print(f'DIRECT round={round_index} backend={name}',flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
            report = json.loads((HERE/output).read_text())
            if report['checkpoint_sha256'] != quality['checkpoint_sha256']:
                raise RuntimeError('Direct benchmark checkpoint mismatch')
            direct.append({'round':round_index, **report})
    if len({r['sample_sha256'] for r in direct}) != 1:
        raise RuntimeError('Backend samples differ')
    with CANDIDATES.open() as f:
        rows = [json.loads(line) for line in itertools.islice(f,128)]
    with (REFERENCE/'epoch_0003_rankings.jsonl').open() as f:
        refs = [json.loads(line) for line in itertools.islice(f,128)]
    if [r['query_id'] for r in rows] != [r['query_id'] for r in refs]:
        raise RuntimeError('HTTP reference sample mismatch')
    expected = {r['query_id']:[d['document_id'] for d in r['ranking']] for r in refs}
    payloads = [{'request_id':r['query_id'],'query':r['query'],'top_k':len(r['candidates']),
                 'documents':[{'document_id':c['document_id'],'text':c['text']} for c in r['candidates']]}
                for r in rows]
    if any(len(p['documents']) != 2 for p in payloads):
        raise RuntimeError('Concurrency sweep requires two documents per request')
    serving = []
    for round_index, order in enumerate((('pytorch','mlx'),('mlx','pytorch')),1):
        for name in order:
            print(f'HTTP round={round_index} backend={name}',flush=True)
            cs = tuple(CONCURRENCIES if round_index == 1 else reversed(CONCURRENCIES))
            report = await benchmark(name,payloads,expected,8,cs,REPEATS,
                                     log_prefix=f'sweep_round_{round_index}_')
            serving.append({'round':round_index, **report})
            write_json(HERE/'sweep_http_raw.json',{'rounds':serving})
    batch_summary, concurrency_summary = [], []
    for size in BATCHES:
        row = {'batch_size':size}
        for name in ('pytorch','mlx'):
            reports = [x for r in direct if r['backend']==name for x in r['results']
                       if x['pair_batch_size']==size]
            row[name] = {}
            for mode in ('forward','scoring'):
                row[name][mode] = {
                    'pairs_per_second':summarize([r['pairs']/t for r in reports for t in r[mode]['repeat_seconds']]),
                    'median_batch_ms':summarize([v for r in reports for v in r[mode]['repeat_median_ms']])}
        batch_summary.append(row)
    for c in CONCURRENCIES:
        row = {'concurrency':c}
        for name in ('pytorch','mlx'):
            reports = [x for r in serving if r['backend']==name for x in r['results'][str(c)]]
            row[name] = {k:summarize([r[k] for r in reports])
                         for k in ('requests_per_second','median_ms','p95_ms','median_queue_ms','median_processing_ms')}
            row[name]['successful_requests'] = sum(r['successful_requests'] for r in reports)
            row[name]['failures'] = sum(r['failures'] for r in reports)
        concurrency_summary.append(row)
    result = {'checkpoint':str(CHECKPOINT.relative_to(ROOT)),
              'checkpoint_sha256':quality['checkpoint_sha256'],
              'dataset_split':'validation', 'test_split_touched':False,
              'sample':'first 128 fixed validation queries, 256 query-document pairs',
              'sample_sha256':direct[0]['sample_sha256'],
              'candidate_sha256':sha256(CANDIDATES),
              'precision':'FP32; MLX_ENABLE_TF32=0; both backends use Apple M5 GPU',
              'repetitions_per_point':6, 'rounds':2,
              'batch_scope':'complete scoring includes tokenization, transfer and CPU-readable scores; separate forward data retained',
              'http_scope':'loopback HTTP, two documents per request, one worker, queue capacity 32, no cross-request batching',
              'variability':'points are medians of six repeats; shading is the interquartile range, not a confidence interval',
              'latency':'direct: median batch latency; HTTP: client-observed request latency after acquiring a concurrency slot',
              'limitations':['Local workload; timing varies with token lengths, background activity and thermal state.',
                             'Increasing HTTP concurrency does not increase the model batch size.',
                             'HTTP results do not measure reranking 50 or 100 documents per query.'],
              'batch':batch_summary,'concurrency':concurrency_summary}
    write_json(HERE/'sweep_results.json',result)
    plot(result)


def plot(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors = {'pytorch':'#2563eb','mlx':'#d97706'}
    for kind, points, key, title in (
        ('batch',result['batch'],'batch_size','Epoch 3: increasing pair batch size'),
        ('concurrency',result['concurrency'],'concurrency','Epoch 3: increasing HTTP concurrency')):
        fig, axes = plt.subplots(1,2,figsize=(12,5),layout='constrained')
        xs = [r[key] for r in points]
        for name in ('pytorch','mlx'):
            if kind == 'batch':
                throughput = [r[name]['scoring']['pairs_per_second'] for r in points]
                latency = [r[name]['scoring']['median_batch_ms'] for r in points]
            else:
                throughput = [r[name]['requests_per_second'] for r in points]
                latency = [r[name]['median_ms'] for r in points]
            label = 'PyTorch / MPS' if name=='pytorch' else 'MLX / Metal'
            for ax,values in zip(axes,(throughput,latency)):
                ax.plot(xs,[v['median'] for v in values],'-o',color=colors[name],label=label)
                ax.fill_between(xs,[v['q25'] for v in values],[v['q75'] for v in values],color=colors[name],alpha=.13)
            if kind == 'concurrency':
                axes[1].plot(xs,[r[name]['p95_ms']['median'] for r in points],'--',color=colors[name],label=f'{label} p95')
        for ax in axes:
            ax.set_xscale('log',base=2)
            ax.set_xticks(xs,labels=[str(x) for x in xs])
            ax.set_xlabel('Pairs per model batch' if kind=='batch' else 'Concurrent HTTP requests')
            ax.grid(alpha=.2)
            ax.legend(fontsize=9)
            ax.set_ylim(bottom=0)
        axes[0].set_ylabel('Scored pairs / second' if kind=='batch' else 'Requests / second')
        axes[1].set_ylabel('Median batch latency (ms)' if kind=='batch' else 'Request latency (ms); solid = median')
        fig.suptitle(title+'\nValidation sample · FP32 · Apple M5 GPU · 6 repeats; shading = IQR',fontsize=13)
        for ext in ('png','svg'):
            fig.savefig(HERE/f'sweep_{kind}.{ext}',dpi=180)
        plt.close(fig)
    print('Saved sweep_results.json, sweep_batch.png/svg, sweep_concurrency.png/svg',flush=True)


if __name__ == '__main__':
    asyncio.run(run())
