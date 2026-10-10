"""Summarize quality and paired bootstrap differences; plot matched 128-token paths."""
import gzip
import json
import os
from pathlib import Path
import tempfile
import numpy as np

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'reranker-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .metrics import metrics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def interval(difference):
    rng = np.random.default_rng(42)
    indices = rng.integers(0, len(difference), size=(2000, len(difference)))
    boot = difference[indices].mean(axis=1)
    return {'mean_difference': float(difference.mean()),
            '95_percent_interval': np.percentile(boot, [2.5, 97.5]).tolist(), 'resamples': 2000, 'seed': 42}


def main():
    report = json.loads((HERE / 'quality.json').read_text())
    summary = {'parameters': report['parameters'], 'datasets': {}, 'paired_ndcg_differences': {}}
    for name, result in report['datasets'].items():
        summary['datasets'][name] = {path: value['metrics'] for path, value in result['paths'].items()}
    for name in ('scifact', 'nfcorpus'):
        ndcg = {}; qrels = {}; query_order = None
        for path in ('scratch_128', 'tinybert_128', 'minilm_128', 'tinybert_512', 'minilm_512'):
            with gzip.open(HERE / f'{name}_{path}_rankings.jsonl.gz', 'rt') as f: rows = [json.loads(line) for line in f]
            order = [r['query_id'] for r in rows]
            if query_order is not None and order != query_order: raise RuntimeError('Bootstrap query alignment mismatch')
            query_order = order
            ndcg[path] = np.asarray([r['metrics']['ndcg_at_10'] for r in rows])
            for r in rows: qrels[r['query_id']] = r['qrels']
        with (ROOT / 'data/beir' / name / 'bm25_top100.jsonl').open() as f:
            baseline = {r['query_id']: metrics(r['document_ids'], qrels[r['query_id']])['ndcg_at_10'] for r in map(json.loads, f)}
        ndcg['bm25'] = np.asarray([baseline[qid] for qid in query_order])
        summary['datasets'][name]['bm25'] = report['retrieval_manifest']['datasets'][name]['bm25_metrics']
        summary['paired_ndcg_differences'][name] = {f'{a}_minus_{b}': interval(ndcg[a] - ndcg[b]) for a, b in (
            ('tinybert_128', 'scratch_128'), ('minilm_128', 'tinybert_128'), ('tinybert_512', 'bm25'))}
    (HERE / 'comparison_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), layout='constrained')
    paths = ('bm25', 'scratch_128', 'tinybert_128', 'minilm_128')
    labels = ['BM25', 'Ours\n4.72M', 'TinyBERT\n4.39M', 'MiniLM\n15.62M']
    colors = ['#777777', '#bf6552', '#2a8193', '#6b77a8']
    for name, ax in zip(('scifact', 'nfcorpus'), axes):
        values = [summary['datasets'][name][p]['ndcg_at_10'] for p in paths]
        bars = ax.bar(labels, values, color=colors, width=.65)
        ax.bar_label(bars, labels=[f'{v:.3f}' for v in values], padding=4)
        ax.set_ylim(0, .8); ax.set_ylabel('nDCG@10'); ax.set_title(f'{name.title()} · full test split')
        ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.suptitle('Same BM25 top-100 candidates · rerankers capped at 128 tokens', fontsize=13)
    fig.savefig(HERE / 'quality_comparison.png', dpi=170)
    print(json.dumps(summary['paired_ndcg_differences'], indent=2))


if __name__ == '__main__': main()
