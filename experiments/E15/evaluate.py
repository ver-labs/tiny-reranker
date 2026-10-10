"""Fixed scratch/TinyBERT/MiniLM quality comparison on full candidate datasets."""
import gzip
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiments.E14.benchmark import environment, load_fixture
from .prepare import ROOT, HERE, sha256, load
from .metrics import metrics, assert_trec_parity

BATCH = 128
PATHS = ('scratch_128', 'tinybert_128', 'minilm_128', 'tinybert_512', 'minilm_512')


def write(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def dataset(name, retrieval):
    if name == 'constructed_test':
        manifest = json.loads((ROOT / 'results/evaluation_candidates.json').read_text())['splits']['test']
        path = ROOT / manifest['candidate_file']
        if sha256(path) != manifest['sha256']: raise RuntimeError('Test file hash mismatch')
        with path.open() as f: source = [json.loads(line) for line in f]
        rows = []
        for r in source:
            # The file was previously fully audited; enforce held-out query IDs here too.
            if int(r['query_id'][:16], 16) % 10 != 9: raise RuntimeError('Non-test query')
            rows.append({'query_id': r['query_id'], 'query': r['query'],
                'document_ids': [c['document_id'] for c in r['candidates']],
                'texts': [c['text'] for c in r['candidates']],
                'qrels': {c['document_id']: c['label'] for c in r['candidates']}})
        if len(rows) != manifest['evaluable_queries'] or sum(len(r['document_ids']) for r in rows) != manifest['candidates']:
            raise RuntimeError('Constructed test count mismatch')
        return rows, {'candidate_sha256': sha256(path), 'queries': len(rows), 'pairs': manifest['candidates'],
            'limitation': 'Constructed mined-triplet lists; pretrained baselines trained on MS MARCO, unseen queries not established.'}
    corpus, queries, qrels = load(name)
    path = ROOT / 'data/beir' / name / 'bm25_top100.jsonl'
    expected = retrieval['datasets'][name]
    if sha256(path) != expected['candidate_file_sha256']: raise RuntimeError('BM25 candidate hash mismatch')
    rows = []
    with path.open() as f:
        for line in f:
            r = json.loads(line); qid = r['query_id']; ids = r['document_ids']
            if len(ids) != 100 or len(set(ids)) != 100: raise RuntimeError('Invalid retrieved candidates')
            rows.append({'query_id': qid, 'query': queries[qid], 'document_ids': ids,
                'texts': [(corpus[d].get('title', '') + ' ' + corpus[d]['text']).strip() for d in ids],
                'qrels': qrels[qid]})
    if {r['query_id'] for r in rows} != set(qrels) or len(rows) != len(qrels):
        raise RuntimeError('Missing or duplicate official test query')
    return rows, {'queries': len(rows), 'pairs': sum(len(r['document_ids']) for r in rows),
        'candidate_sha256': sha256(path), 'dataset_files': json.loads((HERE / 'datasets.json').read_text())[name],
        'retrieval': expected, 'limitation': 'BM25 top-100; complete official qrels; no relevance injection; not the full BEIR suite.'}


def load_baseline(name):
    manifest_path = HERE / ('baseline_manifest.json' if name == 'tinybert' else 'minilm_manifest.json')
    manifest = json.loads(manifest_path.read_text())
    folder = ROOT / ('artifacts/e15_' + name)
    for relative, expected in manifest['file_sha256'].items():
        if sha256(folder / relative) != expected: raise RuntimeError('Baseline asset hash mismatch')
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(folder, local_files_only=True,
        trust_remote_code=False).cuda().eval()
    if model.config.num_labels != 1: raise RuntimeError('Expected a single relevance logit')
    return model, tokenizer, manifest


def evaluate(rows, path, model, tokenizer):
    length = int(path.rsplit('_', 1)[1]); scratch = path.startswith('scratch')
    audit = HERE / f'{CURRENT_DATASET}_{path}_rankings.jsonl.gz'
    temporary = audit.with_suffix('.gz.tmp')
    totals = {k: 0. for k in metrics([], rows[0]['qrels'])}; runs = {}; per_query = []
    total_seconds = 0.; pairs = count = 0; warmed = False
    with gzip.open(temporary, 'wt') as output, torch.inference_mode():
        for start in range(0, len(rows), 256):
            buffered = rows[start:start+256]
            flat = [(r['query'], text) for r in buffered for text in r['texts']]
            scores = []
            for offset in range(0, len(flat), BATCH):
                chunk = flat[offset:offset+BATCH]
                if not warmed:
                    inputs = tokenizer([q for q, _ in chunk], [d for _, d in chunk], padding=True,
                        truncation='longest_first', max_length=length, return_tensors='pt')
                    inputs = {k: v.cuda() for k, v in inputs.items()}
                    for _ in range(10):
                        result = model(**inputs, kernel_mode='original') if scratch else model(**inputs).logits[:, 0]
                        result.cpu()
                    torch.cuda.synchronize(); warmed = True
                torch.cuda.synchronize(); begin = time.perf_counter()
                inputs = tokenizer([q for q, _ in chunk], [d for _, d in chunk], padding=True,
                    truncation='longest_first', max_length=length, return_tensors='pt')
                inputs = {k: v.cuda() for k, v in inputs.items()}
                result = model(**inputs, kernel_mode='original') if scratch else model(**inputs).logits[:, 0]
                values = result.cpu().tolist(); total_seconds += time.perf_counter() - begin
                if len(values) != len(chunk) or not all(np.isfinite(values)): raise RuntimeError('Invalid model scores')
                scores.extend(values)
            offset = 0
            for row in buffered:
                n = len(row['document_ids']); values = scores[offset:offset+n]; ids = row['document_ids']
                order = sorted(range(n), key=lambda i: (-values[i], ids[i]))
                ranked = [ids[i] for i in order]; runs[row['query_id']] = ranked
                values_metrics = metrics(ranked, row['qrels'])
                for key, value in values_metrics.items(): totals[key] += value
                per_query.append(values_metrics['ndcg_at_10'])
                output.write(json.dumps({'query_id': row['query_id'], 'candidate_ids': ids,
                    'qrels': row['qrels'], 'scores': values, 'ranking': ranked, 'metrics': values_metrics}) + '\n')
                offset += n; pairs += n; count += 1
            if count % 4096 == 0 or count == len(rows):
                print(f'{CURRENT_DATASET} {path}: {count:,}/{len(rows):,} queries', flush=True)
    if count != len(rows) or len(runs) != len(rows): raise RuntimeError('Missing model rankings')
    assert_trec_parity({r['query_id']: r['qrels'] for r in rows}, runs)
    temporary.replace(audit)
    return {'queries': count, 'pairs': pairs, 'max_length': length,
        'metrics': {k: value / count for k, value in totals.items()},
        'quality_pass_scoring_seconds': total_seconds, 'timing_limit': 'Single quality pass; not a repeated latency benchmark.',
        'rankings_file': audit.name, 'rankings_sha256': sha256(audit),
        'trec_eval_parity_verified': True}, per_query


def main():
    global CURRENT_DATASET
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    scratch, scratch_tokenizer, _, checkpoint, _ = load_fixture()
    tiny, tiny_tokenizer, tiny_manifest = load_baseline('tinybert')
    mini, mini_tokenizer, mini_manifest = load_baseline('minilm')
    models = {'scratch': (scratch, scratch_tokenizer), 'tinybert': (tiny, tiny_tokenizer), 'minilm': (mini, mini_tokenizer)}
    retrieval = json.loads((HERE / 'retrieval_manifest.json').read_text())
    report = {'status': 'running', 'environment': environment(), 'checkpoint_sha256': sha256(checkpoint),
        'pair_batch_size': BATCH, 'precision': 'FP32; TF32 disabled', 'attention_fastpath': False,
        'parameters': {name: sum(p.numel() for p in model.parameters()) for name, (model, _) in models.items()},
        'models': {'scratch': {'checkpoint': 'runs/full_train/epoch_0003.pt', 'max_length': 128},
                   'tinybert': tiny_manifest, 'minilm': mini_manifest},
        'retrieval_manifest': retrieval, 'datasets': {},
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in [Path(__file__), HERE/'metrics.py', HERE/'prepare.py',
            ROOT/'experiments/E14/model.py', ROOT/'experiments/E14/benchmark.py', ROOT/'scripts/model.py']},
        'metric_definition': 'trec_eval-compatible linear-gain graded nDCG@10; MRR@10 treats grade>0 as relevant; IDCG uses all qrels, including unretrieved positives.',
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'transformers', 'rank-bm25', 'pytrec-eval-terrier']},
        'limitations': ['TinyBERT is the direct-size comparison; MiniLM has substantially more parameters.',
            'Main comparisons use 128 tokens; pretrained 512-token variants are separate longer-context results.',
            'Both published baselines were trained on MS MARCO; our constructed test queries are not established as unseen for them.',
            'Two full BEIR test datasets do not establish a globally best reranker.',
            'No retraining, calibration, checkpoint selection or parameter tuning on test results.']}
    for CURRENT_DATASET in ('constructed_test', 'scifact', 'nfcorpus'):
        rows, manifest = dataset(CURRENT_DATASET, retrieval); output = {'manifest': manifest, 'paths': {}}
        ndcg = {}
        for path in PATHS:
            name = path.rsplit('_', 1)[0]
            output['paths'][path], ndcg[path] = evaluate(rows, path, *models[name])
            print(CURRENT_DATASET, path, output['paths'][path]['metrics'], flush=True)
            report['datasets'][CURRENT_DATASET] = output
            write(HERE / 'quality_progress.json', report)
        if CURRENT_DATASET == 'constructed_test':
            previous = json.loads((ROOT / 'experiments/E14/test_quality.json').read_text())['metrics']['original']
            current = output['paths']['scratch_128']['metrics']
            if current['mrr_at_10'] != previous['mrr_at_10'] or current['ndcg_at_10'] != previous['ndcg_at_10']:
                raise RuntimeError('Scratch constructed-test metrics changed')
        else:
            rng = np.random.default_rng(42); indices = rng.integers(0, len(rows), size=(2000, len(rows)))
            output['paired_bootstrap_ndcg_difference_vs_scratch'] = {}
            for path in PATHS[1:]:
                difference = np.asarray(ndcg[path]) - np.asarray(ndcg['scratch_128'])
                boot = difference[indices].mean(axis=1)
                output['paired_bootstrap_ndcg_difference_vs_scratch'][path] = {
                    'mean_difference': float(difference.mean()), '95_percent_interval': np.percentile(boot,[2.5,97.5]).tolist(),
                    'resamples': 2000, 'seed': 42}
    report['status'] = 'complete'; write(HERE / 'quality.json', report)


if __name__ == '__main__': main()
