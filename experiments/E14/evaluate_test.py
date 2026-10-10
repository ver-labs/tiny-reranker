"""Full fixed held-out ranking evaluation; no tuning or training on test queries."""
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import time

import torch
from scripts.evaluate_initial import check_metrics, ranking_metrics
from .benchmark import ROOT, HERE, environment, load_fixture, save, sha256

MODES = ('original', 'cuda', 'triton')
BATCH_SIZE = 128


def main():
    check_metrics()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    model, tokenizer, _, checkpoint, _ = load_fixture()
    manifest = json.loads((ROOT / 'results/evaluation_candidates.json').read_text())
    expected = manifest['splits']['test']
    candidates = ROOT / expected['candidate_file']
    if sha256(candidates) != expected['sha256']: raise RuntimeError('Test candidate hash mismatch')
    training = json.loads(checkpoint.with_name('config.json').read_text())
    split_file = ROOT / training['split_manifest']['path']
    if sha256(split_file) != training['split_manifest']['sha256']: raise RuntimeError('Training split manifest mismatch')
    split = json.loads(split_file.read_text())
    if split['train_buckets'] != list(range(8)) or split['validation_buckets'] != [8] or split['test_buckets'] != [9]:
        raise RuntimeError('Unexpected query split rule')
    plan = {'checkpoint_sha256': sha256(checkpoint), 'candidate_sha256': expected['sha256'],
        'dataset_split': 'test', 'test_manifest': expected, 'pair_batch_size': BATCH_SIZE,
        'max_length': model.config.max_length, 'precision': 'FP32; TF32 disabled',
        'attention_fastpath': False, 'triton_num_warps': 4,
        'paths': {'original': 'normal PyTorch encoder, fused fast path disabled',
                  'cuda': 'handwritten CUDA C++ port of E11 Metal algorithm',
                  'triton': 'custom Triton residual + LayerNorm'},
        'selection_rule': 'Epoch 3 and implementation settings fixed before this evaluation; no test-driven tuning',
        'environment': environment(), 'split_manifest_sha256': sha256(split_file),
        'code_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in (
            Path(__file__), HERE / 'model.py', HERE / 'kernel.py', HERE / 'residual_layer_norm.cu',
            HERE / 'residual_layer_norm.so', ROOT / 'experiments/E13/kernel.py',
            ROOT / 'experiments/E13/triton_impl.py', ROOT / 'scripts/model.py',
            ROOT / 'scripts/evaluate_initial.py', HERE / 'benchmark.py')}}
    save(HERE / 'test_plan.json', plan)
    accum = {mode: {'mrr_total': 0., 'ndcg_total': 0., 'changed_rankings_vs_pytorch': 0,
        'max_score_error_vs_pytorch': 0., 'absolute_score_error_sum': 0.,
        'pairs_outside_score_tolerance': 0, 'forward_wall_seconds': 0.} for mode in MODES}
    count = pair_count = missing = 0
    seen = set(); warmed = False; batch_index = 0
    audit = HERE / 'full_test_scores.jsonl.gz'; temporary = audit.with_suffix('.gz.tmp')
    differences = []
    begin = time.perf_counter()
    with candidates.open() as source, gzip.open(temporary, 'wt', encoding='utf-8') as output, torch.inference_mode():
        while True:
            rows = [json.loads(line) for line in itertools.islice(source, 512)]
            if not rows: break
            flat = []
            for row in rows:
                qid = hashlib.sha256(' '.join(row['query'].casefold().split()).encode()).hexdigest()
                if qid != row['query_id'] or int(qid[:16], 16) % 10 != 9 or qid in seen:
                    raise RuntimeError('Duplicate, invalid or non-test query')
                seen.add(qid)
                docs = row['candidates']
                if not docs or len({c['document_id'] for c in docs}) != len(docs):
                    raise RuntimeError('Empty or duplicate candidate list')
                for c in docs:
                    if c['label'] not in (0, 1) or hashlib.sha256(c['text'].encode()).hexdigest() != c['document_id']:
                        raise RuntimeError('Invalid candidate label or document ID')
                    flat.append((row['query'], c['text']))
            values = {mode: [] for mode in MODES}
            for start in range(0, len(flat), BATCH_SIZE):
                chunk = flat[start:start + BATCH_SIZE]
                inputs = tokenizer([q for q, _ in chunk], [d for _, d in chunk], padding=True,
                    truncation='longest_first', max_length=model.config.max_length,
                    return_attention_mask=True, return_token_type_ids=True, return_tensors='pt')
                inputs = {k: v.cuda() for k, v in inputs.items()}
                if not warmed:
                    for mode in MODES:
                        for _ in range(10): model(**inputs, kernel_mode=mode, num_warps=4).cpu()
                    torch.cuda.synchronize(); warmed = True
                shift = batch_index % len(MODES)
                chunk_outputs = {}
                for mode in MODES[shift:] + MODES[:shift]:
                    torch.cuda.synchronize(); t = time.perf_counter()
                    scores = model(**inputs, kernel_mode=mode, num_warps=4).cpu()
                    accum[mode]['forward_wall_seconds'] += time.perf_counter() - t
                    if scores.shape != (len(chunk),) or not torch.isfinite(scores).all():
                        raise RuntimeError(f'Invalid score output: {mode}')
                    chunk_outputs[mode] = scores; values[mode].extend(scores.tolist())
                reference = chunk_outputs['original']
                for mode in ('cuda', 'triton'):
                    error = (chunk_outputs[mode] - reference).abs()
                    accum[mode]['max_score_error_vs_pytorch'] = max(accum[mode]['max_score_error_vs_pytorch'], float(error.max()))
                    accum[mode]['absolute_score_error_sum'] += float(error.double().sum())
                    accum[mode]['pairs_outside_score_tolerance'] += int((error > (1e-5 + 1e-4 * reference.abs())).sum())
                batch_index += 1
            offset = 0
            for row in rows:
                n = len(row['candidates']); orders = {}; query_scores = {}; query_metrics = {}
                missing += int(not any(c['label'] for c in row['candidates']))
                for mode in MODES:
                    query_scores[mode] = values[mode][offset:offset+n]
                    ranked = sorted(zip(row['candidates'], query_scores[mode]), key=lambda x: (-x[1], x[0]['document_id']))
                    orders[mode] = [c['document_id'] for c, _ in ranked]
                    mrr, ndcg = ranking_metrics([c['label'] for c, _ in ranked])
                    accum[mode]['mrr_total'] += mrr; accum[mode]['ndcg_total'] += ndcg
                    query_metrics[mode] = {'mrr_at_10': mrr, 'ndcg_at_10': ndcg}
                changed = [mode for mode in ('cuda', 'triton') if orders[mode] != orders['original']]
                for mode in changed: accum[mode]['changed_rankings_vs_pytorch'] += 1
                if changed: differences.append({'query_id': row['query_id'], 'changed_paths': changed})
                output.write(json.dumps({'query_id': row['query_id'],
                    'candidate_ids': [c['document_id'] for c in row['candidates']],
                    'labels': [c['label'] for c in row['candidates']],
                    'scores': query_scores, 'rankings': orders, 'metrics': query_metrics}) + '\n')
                offset += n; count += 1; pair_count += n
            if count % 4096 == 0 or count == expected['evaluable_queries']:
                print(f'Test: {count:,}/{expected["evaluable_queries"]:,} queries; {pair_count:,} pairs; '
                    f'CUDA/Triton changed rankings={accum["cuda"]["changed_rankings_vs_pytorch"]}/'
                    f'{accum["triton"]["changed_rankings_vs_pytorch"]}', flush=True)
                save(HERE / 'test_progress.json', {'queries': count, 'pairs': pair_count, 'accumulators': accum})
    if count != expected['evaluable_queries'] or pair_count != expected['candidates']:
        raise RuntimeError('Full evaluation count mismatch')
    if missing != expected['queries_without_relevant_candidate_included']:
        raise RuntimeError('No-relevant-query count mismatch')
    temporary.replace(audit)
    metrics = {}
    for mode, a in accum.items():
        metrics[mode] = {'mrr_at_10': a['mrr_total'] / count, 'ndcg_at_10': a['ndcg_total'] / count,
            'changed_rankings_vs_pytorch': a['changed_rankings_vs_pytorch'],
            'max_score_error_vs_pytorch': a['max_score_error_vs_pytorch'],
            'mean_absolute_score_error_vs_pytorch': a['absolute_score_error_sum'] / pair_count,
            'pairs_outside_score_tolerance': a['pairs_outside_score_tolerance'],
            'forward_wall_seconds_single_pass': a['forward_wall_seconds']}
    result = {'status': 'complete', 'plan': plan, 'plan_sha256': sha256(HERE / 'test_plan.json'),
        'queries': count, 'pairs': pair_count, 'queries_without_relevant_candidate': missing,
        'metrics': metrics, 'score_parity_verified': all(a['pairs_outside_score_tolerance'] == 0 for a in accum.values()),
        'ranking_parity_verified': not differences, 'changed_queries': differences,
        'score_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'audit_file': audit.name, 'audit_sha256': sha256(audit), 'elapsed_seconds': time.perf_counter() - begin,
        'limitations': ['Constructed MS MARCO mined-triplet candidate lists; not a retrieval leaderboard evaluation.',
            'Normal PyTorch disables fused fast paths to isolate custom-kernel changes.',
            'Single evaluation pass duration is not a latency benchmark; previous rotated benchmarks remain authoritative.',
            'No training, checkpoint selection or tuning on test data.']}
    save(HERE / 'test_quality.json', result)
    print(json.dumps({'metrics': metrics, 'score_parity_verified': result['score_parity_verified'],
                      'ranking_parity_verified': result['ranking_parity_verified']}, indent=2), flush=True)


if __name__ == '__main__': main()
