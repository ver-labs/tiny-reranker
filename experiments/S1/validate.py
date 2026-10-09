"""Layer parity, padding/truncation probes and full held-out validation ranking."""
import argparse
import json
import itertools
import numpy as np
import torch
import mlx.core as mx

from scripts.evaluate_initial import CANDIDATES, CANDIDATE_MANIFEST, ranking_metrics
from .backend import Backend, HERE, FOLDER, REFERENCE, WEIGHTS, convert, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parity-only', action='store_true', help='Run small numerical probes without repeating full validation')
    args = parser.parse_args()
    if not WEIGHTS.exists():
        convert()
    pt, mlx = Backend('pytorch'), Backend('mlx')
    layer_errors = {}
    probes = [(['short query', 'long query ' * 90],
               ['short document', 'long passage ' * 150]),
              (['apple silicon'], ['a single document']),
              (['how to save a model'] * 8, ['model checkpoint ' * (i + 1) for i in range(8)])]
    with CANDIDATES.open() as f:
        rows = [json.loads(s) for s in itertools.islice(f, 32)]
    probes.append(([r['query'] for r in rows for _ in r['candidates']],
                   [c['text'] for r in rows for c in r['candidates']]))
    score_error = 0.0
    for queries, documents in probes:
        inputs = pt.encode(queries, documents)
        assert all(np.array_equal(v, mlx.encode(queries, documents)[k]) for k, v in inputs.items())
        captured = {}
        def hook(name):
            def capture(module, args, output):
                captured[name] = output.detach().cpu().numpy()
            return capture
        handles = [module.register_forward_hook(hook(name))
                   for name, module in pt.model.named_modules()
                   if name in {'embedding_norm', 'final_norm'} or
                      name in {f'encoder_layers.{i}' for i in range(pt.config.num_layers)}]
        try:
            expected = pt.forward(pt.prepare(inputs))
        finally:
            for handle in handles:
                handle.remove()
        states = mlx.model(**mlx.prepare(inputs), trace=True)
        mx.eval(states)
        for name, value in captured.items():
            actual = np.array(states[name])
            error = float(np.max(np.abs(value - actual)))
            layer_errors[name] = max(layer_errors.get(name, 0), error)
            np.testing.assert_allclose(actual, value, atol=2e-4, rtol=1e-4)
        actual = np.array(states['scores'])
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        # Also compare the normal PyTorch inference path without tracing hooks.
        expected = pt.forward(pt.prepare(inputs))
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)
        score_error = max(score_error, float(np.max(np.abs(actual - expected))))
        changed = {k: v.copy() for k, v in inputs.items()}
        pad = changed['attention_mask'] == 0
        changed['input_ids'][pad] = pt.tokenizer.unk_token_id
        changed['token_type_ids'][pad] = 1
        for backend in (pt, mlx):
            np.testing.assert_allclose(backend.forward(backend.prepare(inputs)),
                backend.forward(backend.prepare(changed)), atol=1e-5, rtol=0)
    write_json(HERE / 'parity.json', {'probe_batch_sizes': [len(q) for q, _ in probes],
        'max_score_absolute_error': score_error, 'layer_max_absolute_errors': layer_errors,
        'layer_tolerance': {'atol': 2e-4, 'rtol': 1e-4},
        'score_tolerance': {'atol': 1e-5, 'rtol': 1e-4},
        'identical_tokenization': True, 'padding_ignored': True,
        'mlx_enable_tf32': '0',
        'singleton_and_max_length_truncation_checked': True})
    print('Layer parity and padding probes passed:', layer_errors, flush=True)
    if args.parity_only:
        return
    manifest = json.loads(CANDIDATE_MANIFEST.read_text())['splits']['validation']
    if sha256(CANDIDATES) != manifest['sha256']:
        raise RuntimeError('Candidate hash mismatch')
    reference = json.loads((REFERENCE / 'epoch_0003_summary.json').read_text())
    if reference['candidate_sha256'] != manifest['sha256']:
        raise RuntimeError('Reference candidate mismatch')
    count = changed_rankings = 0
    max_error = mrr_sum = ndcg_sum = 0.0
    reference_file = REFERENCE / 'epoch_0003_rankings.jsonl'
    out = HERE / 'mlx_rankings.jsonl'
    temp = out.with_suffix('.jsonl.tmp')
    with CANDIDATES.open() as source, reference_file.open() as ref, temp.open('w') as output:
        while True:
            rows = [json.loads(s) for s in itertools.islice(source, 32)]
            if not rows:
                break
            flat = [(r, c) for r in rows for c in r['candidates']]
            scores = []
            for i in range(0, len(flat), 64):
                chunk = flat[i:i+64]
                scores.extend(mlx.score([r['query'] for r,c in chunk], [c['text'] for r,c in chunk]))
            offset = 0
            for row in rows:
                chunk = scores[offset:offset+len(row['candidates'])]
                offset += len(chunk)
                ranking = sorted([{'document_id': c['document_id'], 'label': c['label'], 'score': s}
                    for c,s in zip(row['candidates'], chunk)], key=lambda a: (-a['score'], a['document_id']))
                baseline = json.loads(next(ref))
                assert baseline['query_id'] == row['query_id']
                changed_rankings += [a['document_id'] for a in ranking] != [a['document_id'] for a in baseline['ranking']]
                baseline_scores = {a['document_id']: a['score'] for a in baseline['ranking']}
                max_error = max(max_error, max(abs(a['score'] - baseline_scores[a['document_id']]) for a in ranking))
                mrr, ndcg = ranking_metrics([a['label'] for a in ranking])
                mrr_sum += mrr
                ndcg_sum += ndcg
                count += 1
                output.write(json.dumps({'query_id': row['query_id'], 'ranking': ranking,
                                         'mrr_at_10': mrr, 'ndcg_at_10': ndcg}) + '\n')
            if count % 4096 == 0:
                print(f'MLX validation: {count:,} queries', flush=True)
        assert next(ref, None) is None
    assert count == manifest['evaluable_queries']
    temp.replace(out)
    result = {'queries': count, 'candidate_sha256': manifest['sha256'],
        'checkpoint_sha256': reference['checkpoint_sha256'], 'precision': 'float32',
        'pair_batch_size': 64, 'pytorch_mrr_at_10': reference['mrr_at_10'],
        'pytorch_ndcg_at_10': reference['ndcg_at_10'], 'mlx_mrr_at_10': mrr_sum/count,
        'mlx_ndcg_at_10': ndcg_sum/count, 'changed_query_rankings': changed_rankings,
        'max_score_absolute_error': max_error, 'test_split_touched': False}
    write_json(HERE / 'quality.json', result)
    print(json.dumps(result, indent=2), flush=True)
    # Near ties can switch with ordinary floating-point rounding; reject quality drift.
    if max_error > 1e-4 or abs(result['mlx_ndcg_at_10'] - reference['ndcg_at_10']) > 1e-4:
        raise RuntimeError('Investigate numerical/ranking differences before benchmarks')


if __name__ == '__main__':
    main()
