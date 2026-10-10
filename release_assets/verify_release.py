"""Offline CPU FP32 tests using only release artifacts and synthetic fixtures."""
import json
from pathlib import Path
import runpy
import sys
import torch
if __name__ == '__main__':
    sys.path.insert(0, sys.argv[1])
from reranker import Reranker, encode_pairs, sort_results


def verify(path):
    path = Path(path).resolve()
    torch.set_num_threads(1)
    reranker = Reranker(path, device='cpu')
    fixtures = json.loads((path / 'validation_fixtures.json').read_text())
    maximum = 0.0
    all_scores = []
    for case in fixtures['cases']:
        inputs = encode_pairs(reranker.tokenizer, case['queries'], case['documents'], padding=case['padding'])
        assert inputs['input_ids'].shape[1] <= 128
        for key, value in inputs.items():
            assert value.tolist() == case['inputs'][key], f'Tokenization changed: {key}'
        with torch.inference_mode():
            actual = reranker.model(**inputs)
        expected = torch.tensor(case['scores'], dtype=torch.float32)
        error = float((actual - expected).abs().max())
        maximum = max(maximum, error)
        assert torch.isfinite(actual).all() and actual.shape == (len(case['queries']),)
        torch.testing.assert_close(actual, expected, atol=fixtures['atol'], rtol=fixtures['rtol'])
        all_scores.extend(actual.tolist())
    assert max(len(row) for case in fixtures['cases'] for row in case['inputs']['input_ids']) == 128
    plain = encode_pairs(reranker.tokenizer, ['hello'], ['world'])
    assert plain['input_ids'].tolist() == [[101, 7592, 102, 2088, 102]]
    assert plain['token_type_ids'].tolist() == [[0, 0, 0, 1, 1]]
    assert plain['attention_mask'].tolist() == [[1, 1, 1, 1, 1]]
    query = fixtures['ranking_query']
    docs = fixtures['ranking_documents']
    baseline = None
    batch_error = 0.0
    for batch in [1, 2, 3, 8]:
        result = reranker.rank(query, docs, batch_size=batch)
        scores = torch.tensor(result['scores'])
        if baseline is None:
            baseline = scores
        else:
            batch_error = max(batch_error, float((scores - baseline).abs().max()))
            torch.testing.assert_close(scores, baseline, atol=fixtures['atol'], rtol=fixtures['rtol'])
        assert result['ranking'] == sort_results(docs, result['scores'])
        assert result == reranker.rank(query, docs, batch_size=batch)
    tied = reranker.rank('hello', ['world', 'world', 'world'], batch_size=3)
    assert [r['index'] for r in tied['ranking']] == [0, 1, 2]
    assert [r['index'] for r in sort_results(['a','b','c'], [1, 2, 2])] == [1, 2, 0]
    assert reranker.rank('hello', []) == {'scores': [], 'ranking': []}
    # Masked padding may be changed without changing CLS scores.
    inputs = encode_pairs(reranker.tokenizer, ['hello'] * 2, ['world', 'a longer document with more words'])
    changed = {k: v.clone() for k, v in inputs.items()}
    mask = inputs['attention_mask'] == 0
    changed['input_ids'][mask] = reranker.tokenizer.unk_token_id
    changed['token_type_ids'][mask] = 1
    with torch.inference_mode():
        torch.testing.assert_close(reranker.model(**inputs), reranker.model(**changed), atol=fixtures['atol'], rtol=fixtures['rtol'])
    # Execute the exact local README Python block.
    readme = (path / 'README.md').read_text()
    example = readme.split('<!-- local-example -->')[1].split('```python\n')[1].split('```')[0]
    exec(compile(example, 'README.md', 'exec'), {'snapshot': str(path), '__name__': 'readme_example'})
    runpy.run_path(str(path / 'example.py'), run_name='release_example')['main'](path)
    result = {'status': 'passed', 'device': 'cpu', 'precision': 'float32',
              'atol': fixtures['atol'], 'rtol': fixtures['rtol'],
              'max_source_score_error': maximum, 'max_batch_score_error': batch_error,
              'fixture_cases': len(fixtures['cases']), 'fixture_pairs': len(all_scores),
              'token_ids_masks_segments_exact': True, 'readme_example': 'passed',
              'truncation_padding_ties_finite': 'passed'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    result = verify(sys.argv[1])
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(result, indent=2) + '\n')
