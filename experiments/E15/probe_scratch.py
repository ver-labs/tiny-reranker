"""Independently check two full SciFact candidate lists with the CPU model."""
import gzip
import hashlib
import json
from pathlib import Path
import torch
from transformers import BertTokenizer
from scripts.model import ScratchConfig, ScratchReranker

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4); torch.backends.mha.set_fastpath_enabled(False)
    checkpoint = ROOT / 'runs/full_train/epoch_0003.pt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    model = ScratchReranker(ScratchConfig.from_file(ROOT / 'experiments/E1/four_layer/config.json')).eval()
    model.load_state_dict(saved['model_state_dict'], strict=True)
    tokenizer = BertTokenizer.from_pretrained(ROOT / 'artifacts/short_epoch_1024/tokenizer', local_files_only=True)
    folder = ROOT / 'data/beir/scifact'
    with (folder / 'corpus.jsonl').open() as f: corpus = {r['_id']: r for r in map(json.loads, f)}
    with (folder / 'queries.jsonl').open() as f: queries = {r['_id']: r['text'] for r in map(json.loads, f)}
    with gzip.open(HERE / 'scifact_scratch_128_rankings.jsonl.gz', 'rt') as f:
        rows = [json.loads(next(f)) for _ in range(2)]
    probes = []
    with torch.inference_mode():
        for r in rows:
            ids = r['candidate_ids']; values = []
            for start in range(0, len(ids), 32):
                docs = [(corpus[d].get('title','')+' '+corpus[d]['text']).strip() for d in ids[start:start+32]]
                inputs = tokenizer([queries[r['query_id']]]*len(docs), docs,
                    padding=True, truncation='longest_first', max_length=128, return_tensors='pt')
                values.extend(model(**inputs).tolist())
            torch.testing.assert_close(torch.tensor(values), torch.tensor(r['scores']), atol=3e-5, rtol=3e-5)
            order = sorted(range(len(ids)), key=lambda i: (-values[i],ids[i]))
            ranking = [ids[i] for i in order]
            if ranking != r['ranking']: raise RuntimeError('CPU ranking differs')
            probes.append({'query_id': r['query_id'], 'pairs': len(ids),
                'max_score_error': max(abs(a-b) for a,b in zip(values,r['scores'])),
                'ranking_unchanged': True, 'relevant_document_ranks': [i+1 for i,d in enumerate(ranking) if r['qrels'].get(d,0)>0]})
    result = {'status': 'verified', 'device': 'CPU', 'cpu_pair_batch_size': 32,
        'gpu_pair_batch_size': 128, 'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'probe_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'probes': probes}
    (HERE / 'scratch_scifact_cpu_probe.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__': main()
