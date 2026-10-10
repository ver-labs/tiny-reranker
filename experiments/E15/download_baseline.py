"""Download only pinned public TinyBERT weights/tokenizer/config files."""
import hashlib
import json
from pathlib import Path
import urllib.request
import argparse

MODEL_ID = 'cross-encoder/ms-marco-TinyBERT-L2-v2'
REVISION = '81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc'
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['tinybert', 'minilm'], default='tinybert')
    args = parser.parse_args()
    model_id, revision = (MODEL_ID, REVISION) if args.model == 'tinybert' else (
        'cross-encoder/ms-marco-MiniLM-L2-v2', '1b5cd67b15209f24824c50370e0397743aa9b787')
    folder = ROOT / ('artifacts/e15_' + args.model); folder.mkdir(parents=True, exist_ok=True)
    files = ('config.json', 'model.safetensors', 'special_tokens_map.json',
             'tokenizer.json', 'tokenizer_config.json', 'vocab.txt')
    hashes = {}
    for name in files:
        url = f'https://huggingface.co/{model_id}/resolve/{revision}/{name}'
        path = folder / name
        with urllib.request.urlopen(url, timeout=120) as response, path.open('wb') as output:
            while chunk := response.read(1024 * 1024): output.write(chunk)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f'Downloaded {name}: {path.stat().st_size:,} bytes', flush=True)
    manifest = {'model_id': model_id, 'revision': revision, 'file_sha256': hashes,
                'model_card': f'https://huggingface.co/{model_id}', 'license': 'apache-2.0'}
    name = 'baseline_manifest.json' if args.model == 'tinybert' else 'minilm_manifest.json'
    (ROOT / 'experiments/E15' / name).write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__': main()
