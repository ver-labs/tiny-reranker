"""Resolve public model revision and download only pinned required assets."""
import datetime
import json
import urllib.request
from .common import HERE, ASSETS, save, sha

MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L6-v2'


def main():
    existing = None
    if (HERE/'model_manifest.json').exists():
        existing = json.loads((HERE/'model_manifest.json').read_text())
        if all((ASSETS/name).exists() for name in existing['file_sha256']):
            from .common import verify_assets
            verify_assets()
            print('Existing immutable model assets verified')
            return
        revision = existing['revision']
    else:
        with urllib.request.urlopen(f'https://huggingface.co/api/models/{MODEL_ID}',timeout=60) as response:
            metadata = json.load(response)
        revision = metadata['sha']
        assert metadata['cardData']['license'] == 'apache-2.0'
    assert len(revision) == 40
    ASSETS.mkdir(parents=True, exist_ok=True)
    for name in ('config.json','model.safetensors','tokenizer.json','tokenizer_config.json','special_tokens_map.json','vocab.txt','README.md'):
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{revision}/{name}'
        temporary = ASSETS/(name+'.tmp')
        with urllib.request.urlopen(url,timeout=180) as response, temporary.open('wb') as f:
            while chunk := response.read(1024*1024):
                f.write(chunk)
        temporary.replace(ASSETS/name)
        print(name, (ASSETS/name).stat().st_size, flush=True)
    with urllib.request.urlopen('https://www.apache.org/licenses/LICENSE-2.0.txt',timeout=60) as response:
        (ASSETS/'LICENSE.txt').write_bytes(response.read())
    if existing:
        from .common import verify_assets
        verify_assets()
        print('Downloaded immutable assets match the existing manifest')
        return
    import torch
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(ASSETS, local_files_only=True, trust_remote_code=False)
    config = model.config.to_dict()
    assert (config['num_hidden_layers'],config['hidden_size'],config['num_attention_heads'],config['intermediate_size'],config['max_position_embeddings']) == (6,384,12,1536,512)
    assert config['layer_norm_eps'] == 1e-12 and config['hidden_act'] == 'gelu'
    save(HERE/'model_manifest.json', {'model_id':MODEL_ID, 'revision':revision,
         'download_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'license':'Apache-2.0',
         'file_sha256':{p.name:sha(p) for p in sorted(ASSETS.iterdir()) if p.is_file()},
         'parameter_count':sum(p.numel() for p in model.parameters()), 'config':config,
         'interface':'Raw scalar classification logit; no sigmoid; trust_remote_code=False'})


if __name__ == '__main__':
    main()
