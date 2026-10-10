"""Build an allowlisted weights-only release of ORIGINAL full-training epoch 3.

Run: .venv/bin/python -m scripts.hf_release --output /private/tmp/tiny-reranker-release
Never reads an E16 checkpoint or training example file.
"""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from safetensors.torch import save_file, load_file
from transformers import BertTokenizer
from scripts.model import ScratchConfig, ScratchReranker
from scripts.scratch_tokenizer import encode_pairs

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = 'runs/full_train/epoch_0003.pt'
EXPECTED = '2ea2608a9ec98083b89606d13bed1d7c6e7a90dc6160bbe44732166137538d52'
SOURCE_FILES = [
    'scripts/model.py', 'scripts/scratch_tokenizer.py', 'scripts/inference_bundle.py',
    'scripts/checkpoint_io.py', 'scripts/full_train.py', 'scripts/clean_pairs.py',
    'experiments/P1/run.py', 'experiments/E1/four_layer/config.json',
    'artifacts/short_epoch_1024/manifest.json', 'runs/full_train/config.json',
    'runs/full_train/epoch_0003_summary.json', 'runs/full_train/epoch_0003_training.json',
    'data/manifests/msmarco_bm25_triplet.json', 'data/manifests/query_split.json',
    'results/pair_cleaning.json', 'experiments/E15/quality.json',
    'experiments/E15/audit_verification.json', 'experiments/E16/training_audit.json',
    'experiments/E16/README.md', 'experiments/E17/REPORT.md',
    'experiments/E17/NVIDIA_REPORT.md', 'README.md', 'LICENSE',
]


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def refresh_manifest(output):
    path = output / 'release_manifest.json'
    manifest = json.loads(path.read_text())
    manifest['file_sha256'] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*'))
                               if p.is_file() and p != path and '__pycache__' not in p.parts}
    write(path, manifest)


def build(output, repo_id, account_name):
    if output.exists():
        raise FileExistsError(f'Use a new empty output directory: {output}')
    assert sha(ROOT / CHECKPOINT) == EXPECTED
    saved = torch.load(ROOT / CHECKPOINT, map_location='cpu', weights_only=True)
    run = read('runs/full_train/config.json')
    assert saved['provenance_sha256'] == sha(ROOT / 'runs/full_train/config.json')
    assert saved['training_state']['epoch_index'] == 2
    assert saved['training_state']['next_sample_offset'] == run['training_pairs'] == 735276
    assert saved['training_state']['optimizer_updates'] == 137802
    assert run['model_config'] == {k: read('experiments/E1/four_layer/config.json')[k]
                                   for k in ScratchConfig.__dataclass_fields__}
    assert sha(ROOT / 'scripts/model.py') == run['code_sha256']['scripts/model.py']
    original = ScratchReranker(ScratchConfig(**run['model_config'])).eval()
    original.load_state_dict(saved['model_state_dict'], strict=True)
    assert sum(p.numel() for p in original.parameters()) == 4717185
    state = saved['model_state_dict']
    assert len(state) == 57 and all(t.dtype == torch.float32 for t in state.values())
    cfg = dict(run['model_config'], architecture='ScratchReranker', parameter_count=4717185,
               type_vocab_size=2, layer_norm_eps=original.final_norm.eps,
               activation='gelu', gelu_approximate='none', norm_first=True, batch_first=True,
               encoder_bias=True, embedding_norm_placement='after summed embeddings',
               final_norm=True, padding_mask='attention_mask == 0; key padding mask',
               is_causal=False, pooling='first token (CLS)', scoring='biased linear hidden_size to 1; raw logit',
               weight_dtype='float32', tokenizer_class='BertTokenizer',
               input_format='[CLS] query [SEP] document [SEP]', padding_side='right',
               truncation='longest_first', add_special_tokens=True,
               max_length_includes_special_tokens=True)
    output.mkdir(parents=True)
    for name in ['reranker.py','example.py','verify_release.py','LICENSE','LICENSE-CODE','LICENSE-WEIGHTS','LICENSE-TOKENIZER','NOTICE']:
        shutil.copyfile(ROOT / 'release_assets' / name, output / name)
    (output / 'tokenizer').mkdir()
    tokenizer_manifest = read('artifacts/short_epoch_1024/manifest.json')
    for name, digest in tokenizer_manifest['file_sha256'].items():
        if name.startswith('tokenizer/'):
            assert sha(ROOT / 'artifacts/short_epoch_1024' / name) == digest
            shutil.copyfile(ROOT / 'artifacts/short_epoch_1024' / name, output / name)
    source_tokenizer = BertTokenizer.from_pretrained(ROOT / 'artifacts/short_epoch_1024/tokenizer', local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(output / 'tokenizer', local_files_only=True)
    assert source_tokenizer.get_vocab() == tokenizer.get_vocab()
    assert source_tokenizer.special_tokens_map == tokenizer.special_tokens_map
    save_file({k:v.detach().cpu().contiguous() for k,v in state.items()}, str(output / 'model.safetensors'), metadata={'format':'pt'})
    reloaded = load_file(str(output / 'model.safetensors'))
    assert reloaded.keys() == state.keys()
    for key, value in state.items():
        assert value.dtype == reloaded[key].dtype and value.shape == reloaded[key].shape
        assert torch.equal(value, reloaded[key]), key
    write(output / 'config.json', cfg)
    versions = {name: importlib.metadata.version(name) for name in ['torch','transformers','safetensors','huggingface_hub','tokenizers']}
    (output / 'requirements.txt').write_text(''.join(f'{k}=={v}\n' for k,v in versions.items()))
    card = (ROOT / 'release_assets/model_card.md').read_text()
    card = card.replace('@TOKENIZERS_VERSION@', versions['tokenizers']).replace('@REPO_ID@', repo_id)
    card = card.replace('@REVISION@', 'REPLACE_WITH_PUBLISHED_COMMIT')
    card = card.replace('@ACCOUNT_ATTRIBUTION@', f'; publisher account: {account_name}' if account_name else '')
    (output / 'README.md').write_text(card)
    quality = read('experiments/E15/quality.json')
    audit = read('experiments/E15/audit_verification.json')
    diagnostic = read('experiments/E16/training_audit.json')
    assert quality['checkpoint_sha256'] == diagnostic['checkpoint_sha256'] == EXPECTED
    assert audit['quality_sha256'] == sha(ROOT / 'experiments/E15/quality.json')
    for dataset in ['constructed_test','scifact','nfcorpus']:
        assert quality['datasets'][dataset]['paths']['scratch_128']['metrics'] == audit['datasets'][dataset]['scratch_128']['metrics']
    write(output / 'evaluation.json', {
        'checkpoint_sha256': EXPECTED, 'E15_environment': quality['environment'],
        'E15_precision': quality['precision'], 'E15_pair_batch_size': quality['pair_batch_size'],
        'retrieval_manifest': quality['retrieval_manifest'],
        'constructed_test': {k:v for k,v in quality['datasets']['constructed_test'].items() if k=='manifest'},
        'verified_E15_metrics': audit['datasets'],
        'original_E16_diagnostic': {k:diagnostic[k] for k in ['scope','sample_queries','checkpoint_sha256','validation_ablation_metrics']},
        'original_training_settings': run['settings'],
        'original_epoch3_training': read('runs/full_train/epoch_0003_training.json'),
        'original_epoch3_validation': read('runs/full_train/epoch_0003_summary.json'),
        'cleaning': read('results/pair_cleaning.json'),
        'dataset': read('data/manifests/msmarco_bm25_triplet.json'),
        'query_split': read('data/manifests/query_split.json'),
    })
    torch.set_num_threads(1)
    fixtures = {'atol':1e-5, 'rtol':0, 'cases':[],
                'ranking_query':'How do small transformers score documents?',
                'ranking_documents':['A transformer scores token pairs.', 'Soup recipe.', 'many words ' * 250, '', 'Café naïve 中文 punctuation!?']}
    for queries, documents in [
        (['hello'], ['world']),
        (['What is a reranker?'] * 3, ['It scores query-document pairs.','Soup.', 'document words ' * 300]),
        (['query words ' * 200, '', 'Café naïve 中文'], ['long document ' * 300, '', 'Résumé! punctuation & whitespace\t\n']),
        (['hello'] * 8, ['world', '', 'x '*300, 'αβγ', 'a b c', 'world', 'spaces\n\t here', '短文']),
    ]:
        for padding in [True, 'max_length']:
            inputs = encode_pairs(source_tokenizer, queries, documents, max_length=128, padding=padding)
            other = encode_pairs(tokenizer, queries, documents, max_length=128, padding=padding)
            assert all(torch.equal(v,other[k]) for k,v in inputs.items())
            with torch.inference_mode():
                scores = original(**inputs)
            assert scores.shape == (len(queries),) and torch.isfinite(scores).all()
            fixtures['cases'].append({'queries':queries,'documents':documents,'padding':padding,
                                      'inputs':{k:v.tolist() for k,v in inputs.items()}, 'scores':scores.tolist()})
    write(output / 'validation_fixtures.json', fixtures)
    write(output / 'release_manifest.json', {
        'source_git_head': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_worktree_note':'Preexisting uncommitted changes preserved; individual source files hashed as inspected.',
        'source_checkpoint':CHECKPOINT, 'source_checkpoint_sha256':EXPECTED,
        'source_file_sha256':{p:sha(ROOT / p) for p in SOURCE_FILES},
        'tokenizer_source_id':tokenizer_manifest['tokenizer_source_id'],
        'tokenizer_source_revision':tokenizer_manifest['tokenizer_source_revision'],
        'tokenizer_file_sha256':{k:v for k,v in tokenizer_manifest['file_sha256'].items() if k.startswith('tokenizer/')},
        'weight_tensor_count':57, 'parameter_count':4717185,
        'exact_state_dict_parity':True, 'tensor_inventory':{k:{'shape':list(v.shape),'dtype':str(v.dtype)} for k,v in state.items()},
        'tested_versions':versions, 'file_sha256':{},
    })
    refresh_manifest(output)
    print(json.dumps({'stage':str(output),'checkpoint_sha256':EXPECTED,'exact_state_dict_parity':True,'tensors':57,'parameters':4717185,'files':sorted(str(p.relative_to(output)) for p in output.rglob('*') if p.is_file())},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repo-id',default='PERSONAL_NAMESPACE/tiny-reranker-4.7m-experimental')
    parser.add_argument('--account-name',default='')
    args = parser.parse_args()
    build(args.output.resolve(),args.repo_id,args.account_name)
