"""Publish an already verified allowlisted stage using cached HF authentication.

No tokens are accepted on CLI or logged. Existing unrelated repositories are refused.
The first commit contains the artifact; the second pins that immutable commit in usage.
"""
import argparse
import json
import re
from pathlib import Path
import subprocess
import sys
import tempfile

from huggingface_hub import HfApi, ModelCard, snapshot_download
from huggingface_hub.errors import LocalTokenNotFoundError, RepositoryNotFoundError
from scripts.hf_release import EXPECTED, refresh_manifest, sha

NAME = 'tiny-reranker-4.7m-experimental'
FILES = {'LICENSE','LICENSE-CODE','LICENSE-TOKENIZER','LICENSE-WEIGHTS','NOTICE','README.md',
         'config.json','evaluation.json','example.py','model.safetensors','release_manifest.json',
         'requirements.txt','reranker.py','tokenizer/tokenizer.json','tokenizer/tokenizer_config.json',
         'validation_fixtures.json','verify_release.py'}


def check_stage(stage):
    files = {str(p.relative_to(stage)) for p in stage.rglob('*') if p.is_file()}
    if files != FILES:
        raise RuntimeError(f'Unexpected/missing files: {files ^ FILES}')
    manifest = json.loads((stage / 'release_manifest.json').read_text())
    assert manifest['source_checkpoint_sha256'] == EXPECTED
    assert manifest['exact_state_dict_parity']
    assert set(manifest['file_sha256']) == FILES - {'release_manifest.json'}
    for name, expected in manifest['file_sha256'].items():
        assert sha(stage / name) == expected, name
    data = ModelCard.load(stage / 'README.md').data.to_dict()
    assert data['license'] == 'cc-by-nc-4.0' and data['library_name'] == 'pytorch'
    assert 'base_model' not in data and 'pipeline_tag' not in data


def publish(stage, report_dir, python):
    api = HfApi()
    try:
        account = api.whoami()
    except LocalTokenNotFoundError:
        print('AUTHENTICATION_PENDING: run .venv/bin/hf auth login locally')
        return 2
    assert account['type'] == 'user'
    namespace = account['name']
    repo_id = namespace + '/' + NAME
    print(json.dumps({'authenticated_personal_account':namespace,'fullname':account.get('fullname'),'intended_repo':repo_id}))
    card_path = stage / 'README.md'
    card = card_path.read_text().replace('PERSONAL_NAMESPACE/' + NAME, repo_id)
    card = card.replace('verified from local Git authorship.', f'verified from local Git authorship; publisher account: [{namespace}](https://huggingface.co/{namespace}).')
    card_path.write_text(card)
    refresh_manifest(stage)
    check_stage(stage)
    report_dir.mkdir(parents=True, exist_ok=True)
    # Only staged artifacts are imported by this subprocess in a fresh working directory.
    with tempfile.TemporaryDirectory(prefix='hf-prepublish-') as work:
        subprocess.run([str(python), '-B', str(stage / 'verify_release.py'), str(stage),
                        str(report_dir / 'preupload_validation.json')], cwd=work, check=True,
                       env=offline_environment())
    ModelCard.load(card_path).validate()
    try:
        existing = api.model_info(repo_id)
    except RepositoryNotFoundError as error:
        if getattr(getattr(error, 'response', None), 'status_code', None) != 404:
            raise
        existing = None
    if existing:
        old = snapshot_download(repo_id, revision=existing.sha, allow_patterns=['release_manifest.json'],
                                local_dir=tempfile.mkdtemp(prefix='hf-existing-manifest-'))
        manifest_path = Path(old) / 'release_manifest.json'
        if not manifest_path.is_file() or json.loads(manifest_path.read_text()).get('source_checkpoint_sha256') != EXPECTED:
            raise RuntimeError('Refusing unrelated existing model repository')
        if json.loads(manifest_path.read_text()).get('source_checkpoint') != 'runs/full_train/epoch_0003.pt':
            raise RuntimeError('Existing repository source is ambiguous')
    else:
        api.create_repo(repo_id, repo_type='model', private=False, exist_ok=False)
    first = api.upload_folder(repo_id=repo_id, folder_path=stage, allow_patterns=sorted(FILES),
                              commit_message='Release original scratch-trained epoch 3 (experimental educational artifact)')
    pinned = first.oid
    card = re.sub(r'^revision = .*$',
                  f"revision = '{pinned}'  # immutable weights release commit",
                  card_path.read_text(), flags=re.MULTILINE)
    card_path.write_text(card)
    refresh_manifest(stage)
    check_stage(stage)
    final = api.upload_folder(repo_id=repo_id, folder_path=stage,
                              allow_patterns=['README.md','release_manifest.json'],
                              commit_message='Pin immutable artifact revision in usage example',
                              parent_commit=pinned)
    revision = final.oid
    # Save immediately so network failures cannot erase knowledge of a successful upload.
    publication = {'repo_id':repo_id,'url':'https://huggingface.co/' + repo_id,
                   'artifact_revision':pinned,'published_revision':revision,
                   'account':{'name':namespace,'fullname':account.get('fullname')},
                   'source_checkpoint_sha256':EXPECTED,
                   'published_file_sha256':{name:sha(stage / name) for name in sorted(FILES)},
                   'remote_verified':False}
    destination = report_dir / 'publication.json'
    destination.write_text(json.dumps(publication,indent=2) + '\n')
    info = api.model_info(repo_id, revision=revision, files_metadata=True)
    assert info.sha == revision and info.private is False
    remote_names = {s.rfilename for s in info.siblings}
    assert remote_names == FILES | {'.gitattributes'}, remote_names
    assert info.card_data.license == 'cc-by-nc-4.0'
    assert info.card_data.library_name == 'pytorch'
    remote_path = Path(tempfile.mkdtemp(prefix='hf-downloaded-exact-revision-'))
    snapshot_download(repo_id, revision=revision, local_dir=remote_path,
                      allow_patterns=sorted(FILES), force_download=True)
    for name, expected in publication['published_file_sha256'].items():
        assert sha(remote_path / name) == expected, name
    with tempfile.TemporaryDirectory(prefix='hf-remote-smoke-') as work:
        subprocess.run([str(python), '-B', str(remote_path / 'verify_release.py'), str(remote_path),
                        str(report_dir / 'remote_validation.json')], cwd=work, check=True,
                       env=offline_environment())
    # Execute README download block: it pins the first immutable artifact commit.
    readme = (remote_path / 'README.md').read_text()
    download_block = readme.split('```python\n')[1].split('```')[0]
    local_block = readme.split('<!-- local-example -->')[1].split('```python\n')[1].split('```')[0]
    with tempfile.TemporaryDirectory(prefix='hf-readme-download-') as work:
        smoke = Path(work) / 'readme_download.py'
        smoke.write_text(download_block + '\n' + local_block + '\nimport subprocess, sys\nsubprocess.run([sys.executable,"-B",str(__import__("pathlib").Path(snapshot)/"example.py"),snapshot],check=True)\nsubprocess.run([sys.executable,"-B",str(__import__("pathlib").Path(snapshot)/"verify_release.py"),snapshot],check=True)\n')
        subprocess.run([str(python), '-B', str(smoke)],cwd=work,check=True)
    # Explicit unauthenticated public page/API availability, not just owner's authenticated access.
    import httpx
    response = httpx.get(publication['url'],follow_redirects=True)
    response.raise_for_status()
    assert NAME in response.text and 'Experimental' in response.text
    public_info = HfApi(token=False).model_info(repo_id, revision=revision)
    assert public_info.sha == revision and not public_info.private
    publication['remote_verified'] = True
    publication['fresh_download_path'] = str(remote_path)
    destination.write_text(json.dumps(publication,indent=2) + '\n')
    print(json.dumps(publication,indent=2))
    return 0


def offline_environment():
    import os
    env = os.environ.copy()
    env.pop('PYTHONPATH',None)
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_HOME=tempfile.mkdtemp(prefix='hf-empty-cache-'))
    return env


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',type=Path,required=True)
    parser.add_argument('--report-dir',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    args = parser.parse_args()
    # Preserve a virtualenv's interpreter symlink: resolving it selects base Python
    # and loses the isolated environment's installed packages.
    sys.exit(publish(args.stage.resolve(),args.report_dir.resolve(),args.python.absolute()))
