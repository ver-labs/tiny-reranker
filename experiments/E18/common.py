"""E18 provenance and deterministic data; no historical model imports."""
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime,timezone
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = ROOT/'artifacts/e18_minilm_l6'
DATA = ROOT/'data/e16/development.jsonl'
SEED = 1801


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def rows():
    old = json.loads((ROOT/'experiments/E17/plan.json').read_text())
    assert sha(DATA) == old['input_sha256']['data/e16/development.jsonl']
    data = [json.loads(line) for line in DATA.open()]
    assert Counter(r['dataset'] for r in data) == {'scifact':162, 'nfcorpus':323}
    assert len({(r['dataset'], r['query_id']) for r in data}) == 485
    for r in data:
        assert len(r['texts']) == len(r['document_ids']) == len(set(r['document_ids'])) == 100
        assert any(v > 0 for v in r['qrels'].values())
    return data


def fixtures(reserve=False):
    data = rows()
    chosen = []
    for dataset in ('scifact', 'nfcorpus'):
        chosen += sorted((r for r in data if r['dataset'] == dataset),
            key=lambda r: hashlib.sha256(f"1801:{dataset}:{r['query_id']}".encode()).hexdigest())[:32]
    selected = {(r['dataset'], r['query_id']) for r in chosen}
    return [r for r in data if (r['dataset'], r['query_id']) not in selected] if reserve else chosen


def pairs(row, k):
    # Selection retains BM25 prefixes; worker/direct scoring use ID order equally.
    return [(row['query'], text) for doc, text in sorted(zip(row['document_ids'][:k], row['texts'][:k]))]


def ranked(ids, scores):
    assert len(ids) == len(set(ids)) == len(scores) and np.isfinite(scores).all()
    return [d for d,s in sorted(zip(ids,scores), key=lambda item:(-item[1],item[0]))]


def environment():
    import importlib.metadata as m
    result = {'utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(), 'python':sys.version,
              'packages':{p:m.version(p) for p in ('torch','transformers','numpy','httpx','safetensors')},
              'isolation':'No continuous hardware isolation verified'}
    try:
        result['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        result['source_commit']='Copied source snapshot; use frozen source hashes'
    for name, cmd in {'hardware':['system_profiler','SPHardwareDataType'], 'power':['pmset','-g','batt'],
                      'thermal':['pmset','-g','therm'], 'processes':['ps','-axo','pid,comm,%cpu,rss'],
                      'gpu':['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,clocks.sm,utilization.gpu,memory.used','--format=csv']}.items():
        try:
            output = subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT,timeout=10)
            result[name] = '\n'.join(line for line in output.splitlines() if not any(t in line for t in ('Serial Number','UUID','Provisioning UDID')))
        except (OSError,subprocess.SubprocessError) as error:
            result[name] = type(error).__name__
    return result


def verify_assets():
    manifest = json.loads((HERE/'model_manifest.json').read_text())
    for name,digest in manifest['file_sha256'].items():
        assert sha(ASSETS/name) == digest, name
    return manifest


def verify_plan():
    plan = json.loads((HERE/'plan.json').read_text())
    for name,digest in plan['files'].items():
        assert sha(ROOT/name) == digest, name
    verify_assets()
    return plan
