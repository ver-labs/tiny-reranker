"""Sequential GPU execution; never run competing benchmarks simultaneously.

Apple: .venv/bin/python -m experiments.E17.launch --device mps
NVIDIA: .cuda-env/bin/python -m experiments.E17.launch --device cuda
NVIDIA summary remains separate; do not replace it with MPS measurements.
"""
import argparse
import json
import os
import subprocess
import sys
from .common import ROOT, HERE, save, verify_plan, sha


def run(module, args, result=None, timeout=None):
    if result and result.exists():
        output=json.loads(result.read_text())
        if output.get('plan_sha256')!=sha(HERE/'plan.json'):raise RuntimeError('Stale existing output')
        if output.get('status') in ('complete','compatible','unsupported_or_failed','timeout'):
            if 'audit_file' in output and sha(HERE/output['audit_file'])!=output['audit_sha256']:
                raise RuntimeError('Existing score audit changed')
            print('Reusing',result.name,flush=True);return
    try:
        subprocess.run([sys.executable,'-u','-m',module,*args],cwd=ROOT,check=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        if not result:raise
        save(result,{'plan_sha256':sha(HERE/'plan.json'),'status':'timeout','timeout_seconds':timeout,
                     'scope':'Bounded compile probe; no warm timing; disk cache state unknown'})


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['mps','cuda'],required=True);a=p.parse_args()
    plan=verify_plan()
    import torch
    available=torch.backends.mps.is_available() if a.device=='mps' else torch.cuda.is_available()
    if not available:raise RuntimeError(f'{a.device} GPU required; CPU substitution prohibited')
    run('experiments.E17.compile_probe',['--device',a.device],HERE/f'{a.device}_compile.json',120)
    run('experiments.E17.accuracy',['--device',a.device],HERE/f'{a.device}_accuracy.json')
    for session in range(plan['sessions']):
        run('experiments.E17.direct_benchmark',['--device',a.device,'--session',str(session)],HERE/f'{a.device}_direct_s{session}.json')
        if a.device=='mps':
            # The HTTP session covers four sequential independent coordinators.
            run('experiments.E17.http_benchmark',['--session',str(session)],HERE/f'http_session_{session}.json')
    if a.device=='mps':
        run('experiments.E17.summarize',[])
        run('experiments.E17.verify',[])
    else:
        run('experiments.E17.nvidia_summary',[])


if __name__=='__main__':main()
