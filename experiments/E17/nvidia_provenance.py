"""Supplemental pre-execution NVIDIA binary/source guard; no CPU timings."""
import argparse
import json
import platform
from .common import ROOT, HERE, sha, save, verify_plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--require-gpu',action='store_true');a=p.parse_args()
    verify_plan();plan=json.loads((HERE/'nvidia_provenance_plan.json').read_text())
    assert plan['benchmark_plan_sha256']==sha(HERE/'plan.json')
    assert plan['guard_source_sha256']==sha(__file__)
    for relative,digest in plan['files'].items():
        if sha(ROOT/relative)!=digest:raise RuntimeError(f'NVIDIA asset changed: {relative}')
    output={'status':'binary_and_source_hashes_verified','supplement_sha256':sha(HERE/'nvidia_provenance_plan.json'),
            'gpu_execution':'pending; asset checks are not kernel validation','files':plan['files']}
    if a.require_gpu:
        import torch
        if platform.system()!='Linux' or not torch.cuda.is_available() or torch.version.hip is not None:
            raise RuntimeError('Linux/NVIDIA CUDA GPU required; no CPU substitution')
        output['gpu']=torch.cuda.get_device_name()
        if 'L40S' not in output['gpu']:raise RuntimeError('Frozen NVIDIA hardware is L40S')
        output['gpu_execution']='Access verified, benchmark execution follows; no timings in this guard'
    save(HERE/'nvidia_provenance.json',output);print(json.dumps(output,indent=2))


if __name__=='__main__':main()
