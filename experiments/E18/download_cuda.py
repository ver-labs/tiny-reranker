"""Project-local verified NVIDIA compiler/runtime; no shared cluster changes."""
import hashlib
import json
import shutil
import tarfile
import urllib.request
from pathlib import Path
from .common import HERE,save,sha


def main():
    base='https://developer.download.nvidia.com/compute/cuda/redist/'
    root=HERE/'cache/cuda12.6';root.mkdir(parents=True,exist_ok=True)
    data=urllib.request.urlopen(base+'redistrib_12.6.3.json',timeout=60).read()
    manifest=json.loads(data);entries={}
    for package in ('cuda_nvcc','cuda_cudart'):
        entry=manifest[package]['linux-x86_64'];entries[package]=entry
        archive=root/(package+'.tar.xz')
        if not archive.exists() or sha(archive)!=entry['sha256']:
            with urllib.request.urlopen(base+entry['relative_path'],timeout=120) as response,archive.open('wb') as output:shutil.copyfileobj(response,output)
        assert sha(archive)==entry['sha256']
        with tarfile.open(archive) as tar:tar.extractall(root,filter='data')
    nvcc=next(root.glob('cuda_nvcc-*/bin/nvcc'))
    cudart=next(root.glob('cuda_cudart-*'))
    compiler=nvcc.parents[1]
    # NVCC expects toolkit headers/lib alongside bin; merge verified cudart tree.
    shutil.copytree(cudart/'include',compiler/'include',dirs_exist_ok=True)
    shutil.copytree(cudart/'lib',compiler/'lib64',dirs_exist_ok=True)
    save(HERE/'cuda_compiler_manifest.json',{'source':base+'redistrib_12.6.3.json','redistribution_manifest_sha256':hashlib.sha256(data).hexdigest(),'packages':entries,'nvcc_relative_path':str(nvcc.relative_to(HERE)),'scope':'Project-local compiler; no system module modifications'})
    print(nvcc,flush=True)

if __name__=='__main__':main()
