#!/bin/bash
set -euo pipefail
compiler=experiments/E18/cache/cuda12.6/cuda_nvcc-linux-x86_64-12.6.85-archive/bin/nvcc
if [ ! -x "$compiler" ]; then
  .cuda-env/bin/python -m experiments.E18.download_cuda
fi
"$compiler" --version
"$compiler" -O3 -std=c++17 --shared -Xcompiler=-fPIC --fmad=false --prec-div=true --prec-sqrt=true -cudart static -arch=sm_89 \
  experiments/E18/residual_norm.cu -o experiments/E18/residual_norm.so
