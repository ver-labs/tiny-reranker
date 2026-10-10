#!/bin/bash
#SBATCH --job-name=reranker-e14-benchmark
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=e14_benchmark_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SLURM_SUBMIT_DIR/.triton-cache"
module load nvhpc-24.9/nvhpc/24.9
nvcc --version
nvcc -O3 -shared -Xcompiler=-fPIC -arch=sm_89 --fmad=false --prec-div=true --prec-sqrt=true -cudart static experiments/E14/residual_layer_norm.cu -o experiments/E14/residual_layer_norm.so
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.cuda-env/bin/python -m pytest tests/test_e14_cuda.py -q -rs
.cuda-env/bin/python -u -m experiments.E14.benchmark --require-gpu
touch e14_benchmark_complete
