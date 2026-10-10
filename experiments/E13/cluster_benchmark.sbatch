#!/bin/bash
#SBATCH --job-name=reranker-e13-benchmark
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster_benchmark_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SLURM_SUBMIT_DIR/.triton-cache"
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.cuda-env/bin/python -m pytest tests/test_e13_triton.py -q -rs
.cuda-env/bin/python -u -m experiments.E13.benchmark --require-gpu
touch cluster_benchmark_complete
