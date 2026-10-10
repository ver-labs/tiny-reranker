#!/bin/bash
#SBATCH --job-name=reranker-e14-device
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=e14_device_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SLURM_SUBMIT_DIR/.triton-cache"
module load nvhpc-24.9/nvhpc/24.9
.cuda-env/bin/python -u -m experiments.E14.device_benchmark
touch e14_device_complete
