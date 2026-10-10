#!/bin/bash
#SBATCH --job-name=reranker-e14-test
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=e14_test_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR="$SLURM_SUBMIT_DIR/.triton-cache"
module load nvhpc-24.9/nvhpc/24.9
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.cuda-env/bin/python -u -m experiments.E14.evaluate_test
touch e14_test_complete
