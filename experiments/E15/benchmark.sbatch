#!/bin/bash
#SBATCH --job-name=reranker-e15-quality
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=e15_quality_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
hostname
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
.cuda-env/bin/python -m pytest tests/test_e15_metrics.py -q
.cuda-env/bin/python -u -m experiments.E15.evaluate
touch e15_quality_complete
