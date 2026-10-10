#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/E18/nvidia-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
# E18 library must have been built and operation checks saved first.
.cuda-env/bin/python -u -m experiments.E18.launch --device cuda
