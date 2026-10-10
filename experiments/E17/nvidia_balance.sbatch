#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/E17/nvidia-balance-%j.log
set -euo pipefail
# Dependency on prior validation job prevents concurrent GPU execution.
.cuda-env/bin/python -m experiments.E17.nvidia_provenance --require-gpu
.cuda-env/bin/python -m experiments.E17.nvidia_balance
