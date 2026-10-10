#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=experiments/E17/nvidia-%j.log
set -euo pipefail
# Submit only after authenticated access is restored and the frozen inputs are
# copied and verified. No existing jobs are cancelled or changed.
.cuda-env/bin/python -m experiments.E17.nvidia_provenance --require-gpu
.cuda-env/bin/python -m experiments.E17.launch --device cuda
