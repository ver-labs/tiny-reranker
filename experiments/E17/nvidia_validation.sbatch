#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/E17/nvidia-validation-%j.log
set -euo pipefail
# Submit with --dependency=afterok:<benchmark-job> to prevent competing GPU work.
.cuda-env/bin/python -m experiments.E17.gpu_dispatch_probe
.cuda-env/bin/python -m pytest -q tests/test_e13_triton.py tests/test_e14_cuda.py
