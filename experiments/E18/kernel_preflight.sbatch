#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/E18/kernel-preflight-%j.log
set -euo pipefail
bash experiments/E18/build_cuda.sh
.cuda-env/bin/python -m experiments.E18.kernel_check --device cuda
timeout 900 .cuda-env/bin/python -m experiments.E18.compile_probe --device cuda --path compiled
