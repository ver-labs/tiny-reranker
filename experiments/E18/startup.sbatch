#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --output=experiments/E18/startup-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
timeout 900 .cuda-env/bin/python -u -m experiments.E18.startup_probe --device cuda --path compiled
timeout 120 .cuda-env/bin/python -u -m experiments.E18.startup_probe --device cuda --path stock
