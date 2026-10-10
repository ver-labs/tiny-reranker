#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=experiments/E18/preflight-%j.log
set -euo pipefail
.cuda-env/bin/python -m experiments.E18.preflight --device cuda
