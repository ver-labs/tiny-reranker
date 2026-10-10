#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/E18/finish-nvidia-%j.log
set -euo pipefail
export OMP_NUM_THREADS=4
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
.cuda-env/bin/python -u -m experiments.E18.finish_device --device cuda
timeout 900 .cuda-env/bin/python -u -m experiments.E18.combined_probe
.cuda-env/bin/python -m pip freeze > experiments/E18/nvidia_requirements.txt
