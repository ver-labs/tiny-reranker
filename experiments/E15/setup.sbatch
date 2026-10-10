#!/bin/bash
#SBATCH --job-name=reranker-e15-deps
#SBATCH --partition=rg-dev
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=e15_setup_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
.cuda-env/bin/python -m pip install --no-deps rank-bm25==0.2.2 pytrec-eval-terrier==0.5.10
.cuda-env/bin/python -m pip freeze > experiments/E15/cluster_packages.txt
