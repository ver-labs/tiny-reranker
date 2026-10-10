#!/bin/bash
#SBATCH --job-name=reranker-e13-setup
#SBATCH --partition=rg-dev
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cluster_setup_%j.log
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
/nethome/sbagathi3/.local/bin/python3.12 -m venv .cuda-env
.cuda-env/bin/python -m pip install 'torch==2.14.1' 'numpy==2.5.3' 'transformers==5.18.0' 'pytest==9.1.1'
.cuda-env/bin/python -m pip freeze > cluster_packages.txt
touch cluster_setup_complete
