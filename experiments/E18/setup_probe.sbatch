#!/bin/bash
#SBATCH --partition=rg-nextgen-hpc
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:02:00
#SBATCH --output=experiments/E18/setup-probe-%j.log
hostname
printenv MODULEPATH MODULESHOME
ls /etc/profile.d /opt/nvidia /usr/local /tools/armhpc
command -v nvcc
find /opt /usr/local /tools/armhpc -maxdepth 7 -name nvcc -o -name '*nvhpc*' 2>/dev/null
