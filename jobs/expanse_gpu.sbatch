#!/bin/bash
# One-GPU job on Expanse (V100 32 GB).
# Usage: sbatch -J <name> -t 02:00:00 jobs/expanse_gpu.sbatch <command...>
# Example: sbatch -J bench -t 00:20:00 jobs/expanse_gpu.sbatch python scripts/gen_data.py --config configs/data/argon.yaml --benchmark --platform CUDA
#SBATCH -A cla175
#SBATCH -p gpu-shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 10
#SBATCH --gpus=1
#SBATCH --mem=90G
#SBATCH -o logs/%x-%j.out

set -euo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(~/bin/bin/micromamba shell hook -s bash)"
micromamba activate ~/envs/wignerflow   # built with micromamba (conda on Expanse is too old)
cd ~/WignerFlow
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "job $SLURM_JOB_ID on $(hostname): $*"
"$@"
