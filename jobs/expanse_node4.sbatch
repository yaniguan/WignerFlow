#!/bin/bash
# Full Expanse GPU node (4x V100): trains up to 4 configs at once, one per GPU.
# Usage: sbatch -J <name> -t 48:00:00 jobs/expanse_node4.sbatch cfg1.yaml [cfg2.yaml cfg3.yaml cfg4.yaml]
# Why: the per-account job limit on gpu-shared is low, so packing 4 runs per job gets more runs going.
#SBATCH -A cla175
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH -c 10
#SBATCH --gpus=4
#SBATCH --mem=360G
#SBATCH -o logs/%x-%j.out

set -uo pipefail
export MAMBA_ROOT_PREFIX=$HOME/micromamba
eval "$(~/bin/bin/micromamba shell hook -s bash)"
micromamba activate ~/envs/wignerflow
cd ~/WignerFlow
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
gpu=0
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "GPU $gpu: $cfg"
  CUDA_VISIBLE_DEVICES=$gpu python -m equitraj.train --config "$cfg" > "logs/train_${name}_${SLURM_JOB_ID}.out" 2>&1 &
  gpu=$((gpu + 1))
done
wait
