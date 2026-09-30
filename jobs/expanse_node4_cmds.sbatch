#!/bin/bash
# Full Expanse GPU node (4x V100): runs up to 4 shell commands at once, one per GPU.
# Usage: sbatch -J <name> -t 12:00:00 jobs/expanse_node4_cmds.sbatch "cmd 1" "cmd 2" "cmd 3" "cmd 4"
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
for cmd in "$@"; do
  echo "GPU $gpu: $cmd"
  CUDA_VISIBLE_DEVICES=$gpu bash -c "$cmd" > "logs/${SLURM_JOB_NAME}_gpu${gpu}_${SLURM_JOB_ID}.out" 2>&1 &
  gpu=$((gpu + 1))
done
wait
