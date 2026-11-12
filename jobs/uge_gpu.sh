#!/bin/bash
# One-GPU job on a UGE (Grid Engine) GPU cluster.
# Usage (from your WignerFlow checkout):
#   qsub -N <name> -l <gpu resource request>,h_rt=2:00:00 jobs/uge_gpu.sh <command...>
# Set MODULES_INIT to your site's modules init script and WIGNERFLOW_ENV to your conda env path.
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.out

set -eo pipefail
if [ -n "${MODULES_INIT:-}" ] && [ -f "$MODULES_INIT" ]; then
  source "$MODULES_INIT"
  module load mamba
fi
eval "$(conda shell.bash hook)"
conda activate "${WIGNERFLOW_ENV:-$HOME/envs/wignerflow}"
export PYTHONPATH=$PWD:${PYTHONPATH:-}   # so scripts/*.py can import equitraj
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "job $JOB_ID on $(hostname): $*"
"$@"
