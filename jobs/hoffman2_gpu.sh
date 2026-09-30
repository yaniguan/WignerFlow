#!/bin/bash
# One-GPU job on Hoffman2 (UGE). Campus GPUs allow up to 24 h.
# Usage: qsub -N <name> -l gpu,L40S,cuda=1,h_rt=2:00:00,h_data=32G jobs/hoffman2_gpu.sh <command...>
#   GPU choices: A100, H100, H200, L40S. Our group node: add -l highp (g19, 2x L40S, up to 14 days).
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.out

set -euo pipefail
source ~/.bashrc
conda activate wignerflow
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "job $JOB_ID on $(hostname): $*"
"$@"
