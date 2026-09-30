#!/bin/bash
# One-GPU job on Hoffman2 (UGE). Campus GPUs allow up to 24 h.
# Usage (from /u/project/sautet/yaniguan/WignerFlow):
#   qsub -N <name> -l gpu,A100,cuda=1,h_rt=2:00:00,h_data=32G jobs/hoffman2_gpu.sh <command...>
#   GPU choices: A100, H100, H200, L40S. Our group node g19 (2x L40S, up to 14 days): add ,highp
#$ -cwd
#$ -j y
#$ -o logs/$JOB_NAME-$JOB_ID.out

set -eo pipefail
source /u/local/Modules/default/init/modules.sh
module load mamba
eval "$(conda shell.bash hook)"
conda activate /u/project/sautet/yaniguan/envs/wignerflow
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "job $JOB_ID on $(hostname): $*"
"$@"
