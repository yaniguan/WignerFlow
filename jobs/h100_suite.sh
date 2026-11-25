#!/usr/bin/env bash
# Launch H100 benchmark tiers for WignerFlow (`equitraj`).
# Usage:
#   sbatch --array=1-3 jobs/h100_suite.sh configs/argon/h100_v1.yaml
#   sbatch --array=1-3 jobs/h100_suite.sh configs/ala2/h100_v1.yaml
#   sbatch --array=1-3 jobs/h100_suite.sh configs/argon/trajcast_h100.yaml
#SBATCH --job-name=wignerflow-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail

CFG="${1:?provide a config yaml path, e.g. configs/ala2/h100_v1.yaml}"
SEED="${SLURM_ARRAY_TASK_ID:-1}"
BASE_OUT="$(python -c "import yaml, sys; print(yaml.safe_load(open(sys.argv[1]))['out_dir'])" "$CFG")"
RUN_DIR="${BASE_OUT}_seed${SEED}"

mkdir -p logs "$RUN_DIR"
python -c "
import sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
cfg['seed'] = int(sys.argv[2])
cfg['out_dir'] = sys.argv[3]
yaml.safe_dump(cfg, open(sys.argv[4], 'w'))
" "$CFG" "$SEED" "$RUN_DIR" "$RUN_DIR/run_config.yaml"

export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
python -m equitraj.train --config "$RUN_DIR/run_config.yaml" --device cuda --no_tensorboard

Periodic="$(python -c "import yaml, os; d=yaml.safe_load(open('$RUN_DIR/run_config.yaml'))['data_dir']; c=yaml.safe_load(open(os.path.join(d,'config.yaml'))); print(c.get('system','argon'))" )"
if [[ "$Periodic" == "ala2" || "$Periodic" == "aspirin" || "$Periodic" == md17_* ]]; then
  python scripts/evaluate_molecule.py --run "$RUN_DIR" --ckpt best_stable.pt --steps 10000 --seeds 5 --starts 3 --device cuda
else
  python scripts/evaluate.py --run "$RUN_DIR" --ckpt best_stable.pt --steps 10000 --seeds 5 --starts 3 --device cuda
fi
