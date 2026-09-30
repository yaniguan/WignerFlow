#!/bin/bash
# Submit one Expanse GPU job per trajectory of a data config.
# Usage (on Expanse, from ~/WignerFlow): bash jobs/expanse_submit_md.sh configs/data/li_ec_q08.yaml [hours]
# Data goes to ~/WignerFlow/data, which is a symlink to Lustre project storage.
set -euo pipefail
config=$1
hours=${2:-4}
name=$(basename "$config" .yaml)
n_traj=$(python3 -c "import yaml,sys; print(yaml.safe_load(open('$config'))['n_traj'])")
mkdir -p logs
for i in $(seq 0 $((n_traj - 1))); do
  sbatch -J "md_${name}_${i}" -t "${hours}:00:00" -c 4 --mem=16G jobs/expanse_gpu.sbatch \
    python scripts/gen_data.py --config "$config" --traj "$i"
done
