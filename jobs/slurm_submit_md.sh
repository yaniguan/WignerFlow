#!/bin/bash
# Submit one SLURM GPU job per trajectory of a data config.
# Usage (from your WignerFlow checkout): bash jobs/slurm_submit_md.sh configs/data/argon_full.yaml [hours]
set -euo pipefail
config=$1
hours=${2:-4}
name=$(basename "$config" .yaml)
n_traj=$(python3 -c "import sys, yaml; fh = open(sys.argv[1], encoding='utf-8'); print(yaml.safe_load(fh)['n_traj']); fh.close()" "$config")
mkdir -p logs
for i in $(seq 0 $((n_traj - 1))); do
  sbatch -J "md_${name}_${i}" -t "${hours}:00:00" -c 4 --mem=16G jobs/slurm_gpu.sbatch \
    python scripts/gen_data.py --config "$config" --traj "$i"
done
