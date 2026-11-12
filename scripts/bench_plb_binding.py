"""CLI for benchmarking fast pocket MD & 50-fs tau-RAMD on OpenForceField protein-ligand X-ray complexes.

Example:
    python scripts/bench_plb_binding.py --plb-root path/to/protein-ligand-benchmark/data
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from equitraj.plb_bench import benchmark_binding_pose_and_potency


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plb-root", required=True, help="Path to OpenForceField protein-ligand-benchmark/data")
    ap.add_argument("--max-ligands", type=int, default=5)
    ap.add_argument("--json-out", help="Optional output path for JSON benchmark report")
    args = ap.parse_args(argv)

    res = benchmark_binding_pose_and_potency(args.plb_root, max_ligands_per_target=args.max_ligands)
    text = json.dumps(res, indent=2)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
