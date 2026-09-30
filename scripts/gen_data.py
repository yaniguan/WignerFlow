"""Generate ground-truth trajectories.

    python scripts/gen_data.py --config configs/data/argon.yaml
    python scripts/gen_data.py --config configs/data/argon.yaml --traj 3 --platform CUDA
    python scripts/gen_data.py --config configs/data/argon.yaml --benchmark

Trajectories that already exist and are complete are skipped, so the script can be rerun after a crash.
"""

import argparse
import json
import os

import h5py
import yaml

from equitraj.simulate import benchmark, run_trajectory


def is_complete(path):
    if not os.path.exists(path):
        return False
    try:
        with h5py.File(path, "r") as f:
            return bool(f.attrs.get("complete", False))
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--traj", type=int, nargs="*", help="trajectory indices (default: all)")
    ap.add_argument("--platform", help="override the platform in the config")
    ap.add_argument("--out_dir", help="override out_dir in the config")
    ap.add_argument("--benchmark", action="store_true", help="only measure ns/day")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.platform:
        cfg["platform"] = args.platform
    if args.out_dir:
        cfg["out_dir"] = args.out_dir

    if args.benchmark:
        print(json.dumps(benchmark(cfg)))
        return

    os.makedirs(cfg["out_dir"], exist_ok=True)
    with open(os.path.join(cfg["out_dir"], "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    for i in args.traj if args.traj is not None else range(cfg["n_traj"]):
        path = os.path.join(cfg["out_dir"], f"traj_{i:03d}.h5")
        if is_complete(path):
            print(f"skip {path} (complete)")
            continue
        info = run_trajectory(cfg, seed=cfg["seed0"] + i, out_path=path)
        print(f"{path}: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    main()
