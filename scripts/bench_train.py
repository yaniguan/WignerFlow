"""Time training steps and rollout steps for a few model sizes, to pick the experiment size.

    python scripts/bench_train.py --config configs/argon/B_S3a.yaml
Prints one JSON line per setting: seconds per training step (m=1) and per rollout step, and peak GPU memory.
"""

import argparse
import copy
import json
import time

import torch
import yaml

from equitraj.dataset import TrajWindows
from equitraj.model import EquiTraj
from equitraj.train import training_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--settings", default="128:7.0:8,64:7.0:8,64:6.0:8,64:6.0:4,32:6.0:8",
                    help="channels:cutoff:batch, comma separated")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = torch.device("cuda")
    ds = TrajWindows(cfg["data_dir"], "train", cfg["stride_steps"], cfg["k"], m=1, start_every=100)
    types, masses = ds.types.to(dev), ds.masses.to(dev)
    for setting in args.settings.split(","):
        c, cut, bs = setting.split(":")
        mc = copy.deepcopy(cfg["model"])
        mc.update(channels=int(c), cutoff=float(cut))
        model = EquiTraj(n_types=len(ds.elements), k=cfg["k"], **mc).to(dev)
        opt = torch.optim.AdamW(model.parameters(), 3e-4)
        items = [ds[i] for i in range(int(bs))]
        pos = torch.stack([i[0] for i in items]).to(dev)
        vel = torch.stack([i[1] for i in items]).to(dev)
        box = torch.stack([i[2] for i in items]).to(dev) if ds.periodic else None
        torch.cuda.reset_peak_memory_stats()
        try:
            for it in range(6):
                if it == 2:
                    torch.cuda.synchronize()
                    t = time.time()
                loss = training_loss(model, pos, vel, box, types, masses, 1, 0.01, "truncated", 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
            torch.cuda.synchronize()
            train_s = (time.time() - t) / 4
            with torch.no_grad():
                for it in range(4):
                    if it == 1:
                        torch.cuda.synchronize()
                        t = time.time()
                    model.predict(pos[:, :cfg["k"]], vel[:, :cfg["k"]], types, box, masses, n_steps=10)
                torch.cuda.synchronize()
            roll_s = (time.time() - t) / 3
            print(json.dumps({"channels": int(c), "cutoff": float(cut), "batch": int(bs),
                              "params": sum(p.numel() for p in model.parameters()),
                              "train_s_per_step": round(train_s, 3), "rollout_s_per_step_K10": round(roll_s, 3),
                              "peak_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 1)}), flush=True)
        except torch.cuda.OutOfMemoryError:
            print(json.dumps({"channels": int(c), "cutoff": float(cut), "batch": int(bs), "error": "out of memory"}), flush=True)
        del model, opt
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
