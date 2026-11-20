"""Train the force-field baseline (equitraj/forcefield.py) on the forces saved in the ground-truth HDF5 files.

    python scripts/train_ff.py --config configs/argon/FF_S0.yaml

Forces only (conservative by construction: F = -dE/dx). Keeps best.pt, the checkpoint with the lowest validation force RMSE.
Evaluate with scripts/evaluate.py (it integrates with velocity Verlet when the config has `kind: ff`).
"""

import argparse
import json
import math
import os
import time

import h5py
import numpy as np
import torch
import yaml

from equitraj.forcefield import EquiFF
from equitraj.units import NM_TO_A


def load_split(data_dir, ids):
    """Positions (n, N, 3) in Å and forces (n, N, 3) in kJ/mol/Å at the frames where forces were saved."""
    xs, fs = [], []
    for i in ids:
        with h5py.File(os.path.join(data_dir, f"traj_{i:03d}.h5"), "r") as f:
            every = int(f.attrs["force_every"]) // int(f.attrs["save_every"])
            force = f["forces"][:]
            xs.append(f["positions"][::every][: len(force)] * NM_TO_A)
            fs.append(force / NM_TO_A)
            box, elements = f["box"][0] * NM_TO_A, [e.decode() for e in f["elements"][:]]
    return np.concatenate(xs).astype(np.float32), np.concatenate(fs).astype(np.float32), box.astype(np.float32), elements


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    cfg = yaml.safe_load(open(args.config))
    tc, out_dir = cfg["train"], cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as fh:
        yaml.safe_dump(cfg, fh)
    device = torch.device(args.device)
    torch.manual_seed(cfg.get("seed", 0))
    split = yaml.safe_load(open(os.path.join(cfg["data_dir"], "config.yaml")))["split"]
    x_tr, f_tr, box, elements = load_split(cfg["data_dir"], split["train"])
    x_va, f_va, _, _ = load_split(cfg["data_dir"], split["val"])
    types_names = sorted(set(elements))
    types = torch.tensor([types_names.index(e) for e in elements], device=device)
    box_t = torch.tensor(box, device=device)[None].expand(tc["batch_size"], 3, 3)

    model = EquiFF(n_types=len(types_names), **cfg["model"]).to(device)
    scale = float(np.sqrt((f_tr**2).mean()))
    model.force_scale.fill_(scale)
    n_params = sum(p.numel() for p in model.parameters())
    print(json.dumps({"n_params": n_params, "force_rms_kJ_mol_A": scale, "n_train": len(x_tr), "n_val": len(x_va)}), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    x_tr_t, f_tr_t = torch.from_numpy(x_tr), torch.from_numpy(f_tr)

    def lr_at(s):
        warm = tc.get("warmup", 1000)
        if s < warm:
            return tc["lr"] * (s + 1) / warm
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, tc["steps"] - warm))))

    def val_rmse():
        model.eval()
        idx = np.linspace(0, len(x_va) - 1, tc.get("val_configs", 128)).astype(int)
        err = []
        for j in range(0, len(idx), tc["batch_size"]):
            b = idx[j:j + tc["batch_size"]]
            xb = torch.from_numpy(x_va[b]).to(device)
            pred = model.forces(xb, types, box_t[:len(b)])
            err.append(((pred - torch.from_numpy(f_va[b]).to(device)) ** 2).mean().item())
        model.train()
        return float(np.sqrt(np.mean(err)))

    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    best, t0, running = math.inf, time.time(), []
    model.train()
    for step in range(1, tc["steps"] + 1):
        idx = torch.randint(0, len(x_tr_t), (tc["batch_size"],))
        xb, fb = x_tr_t[idx].to(device), f_tr_t[idx].to(device)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        pred = model.forces(xb, types, box_t, create_graph=True)
        loss = ((pred - fb) ** 2).mean() / scale**2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
        opt.step()
        running.append(loss.item())
        if step % tc.get("eval_every", 2000) == 0 or step == tc["steps"]:
            rmse = val_rmse()
            rec = {"step": step, "train_loss": float(np.mean(running)), "val_force_rmse": rmse,
                   "val_force_rmse_rel": rmse / scale, "elapsed_h": (time.time() - t0) / 3600}
            running = []
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            if rmse < best:
                best = rmse
                torch.save({"model": model.state_dict(), "step": step, "cfg": cfg, "elements": types_names},
                           os.path.join(out_dir, "best.pt"))
    with open(os.path.join(out_dir, "done.json"), "w") as fh:
        json.dump({"step": tc["steps"], "best_val_force_rmse": best, "hours": (time.time() - t0) / 3600}, fh)


if __name__ == "__main__":
    main()
