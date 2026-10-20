"""Pick a checkpoint of a finished run with long validation rollouts.

    python scripts/select_checkpoint.py --run results/argon/B_S3a_seed2 --steps 4000 --starts 4

Training saves every evaluated checkpoint in <run>/snapshots/ and picks best.pt with short rollouts.
Short rollouts missed seed 2 of argon v0, whose test rollouts crash after ~4000 steps. This script re-ranks all
snapshots with longer validation rollouts, using the same score as training (crashes, RDF error, temperature
error), writes <run>/selection.json and copies the winner to <run>/selected.pt.
Evaluate it with: python scripts/evaluate.py --run <run> --ckpt selected.pt
"""

import argparse
import json
import os
import shutil

import torch

from equitraj.dataset import TrajWindows
from equitraj.model import make_model
from equitraj.rollout import temperature
from equitraj.train import selection_score, validation_rollout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=4000, help="validation rollout length (training uses ~1000)")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--t_weight", type=float, default=1.0)
    ap.add_argument("--min_step", type=int, default=0, help="skip snapshots before this training step")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    device = torch.device(args.device)
    snaps = sorted(f for f in os.listdir(os.path.join(args.run, "snapshots")) if f.endswith(".pt"))
    rows = []
    ds = t_ref = None
    for name in snaps:
        ck = torch.load(os.path.join(args.run, "snapshots", name), map_location=device, weights_only=False)
        if ck["step"] < args.min_step:
            continue
        cfg = ck["cfg"]
        if ds is None:
            ds = TrajWindows(cfg["data_dir"], "val", cfg["stride_steps"], cfg["k"], m=1)
            t_ref = temperature(torch.stack([ds[i][1][0] for i in range(0, len(ds), max(1, len(ds) // 20))]), ds.masses, ds.n_constraints).mean().item()
        model = make_model(n_types=len(ck["elements"]), k=cfg["k"], stride_fs=ds.stride_fs, **cfg["model"])
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        torch.manual_seed(0)
        m = validation_rollout(model, ds, args.starts, args.steps, args.sampling_steps, device, t_ref)
        m.update(step=ck["step"], file=name, score=selection_score(m, args.t_weight))
        rows.append(m)
        print(json.dumps({k: m.get(k) for k in ("step", "val_crash_frac", "val_T_rel_err", "val_rdf_l1", "score")}), flush=True)
    best = min(rows, key=lambda r: r["score"])
    out = {"run": args.run, "val_steps": args.steps, "starts": args.starts, "t_weight": args.t_weight, "T_ref_K": t_ref,
           "selected": best["file"], "selected_step": best["step"], "candidates": rows}
    with open(os.path.join(args.run, "selection.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    shutil.copy(os.path.join(args.run, "snapshots", best["file"]), os.path.join(args.run, "selected.pt"))
    print(f"selected {best['file']} (step {best['step']}, score {best['score']:.3f})")


if __name__ == "__main__":
    main()
