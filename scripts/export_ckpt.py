"""Export a small, self-contained release checkpoint from a training run.

    python scripts/export_ckpt.py --run results/argon/B_S3a_seed1 --ckpt snapshots/best_step10000.pt \
        --out checkpoints/argon_flow_v0.pt

Drops the optimizer state and adds what scripts/demo.py needs so that it runs without the
training data and without OpenMM: model config, atom types and masses, one starting window
(k frames from a test trajectory), and reference RDF / VACF / temperature from the test trajectories.
The file holds only tensors, numbers, strings and lists, so torch.load(weights_only=True) works.
"""

import argparse
import hashlib
import os

import numpy as np
import torch

from equitraj import analysis
from equitraj.dataset import TrajWindows
from equitraj.rollout import temperature


def reference_stats(ds, k):
    """Ground-truth T, RDF and VACF at the model stride, from the test trajectories."""
    f, dt = ds.f, ds.stride_fs
    gt_x = [np.asarray(p[::f]) for p in ds.pos]
    gt_v = [np.asarray(v[::f]) for v in ds.vel]
    box = ds.boxes[0].numpy()
    n = len(ds.types)
    idx = np.arange(n)
    t_gt = float(np.mean([temperature(torch.from_numpy(v[::10].copy()), ds.masses).mean().item() for v in gt_v]))
    r_max = min(10.0, 0.49 * float(np.diag(box).min())) if ds.periodic else 8.0
    g = []
    for x in gt_x:
        frames = x[::max(1, len(x) // 400)]
        bx = np.tile(box, (len(frames), 1, 1)) if ds.periodic else None
        r, gi, _ = analysis.rdf(frames, bx, idx, idx, r_max, 200)
        g.append(gi)
    max_lag = int(2000 / dt)
    c = np.mean([analysis.vacf(v, max_lag) for v in gt_v], axis=0)
    return {"T_K": t_gt, "rdf_r": r.tolist(), "rdf_g": np.mean(g, axis=0).tolist(),
            "vacf_lag_fs": (np.arange(max_lag) * dt).tolist(), "vacf": (c / c[0]).tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = torch.load(os.path.join(args.run, args.ckpt), map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    ds = TrajWindows(cfg["data_dir"], "test", cfg["stride_steps"], cfg["k"], m=1)
    k = cfg["k"]
    pos, vel, box = ds[0]  # k context frames + 1 target frame; the target is not exported
    payload = {
        "format": "wignerflow-v0",
        "name": cfg["name"],
        "train_step": int(ck["step"]),
        "model": {key: v.cpu() for key, v in ck["model"].items()},
        "model_cfg": cfg["model"],
        "k": k,
        "stride_fs": float(ds.stride_fs),
        "stride_steps": cfg["stride_steps"],
        "elements": [str(e) for e in ck["elements"]],  # plain str, not numpy.str_ (weights_only load)
        "types": ds.types,
        "masses": ds.masses,
        "periodic": bool(ds.periodic),
        "demo": {"pos": pos[:k].clone(), "vel": vel[:k].clone(), "box": box.clone() if ds.periodic else None},
        "reference": reference_stats(ds, k),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(payload, args.out)
    digest = hashlib.sha256(open(args.out, "rb").read()).hexdigest()
    with open(args.out + ".sha256", "w") as fh:
        fh.write(f"{digest}  {os.path.basename(args.out)}\n")
    print(f"wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB), step {payload['train_step']}, sha256 {digest[:12]}...")


if __name__ == "__main__":
    main()
