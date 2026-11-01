"""Rank training snapshots by longer validation rollouts and pick a fine-tuning starting point.

    python scripts/rank_snapshots.py --config configs/argon/B_S3a_stable_ft.yaml \
        --run_dir <old run>/train --top 3 --starts 8 --steps 2000 --out rank.json

Candidates: the snapshots in <run_dir>/snapshots whose log.jsonl entry had val_crash_frac == 0, ordered by
val_rdf_l1 (the --top best), or an explicit --snapshots list. Each candidate is loaded into the model built
from --config (so an enabled repulsion prior is used, with r_min estimated from the training data) and rolled
out on the validation split. Ranking key: equitraj.train.selection_score (crash fraction, -steps survived,
RDF L1 + relative temperature error). Writes the table and the chosen path to --out.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

from equitraj.dataset import TrajWindows
from equitraj.model import EquiTraj
from equitraj.prior import estimate_rmin
from equitraj.rollout import temperature
from equitraj.train import load_init_weights, selection_score, validation_rollout


def candidates_from_log(run_dir, top):
    rows = []
    path = os.path.join(run_dir, "log.jsonl")
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("val_crash_frac", 1.0) == 0.0 and "val_rdf_l1" in m:
                snap = os.path.join(run_dir, "snapshots", f"step{int(m['step']):06d}.pt")
                if os.path.exists(snap):
                    rows.append((m["val_rdf_l1"], snap))
    rows.sort()
    return [p for _, p in rows[:top]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run_dir", help="old training dir with log.jsonl and snapshots/")
    ap.add_argument("--snapshots", nargs="*")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    snaps = list(args.snapshots or []) or candidates_from_log(args.run_dir, args.top)
    if not snaps:
        raise SystemExit("no candidate snapshots")
    device = torch.device(args.device)
    train_ds = TrajWindows(cfg["data_dir"], "train", cfg["stride_steps"], cfg["k"], m=1, start_every=100)
    val_ds = TrajWindows(cfg["data_dir"], "val", cfg["stride_steps"], cfg["k"], m=1)
    t_ref = temperature(torch.stack([val_ds[i][1][0] for i in range(0, len(val_ds), max(1, len(val_ds) // 20))]),
                        val_ds.masses).mean().item()
    rmin = None
    rows = []
    for snap in snaps:
        model = EquiTraj(n_types=len(train_ds.elements), k=cfg["k"], stride_fs=train_ds.stride_fs, **cfg["model"]).to(device)
        missing, src_step = load_init_weights(model, snap, device)
        if model.prior is not None and any(k.startswith("prior.") for k in missing) and not model.prior.rmin_fixed:
            if rmin is None:
                rmin = estimate_rmin(train_ds, model.prior.percentile)
            model.prior.rmin.copy_(rmin)
        model.eval()
        t0 = time.time()
        with torch.no_grad():
            m = validation_rollout(model, val_ds, args.starts, args.steps, args.sampling_steps, device, t_ref)
        m.update({"snapshot": snap, "source_step": src_step, "wall_s": time.time() - t0,
                  "stability_score": list(selection_score(m))})
        print(json.dumps(m), flush=True)
        rows.append(m)
    rows.sort(key=lambda r: tuple(r["stability_score"]))
    out = {"protocol": vars(args), "t_ref_K": t_ref, "ranking": rows, "chosen": rows[0]["snapshot"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print("chosen:", out["chosen"], flush=True)


if __name__ == "__main__":
    main()
