"""Batched Onsager / Nernst-Einstein transport & speciation evaluation for polarizable electrolytes.

Produces `results_wignerflow.json` directly compatible with `02_Na_MD/results_mu.json`, reporting:
  - Batched multi-replica throughput (`ns/day` single-replica and batched `B` replicas on GPU)
  - Self-diffusivities `D_S00, D_S01, D_cation, D_anion` (`10^-10 m^2/s`) + Fickian `alpha`
  - Nernst-Einstein `sigma_self ± SE` and collective Einstein-Helfand `sigma_total ± SE` (`mS/cm`)
  - `ionicity`, `transference_plus_NE`, and `SSIP / CIP / AGG` ion-cluster fractions at `6.0 Å`
  - Head-to-head comparison against `--ref_results_mu` (`MU-PFF`) and `--exp_sigma` (`8.60 mS/cm`).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from equitraj.electrolyte import evaluate_electrolyte_transport
from equitraj.evaluation import load_model
from equitraj.pair_masks import exclusions_from_bonds
from equitraj.rollout import rollout, temperature


def _load_species_meta(data_dir, ds):
    sp_path = Path(data_dir) / "species_meta.json"
    if sp_path.is_file():
        return json.loads(sp_path.read_text(encoding="utf-8"))
    charges = ds.charges.tolist() if ds.charges is not None else [0.0] * len(ds.types)
    mol_species = []
    for q in charges:
        if q > 0.5:
            mol_species.append("cation")
        elif q < -0.5:
            mol_species.append("anion")
        else:
            mol_species.append("S00")
    return {
        "resolution": "inferred",
        "molecule_ids": list(range(len(charges))),
        "mol_species": mol_species,
        "mol_charges": charges,
        "total_mass_amu": float(ds.masses.sum().item()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate WignerFlow electrolyte transport & Onsager conductivity.")
    ap.add_argument("--run", default=None, help="Training run directory")
    ap.add_argument("--ckpt", default="best_stable.pt", help="Checkpoint filename or path (falls back to best.pt)")
    ap.add_argument("--steps", type=int, default=5000, help="Rollout steps per replica (5000 x 200 fs = 1.0 ns)")
    ap.add_argument("--starts", type=int, default=4, help="Starting windows from split")
    ap.add_argument("--seeds", type=int, default=4, help="Independent flow seeds per start (B = starts * seeds)")
    ap.add_argument("--sampling_steps", type=int, default=2, help="Flow ODE steps (2 after distillation)")
    ap.add_argument("--thermostat_tau_fs", type=float, default=0.0, help="Optional CSVR tau in fs (>0 enables S6)")
    ap.add_argument("--energy_rescale", action="store_true", help="Optional kinetic energy_rescale (Switch S5)")
    ap.add_argument("--temperature_K", type=float, default=298.15)
    ap.add_argument("--cluster_cutoff_A", type=float, default=6.0)
    ap.add_argument("--exp_sigma", type=float, default=8.60, help="Experimental conductivity in mS/cm")
    ap.add_argument("--ref_results_mu", default=None, help="Optional path to 02_Na_MD results_mu.json for comparison")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt_arg = args.ckpt
    if args.run is not None and not os.path.isfile(ckpt_arg):
        cand = os.path.join(args.run, ckpt_arg)
        if not os.path.isfile(cand) and ckpt_arg == "best_stable.pt":
            for fallback in ("best.pt", "ckpt.pt"):
                if os.path.isfile(os.path.join(args.run, fallback)):
                    ckpt_arg = fallback
                    break

    device = torch.device(args.device)
    model, cfg, ds, step = load_model(args.run, ckpt_arg, device, args.split, args.data_dir)
    data_dir = args.data_dir or cfg["data_dir"]
    species_meta = _load_species_meta(data_dir, ds)

    k, f, dt = cfg["k"], ds.f, ds.stride_fs
    gt_x = [np.asarray(p[::f]) for p in ds.pos]
    gt_v = [np.asarray(v[::f]) for v in ds.vel]
    t_gt = float(np.mean([temperature(torch.from_numpy(v[::max(1, len(v) // 50)].copy()), ds.masses).mean().item()
                          for v in gt_v]))

    deterministic = model.head_type == "regression"
    seeds = 1 if deterministic else args.seeds
    starts = []
    for i in range(args.starts):
        traj = i % len(gt_x)
        n_avail = max(1, len(gt_x[traj]) - k)
        frac = (i // len(gt_x)) / max(1, int(np.ceil(args.starts / len(gt_x))))
        starts.append((traj, min(int(frac * n_avail), len(gt_x[traj]) - k)))

    pos0 = torch.stack([torch.from_numpy(gt_x[t][s:s + k].copy()) for t, s in starts])
    vel0 = torch.stack([torch.from_numpy(gt_v[t][s:s + k].copy()) for t, s in starts])
    box0 = torch.stack([ds.boxes[t] for t, _ in starts])
    pos = pos0.repeat_interleave(seeds, 0)
    vel = vel0.repeat_interleave(seeds, 0)
    box = box0.repeat_interleave(seeds, 0)

    thermostat = None
    if args.energy_rescale:
        thermostat = {"kind": "energy_rescale", "t_target": args.temperature_K}
    elif args.thermostat_tau_fs > 0:
        thermostat = {"kind": "csvr", "t_target": args.temperature_K, "tau_fs": args.thermostat_tau_fs, "dt_fs": dt}

    excl = (exclusions_from_bonds(ds.bonds, len(ds.types))
            if ds.bonds is not None and len(ds.bonds) > 0 else None)
    torch.manual_seed(0)
    out = rollout(
        model, pos, vel, ds.types, box, ds.masses, args.steps,
        sampling_steps=args.sampling_steps, save_every=1, t_max=4.0 * max(t_gt, args.temperature_K),
        thermostat=thermostat, excluded_pairs=excl,
    )

    crash = out["crash_step"].numpy()
    ok = np.nonzero(crash < 0)[0]
    wall_s = max(float(out["wall_s"]), 1e-9)
    steps_done = max(int(out["steps_done"]), 1)
    n_roll = len(crash)

    out_dir = Path(args.out or (os.path.join(args.run, "eval_electrolyte") if args.run else "results/eval_electrolyte"))
    out_dir.mkdir(parents=True, exist_ok=True)

    speed = {
        "wall_s": float(out["wall_s"]),
        "batch_replicas": int(n_roll),
        "sampling_steps": int(args.sampling_steps),
        "stride_fs": float(dt),
        "ns_per_day_per_rollout": float(steps_done * dt * 1e-6 / wall_s * 86400.0),
        "ns_per_day_all_rollouts": float(n_roll * steps_done * dt * 1e-6 / wall_s * 86400.0),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device),
    }

    if len(ok) == 0 or out["positions"] is None:
        report = {"status": "all_crashed", "train_step": int(step), "speed": speed}
        (out_dir / "results_wignerflow.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return 1

    # Transpose (steps, B_ok, N, 3) -> (B_ok, steps, N, 3) for transport evaluation
    x_ok = np.transpose(out["positions"][:, ok].numpy(), (1, 0, 2, 3))
    report = evaluate_electrolyte_transport(
        x_ok, ds.masses.numpy(), ds.boxes[0].numpy(), dt, species_meta,
        temperature_K=args.temperature_K, cluster_cutoff_A=args.cluster_cutoff_A,
        exp_sigma_mS_cm=args.exp_sigma,
    )
    report["train_step"] = int(step)
    report["resolution"] = species_meta.get("resolution", "unknown")
    report["crashed_rollouts"] = int(n_roll - len(ok))
    report["speed"] = speed

    if args.ref_results_mu and Path(args.ref_results_mu).is_file():
        ref = json.loads(Path(args.ref_results_mu).read_text(encoding="utf-8"))
        ref_sp = ref.get("system_properties", {})
        wf_sp = report["system_properties"]
        report["vs_mu_pff"] = {
            "mu_pff_sigma_total_mS_cm": ref_sp.get("conductivity_total [mS/cm]"),
            "mu_pff_sigma_self_mS_cm": ref_sp.get("conductivity_self [mS/cm]"),
            "mu_pff_density_g_cm3": ref_sp.get("density [g/cm3]"),
            "wf_sigma_total_mS_cm": wf_sp["conductivity_total [mS/cm]"],
            "wf_sigma_total_se_mS_cm": wf_sp["conductivity_total_se [mS/cm]"],
        }

    out_file = out_dir / "results_wignerflow.json"
    out_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
