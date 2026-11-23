"""Evaluate a trained model on a molecule in vacuum (alanine dipeptide, aspirin) by long rollouts without an integrator.

    python scripts/evaluate_molecule.py --run results/ala2/B_S3a_seed1 --ckpt best.pt --steps 10000 --seeds 5 --starts 3

Differences from scripts/evaluate.py (liquid argon):
  - crash test uses bonds: a rollout crashes when a bond length is more than 50% off its reference length;
  - structure is judged by bond-length statistics and the intramolecular pair-distance density (no g(r) in vacuum);
  - alanine dipeptide also gets the phi/psi distribution, compared with the ground truth over a window of the same
    length from the same starting frame, and with the ground truth's own spread (two windows of the same length);
  - no diffusion coefficient.
Writes <run>/eval_mol/metrics.json and figures.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from pathlib import Path

from equitraj.evaluation import first_zero, load_model  # noqa: E402

from equitraj import analysis, plots  # noqa: E402
from equitraj.constraints import DistanceProjector  # noqa: E402
from equitraj.model import angular_momentum  # noqa: E402
from equitraj.pair_masks import exclusions_from_bonds  # noqa: E402
from equitraj.rollout import rollout, temperature  # noqa: E402

COVALENT_RADIUS = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66}  # Å
PHI = (1, 3, 4, 6)  # ala2 heavy atoms follow the SMILES CC(=O)N[C@@H](C)C(=O)NC (see scripts/analyze_gt.py)
PSI = (3, 4, 6, 8)


def find_bonds(x, elements, scale=1.25):
    """Pairs closer than scale x (sum of covalent radii) in one frame. Returns (n_bonds, 2) indices."""
    n = len(elements)
    r = np.array([COVALENT_RADIUS[e] for e in elements])
    d = np.linalg.norm(x[:, None] - x[None], axis=-1)
    i, j = np.where(np.triu(d < scale * (r[:, None] + r[None]), 1))
    return np.stack([i, j], 1)


def bond_lengths(x, bonds):
    """x: (T, N, 3) -> (T, n_bonds)."""
    return np.linalg.norm(x[:, bonds[:, 0]] - x[:, bonds[:, 1]], axis=-1)


def rama_hist(x, bins=36):
    phi = analysis.dihedral(*(x[:, i] for i in PHI))
    psi = analysis.dihedral(*(x[:, i] for i in PSI))
    h = np.histogram2d(phi, psi, bins=bins, range=[[-180, 180], [-180, 180]])[0]
    return h / max(h.sum(), 1.0), phi, psi


def js_divergence(p, q):
    """Jensen-Shannon divergence (nats) of two histograms that each sum to 1."""
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a[a > 0] * np.log(a[a > 0] / b[a > 0]))  # noqa: E731
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="training run directory (optional when --ckpt is a file path)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--data_dir", default=None, help="ground-truth data override; required for standalone checkpoints")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--thermostat_tau_fs", type=float, default=0.0, help=">0: CSVR switch S6 (reported separately)")
    ap.add_argument("--project_constraints", action="store_true", help="switch S4: RATTLE projection on constrained/X-H bonds")
    ap.add_argument("--zero_angular_momentum", action="store_true", help="project out rigid-body rotation in vacuum")
    ap.add_argument("--max_bond_dev", type=float, default=0.5, help="a rollout crashes when a bond is this fraction off")
    ap.add_argument("--min_dist", type=float, default=0.5, help="a rollout crashes when two atoms are closer than this (Å)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    if args.run is None and not os.path.isfile(args.ckpt):
        ap.error("--run is required unless --ckpt points to an existing checkpoint file")
    device = torch.device(args.device)
    tag = "eval_mol"
    if args.project_constraints:
        tag += "_s4"
    if args.thermostat_tau_fs > 0:
        tag += f"_csvr{int(args.thermostat_tau_fs)}"
    default_out = os.path.join(args.run, tag) if args.run is not None else os.path.join("results", f"{tag}_{Path(args.ckpt).stem}")
    out_dir = args.out or default_out
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    model, cfg, ds, step = load_model(args.run, args.ckpt, device, data_dir=args.data_dir)
    assert not ds.periodic, "evaluate_molecule.py is for molecules in vacuum; use evaluate.py for periodic systems"
    cfg_yaml = os.path.join(cfg["data_dir"], "config.yaml")
    system = "molecule"
    if os.path.isfile(cfg_yaml):
        with open(cfg_yaml, encoding="utf-8") as fh:
            system = (yaml.safe_load(fh) or {}).get("system", "molecule")
    k, f, dt = cfg["k"], ds.f, ds.stride_fs
    n_atoms = len(ds.types)
    elements = [str(e) for e in ds.meta["elements"]]
    deterministic = model.head_type == "regression"
    seeds = 1 if deterministic else args.seeds

    gt_x = [np.asarray(p[::f]) for p in ds.pos]
    gt_v = [np.asarray(v[::f]) for v in ds.vel]
    t_gt = float(np.mean([temperature(torch.from_numpy(v[::max(1, min(10, len(v)))].copy()), ds.masses, ds.n_constraints).mean().item() for v in gt_v]))
    bonds = find_bonds(gt_x[0][0], elements)
    excl = exclusions_from_bonds(bonds, n_atoms, max_hops=2) if len(bonds) > 0 else None
    r0 = np.concatenate([bond_lengths(x[:2000], bonds) for x in gt_x]).mean(0) if len(bonds) > 0 else np.zeros(0)

    projector = None
    if args.project_constraints and len(bonds) > 0:
        meta_con = (getattr(ds, "particle_metadata", None) or {}).get("constraints") or []
        if meta_con:
            con_rows = [[int(i), int(j), float(d_nm) * 10.0] for i, j, d_nm in meta_con]
        else:
            xh = [(int(bonds[idx, 0]), int(bonds[idx, 1]), float(r0[idx]))
                  for idx in range(len(bonds)) if "H" in (elements[bonds[idx, 0]], elements[bonds[idx, 1]])]
            con_rows = xh or [[int(bonds[idx, 0]), int(bonds[idx, 1]), float(r0[idx])] for idx in range(len(bonds))]
        projector = DistanceProjector(con_rows)

    starts = []
    for i in range(args.starts):
        traj = i % len(ds.pos)
        n_avail = len(gt_x[traj]) - k - 1 - args.steps  # leave room for a ground-truth window of the same length
        frac = (i // len(ds.pos)) / max(1, int(np.ceil(args.starts / len(ds.pos))))
        starts.append((traj, int(frac * max(n_avail, 0))))
    pos = torch.stack([torch.from_numpy(gt_x[t][s:s + k].copy()) for t, s in starts]).repeat_interleave(seeds, 0)
    vel = torch.stack([torch.from_numpy(gt_v[t][s:s + k].copy()) for t, s in starts]).repeat_interleave(seeds, 0)
    labels = [(t, s) for (t, s) in starts for _ in range(seeds)]
    torch.manual_seed(0)
    thermostat = {"t_target": t_gt, "tau_fs": args.thermostat_tau_fs, "dt_fs": dt} if args.thermostat_tau_fs > 0 else None
    out = rollout(model, pos, vel, ds.types, None, ds.masses, args.steps, sampling_steps=args.sampling_steps,
                  save_every=1, t_max=3 * t_gt, bonds=torch.from_numpy(bonds), bond_r0=torch.from_numpy(r0).float(),
                  max_bond_dev=args.max_bond_dev, min_dist=args.min_dist, excluded_pairs=excl,
                  thermostat=thermostat, constraint_projector=projector,
                  zero_angular_momentum=args.zero_angular_momentum)
    x, v, crash = out["positions"].numpy(), out["velocities"].numpy(), out["crash_step"].numpy()
    n_roll, n_done = len(labels), x.shape[0]
    ok = crash < 0
    m = {"run": args.run, "checkpoint": args.ckpt, "system": system, "train_step": int(step), "head": model.head_type,
         "target": model.target, "stride_fs": dt, "steps": args.steps, "n_rollouts": n_roll, "seeds_per_start": seeds,
         "switches": {"S4_project_constraints": bool(projector is not None), "S6_thermostat_tau_fs": args.thermostat_tau_fs,
                      "zero_angular_momentum": bool(args.zero_angular_momentum)},
         "T_ground_truth_K": t_gt, "n_bonds": len(bonds)}
    steps_before = np.where(ok, args.steps, crash)
    m["stability"] = {"crashed": int((~ok).sum()), "steps_before_crash": steps_before.tolist(),
                      "median_steps_before_crash": float(np.median(steps_before))}
    surv_t = np.arange(0, args.steps + 1, max(1, args.steps // 200))
    plots.line_plot(os.path.join(out_dir, "figs", "survival.png"),
                    [("model", surv_t * dt / 1000, [(steps_before >= s).mean() for s in surv_t])],
                    "rollout time (ps)", "fraction not crashed", "Rollout stability", ylim=(0, 1.05))

    temp = temperature(torch.from_numpy(v.reshape(-1, n_atoms, 3)), ds.masses, ds.n_constraints).numpy().reshape(n_done, n_roll)
    half = args.steps // 2
    if not ok.any():
        m["note"] = "every rollout crashed; structural and dynamical metrics skipped"
        with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(m, fh, indent=2)
        print(json.dumps(m, indent=2))
        return
    m["temperature"] = {"T_model_second_half_K": float(temp[half:, ok].mean()),
                        "rel_err": float(abs(temp[half:, ok].mean() - t_gt) / t_gt)}
    t_axis = np.arange(1, n_done + 1) * dt / 1000
    plots.line_plot(os.path.join(out_dir, "figs", "temperature.png"),
                    [("ground truth", t_axis[::20], np.full(len(t_axis[::20]), t_gt))]
                    + [(f"rollout {j}", t_axis[::20], temp[::20, j]) for j in range(min(n_roll, 4))],
                    "rollout time (ps)", "T (K)", "Temperature (model only)")

    # ---- bond lengths: mean error and spread ratio per bond, against the ground-truth test frames ----
    b_gt = np.concatenate([bond_lengths(xx, bonds) for xx in gt_x])
    b_m = np.concatenate([bond_lengths(x[half:, j], bonds) for j in np.where(ok)[0]])
    m["bonds"] = {"mean_abs_err_A": float(np.abs(b_m.mean(0) - b_gt.mean(0)).mean()),
                  "max_abs_err_A": float(np.abs(b_m.mean(0) - b_gt.mean(0)).max()),
                  "std_ratio_median": float(np.median(b_m.std(0) / b_gt.std(0))),
                  "std_ratio_range": [float((b_m.std(0) / b_gt.std(0)).min()), float((b_m.std(0) / b_gt.std(0)).max())]}

    # ---- intramolecular pair-distance density (all atoms) ----
    idx = np.arange(n_atoms)
    sub = max(1, (n_done - half) // 400)
    g_m = np.mean([analysis.rdf(x[half::sub, j], None, idx, idx, 8.0, 160)[1] for j in np.where(ok)[0]], axis=0)
    r, _, _ = analysis.rdf(gt_x[0][:10], None, idx, idx, 8.0, 160)
    g_g = np.mean([analysis.rdf(xx[::max(1, len(xx) // 400)], None, idx, idx, 8.0, 160)[1] for xx in gt_x], axis=0)
    m["pair_distance"] = {"l1": float(analysis.rdf_l1(r, g_m, g_g))}
    plots.line_plot(os.path.join(out_dir, "figs", "pair_distance.png"), [("ground truth", r, g_g), ("model", r, g_m)],
                    "r (Å)", "p(r)", "Intramolecular pair-distance density")

    # ---- linear and angular momentum drift in vacuum ----
    p_norms, l_drifts = [], []
    for j in np.where(ok)[0]:
        xj = torch.from_numpy(x[:, j])
        vj = torch.from_numpy(v[:, j])
        pj = (vj.double() * ds.masses.double()[None, :, None]).sum(dim=1)
        lj = angular_momentum(xj, vj, ds.masses)
        l0 = angular_momentum(pos[j, -1:], vel[j, -1:], ds.masses)[0]
        p_norms.append(float(pj.norm(dim=-1).max().item()))
        l_drifts.append(float((lj - l0[None]).norm(dim=-1).max().item()))
    m["momentum"] = {
        "max_linear_momentum_amu_A_fs": float(max(p_norms)),
        "max_angular_momentum_drift_amu_A2_fs": float(max(l_drifts)),
    }

    # ---- VACF / VDOS at the model stride ----
    max_lag = max(1, min(max(1, n_done // 4), int(max(1, 2000 / dt)), n_done - half))
    c_m = np.mean([analysis.vacf(v[half:, j], max_lag) for j in np.where(ok)[0]], axis=0)
    c_g = np.mean([analysis.vacf(vv, max_lag) for vv in gt_v], axis=0)
    lag_fs = np.arange(max_lag) * dt
    zm = first_zero(lag_fs, c_m / c_m[0]) if c_m[0] != 0 else None
    zg = first_zero(lag_fs, c_g / c_g[0]) if c_g[0] != 0 else None
    km, sm = analysis.vdos(c_m, dt)
    kg, sg = analysis.vdos(c_g, dt)
    m["vacf"] = {"first_zero_model_fs": None if zm is None else float(zm), "first_zero_gt_fs": None if zg is None else float(zg),
                 "first_zero_rel_err": None if (zm is None or zg is None) else float(abs(zm - zg) / zg),
                 "vdos_l1": float(np.abs(sm / max(sm.sum(), 1e-30) - sg / max(sg.sum(), 1e-30)).sum()), "nyquist_cm1": float(km[-1])}
    plots.line_plot(os.path.join(out_dir, "figs", "vacf.png"),
                    [("ground truth (same stride)", lag_fs / 1000, c_g / max(abs(c_g[0]), 1e-30)),
                     ("model", lag_fs / 1000, c_m / max(abs(c_m[0]), 1e-30))],
                    "t (ps)", "C(t) / C(0)", "Velocity autocorrelation", hline=0.0)
    plots.line_plot(os.path.join(out_dir, "figs", "vdos.png"),
                    [("ground truth (same stride)", kg, sg / max(sg.max(), 1e-30)),
                     ("model", km, sm / max(sm.max(), 1e-30))],
                    "wavenumber (cm⁻¹)", "VDOS (normalized)", "Vibrational density of states")

    # ---- alanine dipeptide: phi/psi against the same-length ground-truth window and the ground truth's own spread ----
    if system == "ala2":
        gt_win = [gt_x[t][s + k:s + k + args.steps] for t, s in labels]
        h_m = np.mean([rama_hist(x[:, j])[0] for j in np.where(ok)[0]], axis=0)
        js_model = np.mean([js_divergence(rama_hist(x[:, j])[0], rama_hist(gt_win[j])[0]) for j in np.where(ok)[0]])
        # the ground truth's own spread: this window against the next one of the same length from the same trajectory
        own = []
        for t, s in set(labels):
            a, b = gt_x[t][s + k:s + k + args.steps], gt_x[t][s + k + args.steps:s + k + 2 * args.steps]
            if len(b) == args.steps:
                own.append(js_divergence(rama_hist(a)[0], rama_hist(b)[0]))
        m["ramachandran"] = {"js_model_vs_matched_gt_window": float(js_model),
                             "js_gt_vs_gt_own_spread": float(np.mean(own)) if own else None}
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, (title, frames) in zip(axes, [("ground truth (matched window)", np.concatenate(gt_win)),
                                              ("model", np.concatenate([x[:, j] for j in np.where(ok)[0]]))]):
            _, phi, psi = rama_hist(frames)
            ax.hist2d(phi, psi, bins=72, range=[[-180, 180], [-180, 180]], cmap="Blues", cmin=1)
            ax.set_xlabel("φ (deg)")
            ax.set_title(title, loc="left", fontsize=10)
        axes[0].set_ylabel("ψ (deg)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "figs", "ramachandran.png"), dpi=150)
        plt.close(fig)

    wall_s = max(float(out["wall_s"]), 1e-9)
    m["speed"] = {"wall_s": out["wall_s"], "ns_per_day_all_rollouts": n_roll * out["steps_done"] * dt * 1e-6 / wall_s * 86400,
                  "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)}
    # criteria for a molecule at this stage (v1): reported, not tuned
    c = {"no_crash": bool(ok.all()), "temperature_5pct": m["temperature"]["rel_err"] < 0.05,
         "bond_mean_err_0.01A": m["bonds"]["mean_abs_err_A"] < 0.01,
         "bond_std_ratio_0.5_2": 0.5 < m["bonds"]["std_ratio_median"] < 2.0}
    c["all_pass"] = all(c.values())
    m["criteria"] = c
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps({key: val for key, val in m.items() if key != "stability"}, indent=2))


if __name__ == "__main__":
    main()
