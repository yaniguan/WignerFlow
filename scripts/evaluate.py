"""Evaluate a trained model by long rollouts that replace the integrator (no OpenMM).

    python scripts/evaluate.py --run results/argon/B_S3a_seed1 --steps 10000 --seeds 5 --starts 3

For each test starting window and seed: roll out --steps steps, then compare with the ground-truth
test trajectories (statistically; trajectories decorrelate after ~1 ps):
    stability (steps before a crash), temperature, RDF, VACF/VDOS (ground truth subsampled to the
    model stride), MSD/diffusion, short-range error, speed.
Writes <run>/eval/metrics.json and figures, and checks the v0 criteria for argon (PLAN.md).
"""

import argparse
import json
import os

import numpy as np
import torch

from equitraj import analysis, plots
from equitraj.dataset import TrajWindows
from equitraj.model import EquiTraj
from equitraj.rollout import rollout, temperature

V0_CRITERIA = {  # argon, model only (no thermostat)
    "no_crash": "all rollouts reach the full number of steps",
    "temperature_rel_err": 0.05,
    "rdf_peak_pos_err_A": 0.05,
    "rdf_peak_height_rel_err": 0.10,
    "vacf_first_zero_rel_err": 0.10,
    "diffusion_rel_err": 0.20,
}


def first_zero(t, c):
    """First time where c crosses zero, linearly interpolated (None if it never does)."""
    i = np.argmax(c < 0)
    if c[i] >= 0:
        return None
    return t[i - 1] + (t[i] - t[i - 1]) * c[i - 1] / (c[i - 1] - c[i])


def load_model(run_dir, ckpt, device, split="test"):
    ck = torch.load(os.path.join(run_dir, ckpt), map_location=device, weights_only=False)
    cfg = ck["cfg"]
    ds = TrajWindows(cfg["data_dir"], split, cfg["stride_steps"], cfg["k"], m=1)
    model = EquiTraj(n_types=len(ck["elements"]), k=cfg["k"], stride_fs=ds.stride_fs, **cfg["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ds, ck["step"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--thermostat_tau_fs", type=float, default=0.0, help=">0: CSVR switch S6 (reported separately)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--split", default="test", help="test for reporting; val for choosing among checkpoints")
    args = ap.parse_args()
    device = torch.device(args.device)
    tag = "eval" if args.thermostat_tau_fs <= 0 else f"eval_csvr{int(args.thermostat_tau_fs)}"
    out_dir = args.out or os.path.join(args.run, tag)
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    model, cfg, ds, step = load_model(args.run, args.ckpt, device, args.split)
    k, f, dt = cfg["k"], ds.f, ds.stride_fs
    n_atoms = len(ds.types)
    deterministic = model.head_type == "regression"
    seeds = 1 if deterministic else args.seeds  # the regression head gives the same rollout for every seed

    # ---- ground-truth reference at the model stride, from the test trajectories ----
    gt_x = [np.asarray(p[::f]) for p in ds.pos]  # each (T, N, 3), frames dt apart
    gt_v = [np.asarray(v[::f]) for v in ds.vel]
    gt_box = [b.numpy() for b in ds.boxes]
    t_gt = float(np.mean([temperature(torch.from_numpy(v[::10].copy()), ds.masses).mean().item() for v in gt_v]))

    # ---- starting windows: evenly spread over the test trajectories ----
    starts = []
    for i in range(args.starts):
        traj = i % len(ds.pos)
        n_avail = len(gt_x[traj]) - k - 1
        frac = (i // len(ds.pos)) / max(1, int(np.ceil(args.starts / len(ds.pos))))
        starts.append((traj, int(frac * n_avail)))
    pos0 = torch.stack([torch.from_numpy(gt_x[t][s:s + k].copy()) for t, s in starts])
    vel0 = torch.stack([torch.from_numpy(gt_v[t][s:s + k].copy()) for t, s in starts])
    box0 = torch.stack([ds.boxes[t] for t, _ in starts]) if ds.periodic else None

    # one batch with all (start, seed) pairs
    pos = pos0.repeat_interleave(seeds, 0)
    vel = vel0.repeat_interleave(seeds, 0)
    box = None if box0 is None else box0.repeat_interleave(seeds, 0)
    labels = [(t, s, seed) for (t, s) in starts for seed in range(seeds)]
    torch.manual_seed(0)
    thermostat = None
    if args.thermostat_tau_fs > 0:
        thermostat = {"t_target": t_gt, "tau_fs": args.thermostat_tau_fs, "dt_fs": dt}
    out = rollout(model, pos, vel, ds.types, box, ds.masses, args.steps, sampling_steps=args.sampling_steps,
                  save_every=1, t_max=3 * t_gt, thermostat=thermostat)
    x = out["positions"].numpy()   # (steps, B, N, 3)
    v = out["velocities"].numpy()
    crash = out["crash_step"].numpy()
    n_roll = len(labels)
    ok = crash < 0
    m = {"run": args.run, "checkpoint": args.ckpt, "split": args.split, "train_step": int(step), "head": model.head_type,
         "target": model.target, "stride_fs": dt, "steps": args.steps, "n_rollouts": n_roll,
         "seeds_per_start": seeds, "thermostat_tau_fs": args.thermostat_tau_fs, "T_ground_truth_K": t_gt}

    # ---- stability ----
    steps_before = np.where(ok, args.steps, crash)
    m["stability"] = {"crashed": int((~ok).sum()), "steps_before_crash": steps_before.tolist(),
                      "median_steps_before_crash": float(np.median(steps_before))}
    surv_t = np.arange(0, args.steps + 1, max(1, args.steps // 200))
    survival = [(steps_before >= s).mean() for s in surv_t]
    plots.line_plot(os.path.join(out_dir, "figs", "survival.png"), [("model", surv_t * dt / 1000, survival)],
                    "rollout time (ps)", "fraction not crashed", "Rollout stability", ylim=(0, 1.05))

    # ---- temperature ----
    temp = temperature(torch.from_numpy(v.reshape(-1, n_atoms, 3)), ds.masses).numpy().reshape(args.steps, n_roll)
    t_axis = np.arange(1, args.steps + 1) * dt / 1000
    every = max(1, args.steps // 500)
    curves = [("ground truth", t_axis[::every], np.full(len(t_axis[::every]), t_gt))]
    for j in range(min(n_roll, 4)):
        curves.append((f"rollout {j}", t_axis[::every], temp[::every, j]))
    plots.line_plot(os.path.join(out_dir, "figs", "temperature.png"), curves, "rollout time (ps)", "T (K)",
                    "Temperature (model only)" if thermostat is None else "Temperature (CSVR on)")
    half = args.steps // 2
    if ok.any():
        t_model = float(temp[half:, ok].mean())
        m["temperature"] = {"T_model_second_half_K": t_model, "rel_err": abs(t_model - t_gt) / t_gt,
                            "T_end_per_rollout": temp[-1].tolist()}

    # ---- short-range error (first 20 steps vs the ground-truth continuation) ----
    n_short = min(20, args.steps)
    err = []
    for j, (t, s, _) in enumerate(labels):
        ref = gt_x[t][s + k:s + k + n_short]
        err.append(np.sqrt(((x[:len(ref), j] - ref) ** 2).sum(-1).mean(-1)))
    err = np.mean(err, axis=0)
    m["short_range_rmsd_A"] = {"step_1": float(err[0]), "step_10": float(err[min(9, len(err) - 1)]),
                               "step_20": float(err[-1])}
    plots.line_plot(os.path.join(out_dir, "figs", "short_error.png"), [("model", np.arange(1, len(err) + 1), err)],
                    "rollout step", "RMSD to ground truth (Å)", "Short-range error")

    if not ok.any():
        m["note"] = "every rollout crashed; structural and dynamical metrics skipped"
        json.dump(m, open(os.path.join(out_dir, "metrics.json"), "w"), indent=2)
        print(json.dumps(m, indent=2))
        return

    # ---- structure: RDF over the second half of all non-crashed rollouts ----
    idx = np.arange(n_atoms)
    stride_rdf = max(1, args.steps // 400)
    r_max = min(10.0, 0.49 * float(np.diag(gt_box[0]).min())) if ds.periodic else 8.0
    g_model = []
    for j in np.where(ok)[0]:
        frames = x[half::stride_rdf, j]
        bx = None if box is None else np.tile(box[j].numpy(), (len(frames), 1, 1))
        r, g, _ = analysis.rdf(frames, bx, idx, idx, r_max, 200)
        g_model.append(g)
    g_model = np.mean(g_model, axis=0)
    g_gt = []
    for t in range(len(gt_x)):
        frames = gt_x[t][::max(1, len(gt_x[t]) // 400)]
        bx = None if box is None else np.tile(gt_box[t], (len(frames), 1, 1))
        _, g, _ = analysis.rdf(frames, bx, idx, idx, r_max, 200)
        g_gt.append(g)
    g_gt = np.mean(g_gt, axis=0)
    pm, hm = analysis.first_peak(r, g_model)
    pg, hg = analysis.first_peak(r, g_gt)
    m["rdf"] = {"l1": float(analysis.rdf_l1(r, g_model, g_gt)), "peak_pos_model_A": float(pm), "peak_pos_gt_A": float(pg),
                "peak_pos_err_A": float(abs(pm - pg)), "peak_height_model": float(hm), "peak_height_gt": float(hg),
                "peak_height_rel_err": float(abs(hm - hg) / hg)}
    plots.line_plot(os.path.join(out_dir, "figs", "rdf.png"), [("ground truth", r, g_gt), ("model", r, g_model)],
                    "r (Å)", "g(r)", "RDF, second half of rollouts")

    # ---- dynamics: VACF / VDOS at the model stride ----
    max_lag = min(args.steps // 4, int(2000 / dt))
    c_model = np.mean([analysis.vacf(v[half:, j], max_lag) for j in np.where(ok)[0]], axis=0)
    c_gt = np.mean([analysis.vacf(vv[:len(vv) // 2 * 2], max_lag) for vv in gt_v], axis=0)
    lag_fs = np.arange(max_lag) * dt
    zm, zg = first_zero(lag_fs, c_model / c_model[0]), first_zero(lag_fs, c_gt / c_gt[0])
    km, sm = analysis.vdos(c_model, dt)
    kg, sg = analysis.vdos(c_gt, dt)
    zm = None if zm is None else float(zm)
    zg = None if zg is None else float(zg)
    m["vacf"] = {"first_zero_model_fs": zm, "first_zero_gt_fs": zg,
                 "first_zero_rel_err": None if (zm is None or zg is None) else float(abs(zm - zg) / zg),
                 "vdos_peak_model_cm1": float(km[np.argmax(sm)]), "vdos_peak_gt_cm1": float(kg[np.argmax(sg)]),
                 "nyquist_cm1": float(km[-1]),
                 "note": f"both sampled every {dt} fs, so frequencies above {km[-1]:.0f} cm^-1 are not resolved"}
    plots.line_plot(os.path.join(out_dir, "figs", "vacf.png"),
                    [("ground truth (same stride)", lag_fs / 1000, c_gt / c_gt[0]), ("model", lag_fs / 1000, c_model / c_model[0])],
                    "t (ps)", "C(t) / C(0)", "Velocity autocorrelation", hline=0.0)
    plots.line_plot(os.path.join(out_dir, "figs", "vdos.png"), [("ground truth (same stride)", kg, sg), ("model", km, sm)],
                    "wavenumber (cm⁻¹)", "VDOS (normalized)", "Vibrational density of states", xlim=(0, min(km[-1], 200)))

    # ---- diffusion (periodic systems) ----
    if ds.periodic:
        max_lag_msd = min(args.steps // 2, int(20000 / dt))
        lags = np.unique(np.linspace(0, max_lag_msd, 100).astype(int))
        msd_model = np.mean([analysis.msd(x[:, j], lags) for j in np.where(ok)[0]], axis=0)
        msd_gt = np.mean([analysis.msd(xx, lags) for xx in gt_x], axis=0)
        t_lag = lags * dt
        fit = (0.25 * t_lag[-1], t_lag[-1])
        dm = analysis.diffusion_coefficient(t_lag, msd_model, *fit)[1]
        dg = analysis.diffusion_coefficient(t_lag, msd_gt, *fit)[1]
        m["diffusion"] = {"D_model_cm2_s": float(dm), "D_gt_cm2_s": float(dg), "rel_err": float(abs(dm - dg) / dg),
                          "fit_range_fs": list(fit)}
        plots.line_plot(os.path.join(out_dir, "figs", "msd.png"), [("ground truth", t_lag / 1000, msd_gt), ("model", t_lag / 1000, msd_model)],
                        "t (ps)", "MSD (Å²)", "Mean squared displacement")

    # ---- speed ----
    m["speed"] = {"wall_s": out["wall_s"], "s_per_step_batch": out["wall_s"] / out["steps_done"],
                  "ns_per_day_per_rollout": out["steps_done"] * dt * 1e-6 / out["wall_s"] * 86400,
                  "ns_per_day_all_rollouts": n_roll * out["steps_done"] * dt * 1e-6 / out["wall_s"] * 86400,
                  "device": torch.cuda.get_device_name(0) if device.type == "cuda" else str(device)}

    # ---- v0 verdict ----
    verdict = {
        "no_crash": bool(ok.all()),
        "temperature": m.get("temperature", {}).get("rel_err", 1.0) < V0_CRITERIA["temperature_rel_err"],
        "rdf_peak_position": m["rdf"]["peak_pos_err_A"] < V0_CRITERIA["rdf_peak_pos_err_A"],
        "rdf_peak_height": m["rdf"]["peak_height_rel_err"] < V0_CRITERIA["rdf_peak_height_rel_err"],
        "vacf_first_zero": m["vacf"]["first_zero_rel_err"] is not None and m["vacf"]["first_zero_rel_err"] < V0_CRITERIA["vacf_first_zero_rel_err"],
        "diffusion": m.get("diffusion", {}).get("rel_err", 1.0) < V0_CRITERIA["diffusion_rel_err"],
    }
    verdict = {key: bool(val) for key, val in verdict.items()}
    verdict["all_pass"] = all(verdict.values())
    m["v0_criteria"] = V0_CRITERIA
    m["v0_verdict"] = verdict
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(m, fh, indent=2)
    print(json.dumps({key: val for key, val in m.items() if key not in ("stability",)}, indent=2))


if __name__ == "__main__":
    main()
