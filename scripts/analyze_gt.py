"""Analyze ground-truth trajectories: energy, temperature, RDF, VACF/VDOS, MSD/diffusion.

    python scripts/analyze_gt.py --data data/argon --out results/argon/ground_truth

Writes metrics.json and figures to --out.
"""

import argparse
import glob
import json
import os

import numpy as np

from equitraj import analysis, plots
from equitraj.trajfile import read_traj
from equitraj.units import NM_PER_PS_TO_A_PER_FS, NM_TO_A, PS_TO_FS


def block_vacf(v, max_lag, block):
    """VACF averaged over non-overlapping blocks (keeps the FFT memory small)."""
    out = []
    for s in range(0, len(v) - block + 1, block):
        out.append(analysis.vacf(v[s : s + block], max_lag))
    return np.mean(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rdf_stride", type=int, default=50, help="use every n-th frame for the RDF")
    ap.add_argument("--vacf_ps", type=float, default=2.0)
    ap.add_argument("--msd_ps", type=float, default=20.0)
    ap.add_argument("--diff_fit_ps", type=float, nargs=2, default=[5.0, 20.0])
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "figs"), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.data, "traj_*.h5")))
    per_traj, rdfs, vacfs, msds = [], [], [], []
    for path in files:
        t = read_traj(path)
        a = t["attrs"]
        if not a["complete"]:
            print(f"skip incomplete {path}")
            continue
        n_atoms = len(t["masses"])
        dt_fs = float(a["frame_interval_fs"])
        dof = analysis.n_dof(n_atoms, int(a["n_constraints"]))
        x = t["positions"] * NM_TO_A
        v = t["velocities"] * NM_PER_PS_TO_A_PER_FS
        box = t["box"] * NM_TO_A
        time_ps = t["time"]
        e_tot = t["potential_energy"] + t["kinetic_energy"]
        temp = analysis.temperature(t["kinetic_energy"], dof)

        # structure: RDF over all atoms of each element pair (argon: Ar-Ar only)
        elements = t["elements"]
        species = sorted(set(elements))
        r_max = min(12.0, 0.5 * box[0].diagonal().min())
        rdf_pairs = {}
        for i, s1 in enumerate(species):
            for s2 in species[i:]:
                ia, ib = np.where(elements == s1)[0], np.where(elements == s2)[0]
                r, g, rho = analysis.rdf(x[:: args.rdf_stride], box[:: args.rdf_stride], ia, ib, r_max)
                rdf_pairs[f"{s1}-{s2}"] = (r, g, rho)
        rdfs.append(rdf_pairs)

        # dynamics
        max_lag = int(round(args.vacf_ps * PS_TO_FS / dt_fs))
        vacfs.append(block_vacf(v, max_lag, block=10 * max_lag))
        msd_stride = max(1, int(round(50.0 / dt_fs)))  # positions every ~50 fs
        lags = np.arange(0, int(round(args.msd_ps * PS_TO_FS / (dt_fs * msd_stride))) + 1)
        msds.append(analysis.msd(x[::msd_stride], lags))
        msd_dt_fs = dt_fs * msd_stride

        per_traj.append({
            "file": os.path.basename(path),
            "seed": int(a["seed"]),
            "T_mean_K": float(temp.mean()),
            "T_std_K": float(temp.std()),
            "E_total_std_kJmol": float(e_tot.std()),
            "energy_drift_kJmol_per_ns_per_dof": float(analysis.energy_drift(time_ps, e_tot, dof)),
            "ns_per_day_with_io": float(a["ns_per_day_with_io"]),
            "platform": str(a["platform"]),
            "precision": str(a["precision"]),
        })
        print(json.dumps(per_traj[-1]), flush=True)
        e0 = e_tot - e_tot[0]
        per_traj[-1]["_energy_curve"] = (time_ps[::100], e0[::100])

    # --- aggregate ---
    metrics = {"n_traj": len(per_traj), "frame_interval_fs": dt_fs, "per_traj": [
        {k: v for k, v in p.items() if not k.startswith("_")} for p in per_traj]}
    metrics["T_mean_K"] = float(np.mean([p["T_mean_K"] for p in per_traj]))
    drifts = [p["energy_drift_kJmol_per_ns_per_dof"] for p in per_traj]
    metrics["energy_drift_kJmol_per_ns_per_dof"] = {"mean": float(np.mean(drifts)), "max_abs": float(np.max(np.abs(drifts)))}

    metrics["rdf"] = {}
    for pair in rdfs[0]:
        r = rdfs[0][pair][0]
        gs = np.array([d[pair][1] for d in rdfs])
        rho = rdfs[0][pair][2]
        g_mean = gs.mean(axis=0)
        peak_r, peak_g = analysis.first_peak(r, g_mean)
        r_min = analysis.first_minimum_after_peak(r, g_mean)
        cn = analysis.coordination_number(r, g_mean, rho, r_min)
        spread = [analysis.rdf_l1(r, g, g_mean) for g in gs]
        metrics["rdf"][pair] = {
            "first_peak_A": float(peak_r), "first_peak_height": float(peak_g),
            "first_min_A": float(r_min), "coordination_number": float(cn),
            "per_traj_L1_to_mean": {"median": float(np.median(spread)), "max": float(np.max(spread))},
        }
        np.savetxt(os.path.join(args.out, f"rdf_{pair}.txt"), np.c_[r, g_mean, gs.min(0), gs.max(0)],
                   header="r_A g_mean g_min g_max")
        plots.band_plot(os.path.join(args.out, "figs", f"rdf_{pair}.png"), r, g_mean, gs.min(0), gs.max(0),
                        "ground truth", "r (Å)", "g(r)", f"RDF {pair}")

    c = np.mean(vacfs, axis=0)
    t_fs = np.arange(len(c)) * dt_fs
    cn_ = c / c[0]
    zero = t_fs[np.argmax(cn_ < 0)] if np.any(cn_ < 0) else None
    i_min = int(np.argmin(cn_))
    k, s = analysis.vdos(c, dt_fs)
    metrics["vacf"] = {
        "first_zero_fs": float(zero) if zero is not None else None,
        "min_value_normalized": float(cn_[i_min]), "min_time_fs": float(t_fs[i_min]),
        "vdos_peak_cm1": float(k[np.argmax(s)]),
        "nyquist_cm1": float(k[-1]),
    }
    np.savetxt(os.path.join(args.out, "vacf.txt"), np.c_[t_fs, cn_], header="t_fs vacf_normalized")
    plots.line_plot(os.path.join(args.out, "figs", "vacf.png"), [("ground truth", t_fs / 1000, cn_)],
                    "t (ps)", "C(t) / C(0)", "Velocity autocorrelation", hline=0.0)
    kmax = min(k[-1], 5 * metrics["vacf"]["vdos_peak_cm1"] + 50)
    plots.line_plot(os.path.join(args.out, "figs", "vdos.png"), [("ground truth", k, s)],
                    "wavenumber (cm⁻¹)", "VDOS (normalized)", "Vibrational density of states", xlim=(0, kmax))

    m = np.mean(msds, axis=0)
    t_msd = np.arange(len(m)) * msd_dt_fs
    d, d_cm2 = analysis.diffusion_coefficient(t_msd, m, args.diff_fit_ps[0] * 1000, args.diff_fit_ps[1] * 1000)
    d_each = [analysis.diffusion_coefficient(t_msd, mm, args.diff_fit_ps[0] * 1000, args.diff_fit_ps[1] * 1000)[1] for mm in msds]
    metrics["diffusion"] = {"D_cm2_per_s": float(d_cm2), "D_std_over_traj": float(np.std(d_each)),
                            "fit_range_ps": args.diff_fit_ps}
    plots.line_plot(os.path.join(args.out, "figs", "msd.png"), [("ground truth", t_msd / 1000, m)],
                    "t (ps)", "MSD (Å²)", "Mean squared displacement")

    plots.line_plot(os.path.join(args.out, "figs", "energy.png"),
                    [(f"traj {i}", p["_energy_curve"][0], p["_energy_curve"][1]) for i, p in enumerate(per_traj[:6])],
                    "t (ps)", "E_total − E_total(0) (kJ/mol)", "NVE total energy (first 6 trajectories)")

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_traj"}, indent=2))


if __name__ == "__main__":
    main()
