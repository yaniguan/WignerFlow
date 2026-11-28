"""Rebuild RESULTS.md from results/*/ground_truth/metrics.json and results/*/*/eval*/metrics.json.

    python scripts/make_results.py
"""

import glob
import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml


def _read_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_yaml(path):
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def fmt(x, spec=".3g"):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "pass" if x else "**fail**"
    return format(x, spec)


def ground_truth_rows():
    rows = []
    for path in sorted(glob.glob("results/*/ground_truth/metrics.json")):
        system = Path(path).parts[1]
        m = _read_json(path)
        pt = m["per_traj"]
        first_pair = next(iter(m["rdf"]))
        rdf = m["rdf"][first_pair]
        d = m.get("diffusion", {}).get("D_cm2_per_s")
        rows.append(f"| {system} | {m['n_traj']} | {m['T_mean_K']:.1f} | {fmt(m['energy_drift_kJmol_per_ns_per_dof']['mean'])} "
                    f"(max {fmt(m['energy_drift_kJmol_per_ns_per_dof']['max_abs'])}) | {first_pair}: {rdf['first_peak_A']:.2f} Å, "
                    f"{rdf['first_peak_height']:.2f} | {fmt(d)} | {pt[0]['ns_per_day_with_io']:.0f} ({pt[0]['platform']}) |")
    return rows


def _gt_energy_drifts():
    drifts = {}
    for path in sorted(glob.glob("results/*/ground_truth/metrics.json")):
        system = Path(path).parts[1]
        m = _read_json(path)
        ed = m.get("energy_drift_kJmol_per_ns_per_dof", {})
        if "mean" in ed:
            drifts[system] = float(ed["mean"])
    return drifts


def eval_rows():
    gt_drifts = _gt_energy_drifts()
    rows = []
    for path in sorted(glob.glob("results/*/*/eval*/metrics.json")):
        parts = Path(path).parts
        m = _read_json(path)
        if m.get("split", "test") != "test":
            continue  # validation-set evaluations are only for choosing checkpoints
        v = m.get("v0_verdict", {})
        system = parts[1]
        name = f"{parts[1]}/{parts[2]}" + ("" if parts[3] == "eval" else f" ({parts[3]})") + f", step {m.get('train_step', '?')}"
        st = m["stability"]
        cfg_path = os.path.join(*parts[:3], "config.yaml")
        cfg = _read_yaml(cfg_path) if os.path.exists(cfg_path) else {}
        target = cfg.get("model", {}).get("target", m.get("target", "?"))
        seed = cfg.get("seed", "?")
        dev = str(m.get("speed", {}).get("device", ""))
        is_cpu = dev.lower() == "cpu" or "cpu" in parts[3].lower()
        rollout_label = f"{m['n_rollouts']} × {m['steps']}" + (" (CPU)" if is_cpu else "")
        t_m = m.get("temperature", {}).get("T_model_second_half_K")
        t_err = None if t_m is None else (t_m - m["T_ground_truth_K"]) / m["T_ground_truth_K"]
        en = m.get("energy")
        if isinstance(en, dict) and en.get("H_drift_kJ_mol_ns_dof_mean") is not None:
            gt_d = gt_drifts.get(system)
            gt_suffix = f" (gt {fmt(gt_d)})" if gt_d is not None else ""
            h_drift_str = f"{fmt(en['H_drift_kJ_mol_ns_dof_mean'], '+.3g')}{gt_suffix}"
            du_str = fmt(en.get("U_model_minus_gt_per_atom"), "+.3g")
            dk_str = fmt(en.get("K_model_minus_gt_per_atom"), "+.3g")
        else:
            h_drift_str = "not measured"
            du_str = "not measured"
            dk_str = "not measured"
        rows.append(
            f"| {name} | {m['head']} | {target} | {seed} | {m['stride_fs']:.0f} | {rollout_label} | "
            f"{m['n_rollouts'] - st['crashed']}/{m['n_rollouts']} | "
            f"{fmt(t_m, '.1f')} / {m['T_ground_truth_K']:.1f} ({fmt(t_err, '+.1%')}) | "
            f"{h_drift_str} | {du_str} | {dk_str} | "
            f"{fmt(m.get('rdf', {}).get('peak_pos_err_A'))} | {fmt(m.get('rdf', {}).get('peak_height_rel_err'))} | "
            f"{fmt(m.get('vacf', {}).get('first_zero_rel_err'))} | {fmt(m.get('diffusion', {}).get('rel_err'))} | "
            f"{fmt(v.get('all_pass')) if v else '–'} |")
    return rows


def stability_figure(path="results/argon/stability_vs_step.png"):
    """Validation rollouts (1000 steps, every 2000 training steps) of the flow-head residual runs, one line per seed.

    Top: final temperature of the rollout relative to the ground truth. Bottom: RDF L1 error.
    Dots mark the checkpoint that was tested with 10^4-step rollouts (value = test temperature / truth).
    """
    import matplotlib.pyplot as plt

    from equitraj import plots

    runs = sorted(glob.glob("results/argon/B_S3a_seed*/log.jsonl"))
    if not runs or not os.path.exists("results/argon/ground_truth/metrics.json"):
        return None
    t_gt = _read_json("results/argon/ground_truth/metrics.json")["T_mean_K"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
    for k, log in enumerate(runs):
        name = Path(log).parts[2]
        with open(log, encoding="utf-8") as fh:
            parsed = [json.loads(line) for line in fh if '"val_T_end"' in line]
        rows = [r for r in parsed if r.get("val_T_end") is not None and r.get("val_rdf_l1") is not None]
        step = [r["step"] for r in rows]
        c = plots.SERIES[k % len(plots.SERIES)]
        label = name.replace("B_S3a_", "")
        ax1.plot(step, [r["val_T_end"] / t_gt for r in rows], color=c, linewidth=2, label=label)
        ax2.plot(step, [r["val_rdf_l1"] for r in rows], color=c, linewidth=2)
        ev = os.path.join(os.path.dirname(log), "eval", "metrics.json")
        if os.path.exists(ev):
            m = _read_json(ev)
            t_test = m.get("temperature", {}).get("T_model_second_half_K")
            if t_test is not None:
                ax1.scatter([m["train_step"]], [t_test / t_gt], color=c, s=60, zorder=3, edgecolor="black")
            else:  # every test rollout crashed
                ax1.scatter([m["train_step"]], [0.0], color=c, s=60, zorder=3, marker="x")
    ax1.axhline(1.0, color=plots.GRAY, linewidth=1, linestyle="--")
    ax1.axhspan(0.95, 1.05, color=plots.GRAY, alpha=0.15, linewidth=0)
    ax1.set_ylim(0, 2.8)
    plots._style(ax1, "", "T / T_ground-truth", "Validation rollout temperature by training step")
    ax1.legend(frameon=False, ncol=3, fontsize=9)
    ax2.set_yscale("log")
    plots._style(ax2, "training step", "RDF L1 error", "RDF error of validation rollouts")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main():
    lines = ["# Results", "",
             "Generated by `python scripts/make_results.py` from the `metrics.json` files under `results/`.",
             "Rollouts replace the integrator completely (no OpenMM). \"Model only\" means no thermostat and no energy rescaling.",
             "", "## Ground truth (OpenMM, NVE)", "",
             "| System | Trajectories | T (K) | Energy drift (kJ/mol/ns/dof) | First RDF peak | D (cm²/s) | OpenMM speed (ns/day) |",
             "|---|---|---|---|---|---|---|"]
    lines += ground_truth_rows()
    lines += ["", "## Model rollouts", "",
              "Errors are relative unless marked Å; the temperature column shows the signed error. "
              "`H drift` reports total-energy drift (`U_ref + K`, in `kJ/mol/ns/dof`) evaluated under the reference "
              "Lennard-Jones force field next to the OpenMM NVE ground-truth drift, and `ΔU/atom`, `ΔK/atom` (in `kJ/mol`) "
              "report second-half potential and kinetic energy offsets per atom relative to ground truth (`not measured` "
              "for earlier runs evaluated before the `energy` block was recorded; short CPU evaluations are labeled `(CPU)` "
              "and are distinct from the full `15 × 10000` GPU protocol). "
              "v0 criteria (argon): no crash (`T_ref / 3 <= T <= 3 T_ref`, no NaN, no overlap); |ΔT|/T < 5%; "
              "RDF peak position < 0.05 Å; peak height < 10%; VACF first zero < 10%; D < 20%.", "",
              "| Run | Head | Target | Seed | Stride (fs) | Rollouts × steps | Not crashed | T model / truth (K) | H drift (kJ/mol/ns/dof) | ΔU/atom (kJ/mol) | ΔK/atom (kJ/mol) | RDF peak pos (Å) | RDF peak height | VACF zero | D | v0 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    rows = eval_rows()
    lines += rows if rows else ["| (no evaluated models yet) |" + " |" * 15]
    lines += [
        "",
        "## Reading the `kinematics` and `energy` diagnostics (`metrics.json`)",
        "",
        "- **Position-velocity consistency (`kinematics` block, report-only):** `scripts/evaluate.py` computes three checks "
        "identically on surviving model rollouts and on ground-truth trajectories subsampled at the same 50 fs stride:",
        "  1. `trapezoid_residual` = `RMS(Δx - 0.5 * (v_t + v_{t+1}) * Δt) / RMS(Δx)`. At a 50 fs stride in liquid argon, "
        "atomic accelerations curve the trajectory between saved frames, so ground truth has a nonzero baseline (~0.021); "
        "a substantially larger model value indicates decoupled `Δx` and `v_new` heads.",
        "  2. `displacement_temperature_K` (from chord velocities `Δx / Δt`) vs `velocity_temperature_K` (from predicted `v`) and "
        "`temperature_rel_gap = (T_vel - T_disp) / T_disp`. At 50 fs, chord averaging slightly filters high-frequency motion in "
        "ground truth (`temperature_rel_gap` ~ +1.1%); a large excess gap shows velocity statistics and spatial step sizes "
        "disagree.",
        "  3. `green_kubo_D_cm2_s` (trapezoidal time integral `(1/3) ∫ <v(0)·v(t)> dt` over the VACF window) vs `einstein_D_cm2_s` "
        "(linear MSD slope `slope / 6`), testing whether velocity-space memory and position-space transport give consistent "
        "diffusivity.",
        "- **Reference-Hamiltonian energy (`energy` block):** Direct 50 fs predictors do not integrate a conservative Hamiltonian; "
        "`H_drift_kJ_mol_ns_dof_mean` measures linear drift of `H = U_ref(x) + K(v)` per ns per degree of freedom on surviving "
        "rollouts, while `U_model_minus_gt_per_atom` and `K_model_minus_gt_per_atom` show whether steady-state energy errors "
        "arise from steric configurational offset (`ΔU`) or thermal offset (`ΔK`). Note: seed 3 and seed 4 `best.pt` weights "
        "are not tracked in git, so their `15 × 10000` energy drift rows remain `not measured` pending GPU re-evaluation.",
    ]
    fig = stability_figure()
    if fig:
        lines += ["", "## Stability by training step", "",
                  "![Validation rollouts by training step](results/argon/stability_vs_step.png)", "",
                  "Each line is one training seed (flow head, residual targets, default `best.pt` selection). Rollouts are 1000 steps on validation "
                  "trajectories, evaluated every 2000 training steps; the grey band is ±5% of the ground-truth temperature. "
                  "Dots are the checkpoints that were tested with 10⁴-step rollouts (dot height = test temperature / truth); an x marks a checkpoint whose 10⁴-step test rollouts all crashed."]
    with open("RESULTS.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"RESULTS.md: {len(rows)} evaluated runs")


if __name__ == "__main__":
    main()
