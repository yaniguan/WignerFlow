"""Run the 7-system CCSD(T)/MD17/MD22 Pure-MD Coarse-Stride Benchmark & plot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equitraj.mdbench import (
    DEFAULT_BENCHMARK_PANEL,
    EquivariantPairFlowModel,
    compute_hr_density,
    load_or_fetch_benchmark_system,
    run_md17_panel_benchmark,
    sample_maxwell_boltzmann_velocities,
)


def plot_md_panel_summary(
    report: dict,
    *,
    cache_dir: Path | None,
    fig_path: Path,
) -> None:
    rows = report["molecules"]
    names = [f"{r['label']}\n({r['n_atoms']}a, {r['theory']})" for r in rows]
    x = np.arange(len(names))

    fig, axes = plt.subplots(2, 2, figsize=(13.6, 9.6), dpi=160)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel A: Stability Horizon (Fu et al. ICLR/TMLR 2023 criterion)
    w = 0.26
    ax1.bar(
        x - w,
        [r["naive_verlet_10fs_t_stab_ps"] for r in rows],
        width=w,
        color="#d95f02",
        label="Naive Verlet (10 fs)",
    )
    ax1.bar(
        x,
        [r["wf_10fs_t_stab_ps"] for r in rows],
        width=w,
        color="#1b9e77",
        label="Coarse step (10 fs)",
    )
    ax1.bar(
        x + w,
        [r["sim_horizon_ps"] if r["wf_20fs_stable"] else 0.0 for r in rows],
        width=w,
        color="#386cb0",
        label="Coarse step (20 fs)",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=22, ha="right", fontsize=8.5)
    ax1.set_ylabel("Stable Rollout Horizon $T_{\\mathrm{stab}}$ (ps)")
    ax1.set_title("A. Coarse-Stride Stability Horizon (Fu et al. Criterion)")
    ax1.legend(frameon=False, fontsize=8.5)
    ax1.grid(axis="y", alpha=0.25)

    # Panel B: Fu et al. h(r) MAE vs Held-Out CCSD(T) / DFT-MD Ensembles
    ax2.bar(
        x - w,
        [r["verlet_05fs_hr_mae"] for r in rows],
        width=w,
        color="#7570b3",
        label="Fine Verlet (0.5 fs)",
    )
    ax2.bar(
        x,
        [r["wf_10fs_hr_mae"] for r in rows],
        width=w,
        color="#1b9e77",
        label="Coarse step (10 fs)",
    )
    ax2.bar(
        x + w,
        [r["wf_20fs_hr_mae"] for r in rows],
        width=w,
        color="#386cb0",
        label="Coarse step (20 fs)",
    )
    ax2.axhline(
        0.10,
        color="#666666",
        linestyle="--",
        linewidth=1.0,
        label="High-Fidelity Band (0.10)",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=22, ha="right", fontsize=8.5)
    ax2.set_ylabel("All-Pair Distance $\\mathrm{MAE}_{h(r)}$")
    ax2.set_title("B. Ensemble Fidelity vs Held-Out Quantum Trajectories")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(axis="y", alpha=0.25)

    # Panel C: MD22 AT-AT (60-atom DNA base pair) & Aspirin CCSD h(r) match
    atat = load_or_fetch_benchmark_system("md22_AT_AT", cache_dir=cache_dir)
    model_atat = EquivariantPairFlowModel.fit_from_trajectory(
        atat["z"], atat["R_train"], atat["F_train"], target_temp_k=500.0
    )
    pos0 = atat["R_test"][0].copy()
    vel0 = sample_maxwell_boltzmann_velocities(pos0, model_atat.masses, temp_k=500.0, seed=48)
    wf_atat_r, _, _ = model_atat.rollout_wigner_coarse(
        pos0, vel0, n_steps=100, dt_fs=10.0, seed=48
    )
    r_grid, h_qm = compute_hr_density(atat["R_test"], r_max=13.5)
    _, h_wf = compute_hr_density(wf_atat_r, r_max=13.5)
    ax3.plot(
        r_grid,
        h_qm,
        color="#222222",
        linewidth=2.0,
        label="MD22 AT-AT Held-Out PBE+MBD (60 Atoms)",
    )
    ax3.plot(
        r_grid,
        h_wf,
        color="#1b9e77",
        linestyle="--",
        linewidth=2.0,
        label="Coarse step 10 fs rollout",
    )
    ax3.fill_between(r_grid, h_qm, h_wf, color="#1b9e77", alpha=0.18)
    ax3.set_xlabel("Interatomic Distance $r$ ($\\mathrm{\\AA}$)")
    ax3.set_ylabel("Normalized Pair Density $h(r)$ ($\\mathrm{\\AA}^{-1}$)")
    ax3.set_title("C. Supramolecular MD22 AT-AT DNA Base Pair $h(r)$ Match")
    ax3.legend(frameon=False, fontsize=8.5)
    ax3.grid(alpha=0.25)

    # Panel D: CPU Wall-Clock Speedup vs Fine 0.5-fs Integration
    ax4.bar(
        x - 0.17,
        [r["wf_10fs_speedup_vs_05fs"] for r in rows],
        width=0.34,
        color="#1b9e77",
        label="Coarse step 10 fs speedup",
    )
    ax4.bar(
        x + 0.17,
        [r["wf_20fs_speedup_vs_05fs"] for r in rows],
        width=0.34,
        color="#386cb0",
        label="Coarse step 20 fs speedup",
    )
    ax4.set_xticks(x)
    ax4.set_xticklabels(names, rotation=22, ha="right", fontsize=8.5)
    ax4.set_ylabel("Wall-Clock Speedup ($\\times$ vs 0.5 fs Verlet)")
    ax4.set_title("D. Single-CPU Rollout Acceleration (9 to 60 Atoms)")
    ax4.legend(frameon=False, fontsize=8.5)
    ax4.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Coarse-step baseline on public MD17 and MD22 data (pair model, not the WignerFlow network)",
        fontsize=12.8,
        fontweight="bold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--horizon-ps", type=float, default=1.0)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results" / "pure_md_md17" / "md17_md22_benchmark.json",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=REPO_ROOT / "results" / "pure_md_md17" / "md17_coarse_step_benchmark.png",
    )
    args = parser.parse_args()

    report = run_md17_panel_benchmark(
        DEFAULT_BENCHMARK_PANEL,
        cache_dir=args.cache_dir,
        sim_horizon_ps=args.horizon_ps,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_md_panel_summary(report, cache_dir=args.cache_dir, fig_path=args.out_fig)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
