"""Run fine-grained 3D solvation-shell analysis on Top 15 novel electrolytes."""

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

from equitraj.solvation import analyze_top15_funnel_candidates


def plot_solvation_summary(
    report: dict,
    rdf_curves: dict,
    fig_path: Path,
) -> None:
    cands = report["candidates"]
    base = report["commercial_baseline"]
    all_rows = [base] + cands
    labels = ["EC:DMC\n(Base)"] + [f"#{r['rank']} {r['backbone']}" for r in cands]
    x = np.arange(len(all_rows))

    fig, axes = plt.subplots(2, 2, figsize=(14.2, 9.8), dpi=160)
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Panel A: Stacked SSIP / CIP / AGG-I / AGG-net speciation vs Goldilocks CIP window
    ssip = np.array([r["ssip_frac"] * 100.0 for r in all_rows])
    cip = np.array([r["cip_frac"] * 100.0 for r in all_rows])
    agg1 = np.array([r["agg_dimer_frac"] * 100.0 for r in all_rows])
    aggn = np.array([r["agg_network_frac"] * 100.0 for r in all_rows])

    ax1.bar(x, ssip, color="#4daf4a", label="SSIP (Free Solvated Li+)")
    ax1.bar(x, cip, bottom=ssip, color="#377eb8", label="CIP (Single Ion Pair)")
    ax1.bar(x, agg1, bottom=ssip + cip, color="#ff7f00", label="AGG-I (Finite Dimer <=4)")
    ax1.bar(x, aggn, bottom=ssip + cip + agg1, color="#e41a1c", label="AGG-net (Percolating >=5)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=36, ha="right", fontsize=7.8)
    ax1.set_ylabel("Primary Li$^+$ Speciation (%)")
    ax1.set_ylim(0, 105)
    ax1.set_title("A. Bipartite Li$^+$ Solvation Speciation (Top 15 vs Commercial EC:DMC)")
    ax1.legend(frameon=False, fontsize=8.0, loc="lower right")
    ax1.grid(axis="y", alpha=0.22)

    # Panel B: Partial RDF g_{Li,beta}(r) & Running CN(r) for Champion #1
    champ_key = [k for k in rdf_curves if k.startswith("rank_1_")][0]
    crv = rdf_curves[champ_key]
    r_a = crv["r_a"]
    ax2.plot(r_a, crv["g_li_backbone"], color="#1b9e77", linewidth=2.0, label="$g_{\\mathrm{Li-Backbone}}(r)$")
    ax2.plot(r_a, crv["g_li_anion"], color="#d95f02", linewidth=2.0, label="$g_{\\mathrm{Li-Anion}}(r)$")
    ax2.plot(r_a, crv["g_li_additive"], color="#7570b3", linestyle="--", linewidth=1.8, label="$g_{\\mathrm{Li-Additive}}(r)$")
    ax2.axvline(cands[0]["rc_li_solv_a"], color="#444444", linestyle=":", linewidth=1.1, label=f"Auto $r_c={cands[0]['rc_li_solv_a']:.2f}$ $\\mathrm{{\\AA}}$")
    ax2_r = ax2.twinx()
    ax2_r.plot(r_a, crv["cn_li_backbone"] + crv["cn_li_additive"] + crv["cn_li_anion"], color="#222222", alpha=0.55, linewidth=1.5, label="Total $CN(r)$")
    ax2_r.set_ylabel("Running Coordination $CN(r)$", fontsize=9)
    ax2.set_xlabel("Li$^+$–Donor Distance $r$ ($\\mathrm{\\AA}$)")
    ax2.set_ylabel("Partial RDF $g(r)$")
    ax2.set_title(f"B. Radial Shell Structure of #1 ({cands[0]['backbone']} + {cands[0]['additive_name']})")
    ax2.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax2.grid(alpha=0.22)

    # Panel C: Trade-off Pareto front — Sub-zero -20C Conductivity vs CIP Anion-SEI Fraction
    ax3.axvspan(15.0, 45.0, color="#377eb8", alpha=0.12, label="Yu/Cui/Wang Goldilocks CIP Window (15-45%)")
    ax3.scatter(
        [base["cip_frac"] * 100.0],
        [base["sigma_253K_mS_cm"]],
        color="#d95f02",
        marker="s",
        s=95,
        zorder=5,
        label="Commercial 1M LiPF$_6$ EC:DMC",
    )
    sc = ax3.scatter(
        [r["cip_frac"] * 100.0 for r in cands],
        [r["sigma_253K_mS_cm"] for r in cands],
        c=[r["fast_charge_sei_readiness_100"] for r in cands],
        cmap="viridis",
        s=85,
        edgecolors="#222222",
        linewidths=0.6,
        zorder=6,
        label="Top 15 Novel Funnel Candidates",
    )
    for r in cands[:5]:
        ax3.annotate(
            f"#{r['rank']} {r['backbone']}",
            (r["cip_frac"] * 100.0 + 0.7, r["sigma_253K_mS_cm"] + 0.2),
            fontsize=7.5,
        )
    cbar = fig.colorbar(sc, ax=ax3, pad=0.02)
    cbar.set_label("Fast-Charge + SEI Readiness (/100)", fontsize=8.5)
    ax3.set_xlabel("Primary-Shell Contact Ion Pair Fraction $\\chi_{\\mathrm{CIP}}$ (%)")
    ax3.set_ylabel("Sub-Zero Conductivity $\\sigma_{253\\mathrm{K}}$ (mS/cm)")
    ax3.set_title("C. Sub-Zero Transport vs Anion-Derived SEI Speciation Trade-Off")
    ax3.legend(frameon=False, fontsize=8.0, loc="upper right")
    ax3.grid(alpha=0.22)

    # Panel D: Cage Exchange Residence Time tau_ex & Additive Enrichment alpha_enr
    ranks_x = np.arange(len(cands))
    ax4.bar(
        ranks_x - 0.18,
        [r["tau_exchange_ps"] for r in cands],
        width=0.36,
        color="#1b9e77",
        label="Cage Exchange $\\tau_{\\mathrm{ex}}$ (ps)",
    )
    ax4.axhline(
        base["tau_exchange_ps"],
        color="#d95f02",
        linestyle="--",
        linewidth=1.2,
        label=f"EC:DMC Baseline $\\tau_{{\\mathrm{{ex}}}}={base['tau_exchange_ps']:.1f}$ ps",
    )
    ax4_r = ax4.twinx()
    ax4_r.plot(
        ranks_x + 0.18,
        [r["additive_enrichment_alpha"] for r in cands],
        "o-",
        color="#7570b3",
        linewidth=1.5,
        markersize=5,
        label="Additive Enrichment $\\alpha_{\\mathrm{enr}}$",
    )
    ax4_r.set_ylabel("Additive Enrichment $\\alpha_{\\mathrm{enr}}$", color="#7570b3")
    ax4.set_xticks(ranks_x)
    ax4.set_xticklabels([f"#{r['rank']}" for r in cands], fontsize=8.5)
    ax4.set_xlabel("Funnel Candidate Rank")
    ax4.set_ylabel("Solvation Cage Residence Time $\\tau_{\\mathrm{ex}}$ (ps)")
    ax4.set_title("D. Solvation Shell Exchange Kinetics & Additive Competition")
    ax4.legend(frameon=False, fontsize=8.0, loc="upper left")
    ax4.grid(axis="y", alpha=0.22)

    fig.suptitle(
        "Fine-Grained 3D Solvation Shell, Ion Speciation & Exchange Kinetics of Top 15 Novel Electrolytes",
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
    parser.add_argument(
        "--top15-json",
        type=Path,
        default=REPO_ROOT / "results" / "funnel_1m" / "top15_novel_electrolytes.json",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=REPO_ROOT / "results" / "solvation_top15" / "top15_solvation_report.json",
    )
    parser.add_argument(
        "--out-fig",
        type=Path,
        default=REPO_ROOT / "results" / "solvation_top15" / "top15_solvation_speciation.png",
    )
    parser.add_argument("--ion-pairs", type=int, default=8)
    parser.add_argument("--prod-frames", type=int, default=28)
    args = parser.parse_args()

    funnel_data = json.loads(args.top15_json.read_text(encoding="utf-8"))
    report, rdf_curves = analyze_top15_funnel_candidates(
        funnel_data["top_candidates"],
        n_ion_pairs=args.ion_pairs,
        prod_frames=args.prod_frames,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    plot_solvation_summary(report, rdf_curves, args.out_fig)
    print(json.dumps(report["top15_summary"], indent=2))


if __name__ == "__main__":
    main()
