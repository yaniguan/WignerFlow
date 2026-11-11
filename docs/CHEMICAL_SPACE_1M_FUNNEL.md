# 1,000,000-Candidate (`1,000k`) Unknown Electrolyte Chemical Space Funnel

`equitraj/funnel.py` and `scripts/electrolyte_funnel_1m.py` construct a bijectively indexed `1,000,000`-candidate battery electrolyte chemical space (`10 salts × 20 backbones × 100 novel functional small-molecule additives × 10 ternary blend profiles × 5 molarities`) and screen it in **`~45 s` on CPU** via a 3-stage `WignerFlow-Align` funnel:

1. **Tier 1 (`1,000,000 -> 180`, `~1.03 s` CPU)**: Vectorized Kronecker-factored physicochemical tensor evaluation through `CALiSolTransportAligner` at `298.15 K` (`+25 °C`) and `253.15 K` (`-20 °C`) with Pareto diversity across solvent backbones and additive families.
2. **Tier 2 (`180 -> 20`, `~25.1 s` CPU)**: 3D periodic box packing + `Electrolyte-Busters` vdW non-overlap audit + 6-step `FastRollout` burst stability check.
3. **Tier 3 (`20 -> Top 15`, `~19.1 s` CPU)**: Sweet-spot periodic `WignerFlow-Align` trajectory rollout extracting `Cluster-Nernst–Einstein` ionicity $I_{\text{cNE}}$, solvation cage hopping $\chi_{\text{hop}}$, $t_+$, and Yeh–Hummer-restored conductivities (saved in `results/funnel_1m/top15_novel_electrolytes.json`).
