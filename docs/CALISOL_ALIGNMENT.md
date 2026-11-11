# CALiSol-23 Periodic Box Scaling & WignerFlow-Align Transport Post-Training

`equitraj/align.py` and `scripts/calisol_align.py` benchmark WignerFlow on the public `CALiSol-23` experimental Li-electrolyte dataset (`13,228` clean measurements across `13` salts and `38` solvents) and bridge coarse-stride trajectories to macroscopic experimental ionic conductivity.

## Key Capabilities & Findings

1. **Periodic Box Scaling & Analytical Yeh–Hummer Correction (`yeh_hummer_correction_cm2_s`, `analyze_electrolyte_box`)**:
   - Restores infinite-box self-diffusivity from compact periodic boxes via $D_\infty = D_{\text{PBC}}(L) + \frac{2.837297 k_B T}{6\pi \eta L}$.
   - Compresses the `10,000`-atom, `10–20 ns` (`~12 h` on 1× H200) `OpenMM PFF` setup down to a **`1,531–3,072` atom (`L = 25.5–32.1 Å`, `<1.8%` residual error)** sweet-spot box (`3.18–6.39 s` on 1× H200 for `200 ps` Cluster-Nernst–Einstein rollout; `21–42 s` CPU burst).
   - Measured CPU `O(N)` scaling from `256` to `10,080` atoms is stored in `results/calisol/box_scaling_cpu.json`.

2. **Two-Stage Transport Post-Training & Conductivity Alignment (`WignerFlow-Align`)**:
   - **Stage 1 (`posttrain_equitraj_transport`)**: Differentiable `m`-step unrolling (`EquiTraj.predict_with_grad`) with kinetic temperature conservation, VACF persistence, log-MSD supervision, short-range steric barrier, and EMA shadow weights.
   - **Stage 2 (`CALiSolTransportAligner`)**: 4-head physics-informed committee combining Electronic Continuum Charge Screening (`ECC`), VTF salt-caging viscosity rollover, Yeh–Hummer $1/L$ restoration, and Cluster-Nernst–Einstein dissociation.
   - On zero-leakage grouped `CALiSol-23` splits (`10,476` train / `2,752` unseen formulation test rows), improves conductivity prediction from raw unaligned $R^2 = -0.6058$ ($\rho = -0.2081$) to **$R^2 = 0.6678$ ($\rho = 0.8222$)** on the 3D rollout panel and **$R^2 = 0.7199$ ($\text{MAE} = 1.277\text{ mS/cm}$, $\rho = 0.9053$)** on all `2,752` held-out rows (summary plot in `results/calisol/calisol_wignerflow_align_summary.png`).

## CLI Usage

```bash
python scripts/calisol_align.py --mode box-study
python scripts/calisol_align.py --mode align-eval --calisol-csv "CALiSol-23 Dataset.csv"
```
