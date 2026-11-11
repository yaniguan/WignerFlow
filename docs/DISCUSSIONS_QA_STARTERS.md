# WignerFlow Scientific Q&A Starters (GitHub Discussions)

Reference Q&A topics for the [WignerFlow `Q&A` Discussions board](https://github.com/yaniguan/WignerFlow/discussions/categories/q-a) (8 topics — matching the **Bronze Galaxy Brain** milestone when asked by a community member/collaborator and answered + marked as Accepted Answer):

1. **Why can WignerFlow use a `1,500–3,000` atom periodic box instead of the `10,000`-atom box required by `OpenMM PFF`?**
   - *Core answer*: Uncorrected periodic boundary conditions underestimate self-diffusion by $25\%\text{–}38\%$ due to Oseen hydrodynamic image drag, and collective Einstein–Helfand (`EH`) tracks a single global 3D dipole vector without $1/\sqrt{N}$ particle averaging. Applying the analytical Yeh–Hummer restoration $\Delta D_{\text{YH}} = \frac{2.837297 k_B T}{6\pi \eta L}$ combined with particle-averaged Cluster-Nernst–Einstein (`cNE`) reduces finite-size error below $1.8\%$ at $L \approx 25.5\text{–}32.1\text{ Å}$ (`10–20` ion pairs).

2. **Why does raw Nernst–Einstein conductivity fail on `CALiSol-23` above `1.2 M` salt concentration, and how does `WignerFlow-Align` fix it?**
   - *Core answer*: Standard Nernst–Einstein treats neutral contact ion pairs (`CIP`, net charge $z_c = 0$) as independent charge carriers and misses both Electronic Continuum Screening (`ECC`, $q_{\text{eff}} \approx 1/\sqrt{\varepsilon_\infty}$) and exponential VTF salt-caging viscosity growth. `WignerFlow-Align` combines multi-step unrolled velocity/MSD post-training with a 4-head `ECC + VTF + Yeh–Hummer + cNE` physics committee, recovering the `0.7–1.0 M` `EC:DMC` conductivity peak (`11.88 mS/cm`).

3. **How does WignerFlow jump `50 fs` per step without violating the `1–2 fs` bond-vibration Nyquist limit?**
   - *Core answer*: Classical symplectic integrators (`Velocity Verlet`) integrate instantaneous forces $-\nabla U(\mathbf{x})$ and blow up whenever $\Delta t > 2/\omega_{\max}$. WignerFlow learns the conditional transition operator $p(\Delta \mathbf{x}_{t\to t+\Delta\tau}, \mathbf{v}_{t+\Delta\tau} \mid \mathbf{x}_{t-k+1:t}, \mathbf{v}_{t-k+1:t})$ via `SO(2)` edge-frame flow matching, analytically averaging over fast bond oscillations while advancing slow diffusive coordinates.

4. **Why is `SO(2)` edge-aligned message passing faster than standard `SO(3)` Clebsch–Gordan tensor products?**
   - *Core answer*: Rotating node irreps into each directed edge's local frame ($\hat{\mathbf{r}}_{ij} \parallel \hat{\mathbf{y}}$) makes irreps block-diagonal in azimuthal order $m$, reducing tensor-product complexity from $O(L_{\max}^6)$ to $O(L_{\max}^3)$ while preserving exact $O(3)$ equivariance (`< 1e-10` error in `float64`).

5. **How do the dual zero-ground-truth epistemic gates detect neural MD rollout instability before atomic blowup?**
   - *Core answer*: Without running reference MD, comparing a 2-step Heun ODE step against a 1-step Euler step measures local vector-field curvature ($\Delta_{\text{ODE}}$), while sampling $K$ stochastic flow noise draws from identical history frames measures epistemic trajectory spread ($\sigma_{\text{flow}}$), flagging out-of-distribution states steps before steric collapse.

6. **Why must `50-fs tau-RAMD` pulling force scale with ligand molecular weight on pharmaceutical targets like `CDK2` and `MCL1`?**
   - *Core answer*: A fixed force $F_0$ produces center-of-mass acceleration $a = F_0 / M_{\text{lig}}$, under-accelerating `400–500 Da` drug-like ligands relative to fragments. Mass-calibrated Kokh scaling ($F_0 \propto M_{\text{lig}}$) normalizes egress acceleration and yields `+54%` longer residence times for native X-ray poses over perturbed decoys.

7. **Why does `S1` single-step teacher forcing under-predict molten salt and electrolyte diffusivity, and how does unrolled post-training fix it?**
   - *Core answer*: Under single-step MSE/flow supervision, conditional mean regression slightly contracts velocity magnitudes ($\|\hat{\mathbf{v}}\| < \|\mathbf{v}\|$), compounding exponentially over autoregressive steps and shrinking the Green–Kubo VACF integral. Unrolling `EquiTraj.predict_with_grad` over $m$ steps with explicit $\mathcal{L}_T$, $\mathcal{L}_{\text{VACF}}$, and $\mathcal{L}_{\text{MSD}}$ eliminates velocity damping.

8. **How does the 3-stage `1,000k` (`1,000,000`-candidate) funnel screen novel electrolyte additives in `45 s` on CPU?**
   - *Core answer*: Stage 1 evaluates Kronecker-factored physicochemical tensors across all `1,000,000` blends at `298 K` and `253 K` in `1.03 s`; Stage 2 applies 3D periodic `Electrolyte-Busters` vdW clash checks + 6-step `FastRollout` bursts to the top `180` backbone-diverse candidates (`25.1 s`); Stage 3 runs sweet-spot periodic `WignerFlow-Align` trajectories on the top `20` regimes (`19.1 s`).
