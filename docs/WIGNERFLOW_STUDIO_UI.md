# WignerFlow Studio · Interactive Web Workbench & 10-Gate Physical Compliance Lab

`WignerFlow Studio` (`equitraj/studio.py` + `equitraj/compliance.py`) provides a **zero-external-dependency browser workbench** connected directly to a live Python/PyTorch backend so researchers can interactively test every capability of WignerFlow and verify all fundamental algebraic & statistical-mechanical compliance laws in real time.

---

## 1. Quick Launch

```bash
# Start interactive Web Workbench on http://127.0.0.1:8765
python scripts/wignerflow_studio.py --port 8765

# Headless CLI 10-Gate Compliance Certificate (float64 exact algebraic mode)
python scripts/wignerflow_studio.py --audit-only --target aspirin_hetero_o3 --precision float64 --json-out audit.json
```

---

## 2. Five Interactive Workspaces

| Workspace | Core Capabilities | Live Outputs |
| :--- | :--- | :--- |
| **1. Equivariant & Physics Compliance (`G1–G10`)** | Audits `EquiTraj` checkpoints (`argon_flow_v1`, `argon_flow_v0`) and hetero-atomic `O(3)`/`SO(3)` models under custom Euler angles $(\alpha, \beta, \gamma)$ + **`F1–F10` Chaos Fault-Injection Drawer**. | 3D Twin Vector Overlay (`f(X)` vs $R^{-1}f(RX)$), 5-Axis $-\log_{10}\varepsilon$ Symmetry Radar, 10-Gate Traffic-Light Scorecard. |
| **2. Coarse-Stride & Custom `.xyz` Rollout** | Runs CPU `BAOAB` trajectories on built-in presets (`aspirin`, `ala2`, `ethanol`, `li_ec_pf6`) or pasted `.xyz` molecules with automatic covalent bond/angle inference. | Animated 3D Ball-and-Stick Viewport, $T(t)$, Bond RMS Drift, `PoseBusters` $\rho_{\text{vdW}}$, 1-click `.xyz` download. |
| **3. Drug Binding Pocket Triage** | Multi-replica `OpenBPMD` (`CompScore`), `PoseBusters` sterics, and 50-fs direction-retention $\tau$-RAMD unbinding probe (`COX-1 + Aspirin` anchor vs induced-fit vs decoy). | 3D Pocket+Ligand Trajectory, `RMSD(t)`, H-Bond count, `GO / HOLD / REJECT` RB-FEP gate. |
| **4. Battery Electrolyte & `1,000k` Solvation** | Periodic $\text{Li}^+$ box screening (`Cluster-NE` $\sigma_{\text{cNE}}$ vs naive $\sigma_{\text{NE}}$, Haven ionicity, `CAGS`) + Top 15 `1,000k` funnel `SSIP/CIP/AGG-I/AGG-net` explorer. | 3D Periodic Unit Cell, $g_{\text{Li-O}}(r)$, Stacked Solvation Speciation Bars, Readiness Score. |
| **5. Unified Benchmark Scorecard** | Cross-domain evaluation tables (`MD17-CCSD(T)` + 60-atom `MD22 AT-AT`, `CALiSol-23` conductivity alignment, `1,000k` solvation, public MD dataset catalog). | Interactive comparison tables + `.wignerstudio.json` session receipt export. |

---

## 3. The 10 Physical Compliance Gates (`G1–G10`) & Falsifiable Faults (`F1–F10`)

1. **`G1` — Proper `SO(3)` Roto-Equivariance & `±ŷ` Pole Conditioning** (`F1`, `F2`): Verifies $f(RX, RV, RL) = Rf(X, V, L)$ under arbitrary Euler/Haar rotations AND edges aligned within $10^{-5\circ}$ of the $\pm\hat{y}$ spherical-harmonic pole.
2. **`G2` — Full `O(3)` Mirror Reflection & Chiral Enantiomer Parity** (`F3` & `parity=False`): Verifies polar vector sign flip under $\det Q = -1$ ($Q = \operatorname{diag}(1,1,-1)$ and $-I_3$). Injecting pseudovector $\mathbf{r}_{\text{rel}} \times \mathbf{v}$ (`F3`) keeps `G1` `PASS` while surgically tripping `G2` `FAIL`.
3. **`G3` — `T(3)` Shift Invariance & Mass-Weighted Zero-COM Subspace** (`F4`): Verifies invariance under large coordinate shifts and exact mass-weighted momentum conservation ($\sum_i m_i \Delta\mathbf{v}_i = \mathbf{0}$) vs unweighted point-cloud centering leak (`F4`).
4. **`G4` — `S_N` Atom-Permutation Equivariance**: Verifies exact commutation under random atom index bijections $\pi \in S_N$ across positions, velocities, charges, covalent bond hops, and flow noise.
5. **`G5` — Periodic Lattice Covariant Wrapping & $C^2$ Cutoff Continuity** (`F5`): Verifies Nyquist cell ratio $L_{\min}/r_{\text{cut}} \ge 2.05$, $\mathbb{Z}^3$ lattice-image shift invariance, and smooth `envelope()` vanishing at $r_{\text{cut}}$.
6. **`G6` — Non-Conservative Curl Loop Defect & Secular Drift** (`F6`): Measures closed-parallelogram force circulation $|\oint \mathbf{F}\cdot d\mathbf{x}| / k_B T$ (`Bigi, Langer & Ceriotti, ICML 2025`) and per-step secular kinetic drift.
7. **`G7` — Canonical `CSVR` Gamma KS & Anti-Flying-Ice-Cube Equipartition** (`F7`): Audits $K_{\text{COM}} / K_{\text{tot}} < 10^{-3}$ (`Harvey, Tan & Cheatham 1998`), canonical Gamma kinetic distribution, and multi-element temperature equipartition.
8. **`G8` — `PoseBusters` Steric Core-Exclusion & Bond Topology** (`F8`): Audits non-bonded van der Waals overlap ratio $\rho_{\text{vdW}}^{\min} \ge 0.72$, `RepulsionPrior` guard rail, and covalent bond deviation.
9. **`G9` — Ensemble Structural & Thermal Fidelity** (`F9`): Audits 100% rollout survival, thermal accuracy $|T - T_{\text{ref}}|/T_{\text{ref}}$, and liquid pair-correlation `RDF L1` error.
10. **`G10` — Dual Zero-Committee Epistemic UQ & Flow Straightness** (`F10`): Audits 2nd-order Heun vs 1st-order Euler discrepancy $\delta_{\text{ODE}}$, $O(3)$-invariant multi-seed flow spread $\sigma_{\text{flow}}$, and flow ODE trajectory straightness.
