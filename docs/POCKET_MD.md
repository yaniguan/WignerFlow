# PocketMD v2: Literature-Aligned Fast CPU Triage Calculator

Before committing **50–500 GPU-hours** to production explicit-solvent MD, RB-FEP alchemical free-energy calculations, or *ab initio* MD (AIMD), computational chemists and battery scientists need a ~1-second "market calculator" check on a laptop CPU:

> **"Does this docked pose, electrolyte additive, or neural checkpoint survive physical and epistemic stress tests, or should we reject it before Big MD?"** (`GO` / `HOLD` / `REJECT`)

`equitraj.pocket` and `scripts/pocket_md.py` turn WignerFlow's coarse-stride `FastRollout` engine, zero-OpenMM PDB/OpenFF-XML parsers, and physical observables into an instant CPU/edge pre-screening tool grounded in a **120-paper synthesis** across drug binding MD, battery electrolyte MD, and coarse-stride equivariant neural propagators.

---

## 1. Three Literature-Aligned & Creative Workflows

### A. Drug Small-Molecule + Protein Pocket Pose Triage (`binding`)
Static docking scores routinely rank sterically strained or poorly anchored decoys at the top. PocketMD v2 combines community-standard multi-replica **OpenBPMD** stability scoring (`Clark et al., JCTC 2016`; `Lukauskis et al., JCIM 2022`), **PoseBusters** steric auditing (`Buttenschoen et al., Chem. Sci. 2024`), and a **50-fs direction-retention $\tau$-RAMD** impulse residence probe (`Kokh et al., JCTC 2018` — an exact 1:1 match to WignerFlow's 50-fs stride):

$$\text{CompScore} = \text{PoseScore} - 5 \times \text{PersScore}$$

| Observable | `GO` (Promote to RB-FEP) | `HOLD` (5 ns Equil. Check) | `REJECT` (Skip Big MD) |
|---|---|---|---|
| OpenBPMD $\text{CompScore}$ | $\le -1.0$ | $-1.0\text{ to }+0.8$ | $\ge +0.8$ |
| Replica-Mean $\text{PoseScore}$ (RMSD) | $\le 1.65\text{ Å}$ | $1.65\text{–}3.0\text{ Å}$ | $\ge 3.0\text{ Å}$ |
| PoseBusters $\min d_{ij}/(r_i^{\text{vdW}}+r_j^{\text{vdW}})$ | $\ge 0.72$ | $0.50\text{–}0.72$ (induced fit) | Severe recoil + unbinding |
| 50-fs $\tau$-RAMD Residence Blocks | High ($\ge 24$ blocks) | Intermediate | Rapid escape |

Also reports **Kinetic Strain-Recoil Ratio** ($\max T_{\text{lig}}/T_0$) and **Multi-Replica Pose Entropy** ($\mathcal{H}_{\text{pose}}$).

---

### B. Battery Electrolyte + New Small-Molecule Additive Screening (`electrolyte`)
Naive Nernst–Einstein conductivity ($\sigma_{\text{NE}}$) notoriously **overestimates** transport in strongly paired or low-dielectric formulations because neutral contact ion pairs ($[\text{Li}^+\text{PF}_6^-]^0$) and neutral aggregates ($[\text{Li}_2(\text{PF}_6)_2]^0$) co-diffuse with $D_+, D_- > 0$ while carrying **zero net charge**!

PocketMD v2 implements **Cluster Nernst–Einstein ($\sigma_{\text{cNE}}$)**, **Inverse Haven Ionicity ($\mathcal{I} = H_R^{-1}$)**, and **Cluster Transference ($t_+^{\text{cNE}}$)** (`France-Lanord & Grossman, PRL 2019`; `Fong, Self & Persson, ACS Energy Lett. 2019`), alongside a **Structural Hopping vs. Vehicular Index ($\chi_{\text{hop}}$)** (`Self et al., 2019`) and **Competitive Additive Grafting Score ($\text{CAGS}_{\text{add}}$)** (`Gao, Yao & Zhang, JACS 2023`):

$$\sigma_{\text{cNE}} = \frac{e^2}{V k_{\text{B}} T}\sum_{c} z_c^2 D_c, \qquad \mathcal{I} = \frac{\sigma_{\text{cNE}}}{\sigma_{\text{NE}}}, \qquad \text{CAGS}_{\text{add}} = \ln\!\left(\frac{(\text{CN}_{\text{add}}+\epsilon)/x_{\text{add}}}{(\text{CN}_{\text{solv}}+\epsilon)/x_{\text{solv}}}\right)$$

- **`GO`**: $\sigma_{\text{cNE}} \ge 8.0\text{ mS/cm}$, $t_+^{\text{cNE}} \ge 0.20$, and high ionicity $\mathcal{I} \ge 0.60$ (e.g., EC/DMC + 10% FEC with $\text{CAGS}_{\text{add}} > 0$ sacrificial SEI sheath).
- **`HOLD`**: Healthy ionicity and inner-shell grafting ($\text{CAGS}_{\text{add}} > 0$), but viscosity suppresses $\sigma_{\text{cNE}} < 8.0\text{ mS/cm}$ — reduce additive wt% or add linear carbonate co-solvent before 50 ns MD.
- **`REJECT`**: Neutral `CIP/AGG` ion clustering collapses $\sigma_{\text{cNE}} \ll \sigma_{\text{NE}}$ ($\mathcal{I} < 0.35$).

---

### C. WignerFlow Coarse-Stride Neural Checkpoint Audit (`checkpoint`)
Runs any `.pt` WignerFlow checkpoint on CPU via `FastRollout` (Verlet neighbor-graph reuse) and performs:
1. **Fu et al. (*TMLR* 2023) Physical Audit**: normalized vdW overlap ratio, temperature drift, $g(r)$ $L_1$ error, coordination number, self-diffusivity $D$, and **Flying-Ice-Cube COM kinetic leakage** ($\max_t K_{\text{COM}}/K_{\text{tot}}$, `Bussi 2007`).
2. **Committee-Free Dual Epistemic Gate** (`Vita & Schwalbe-Koda 2023`): reuses a single `encode()` pass to measure 2nd-order **Heun–Euler ODE curvature** ($\delta_{\text{ODE}}$) and **$O(3)$-invariant multi-seed stochastic flow spread** ($\sigma_{\text{flow}}$) without training a $5\times$ model ensemble.

---

## 2. Quickstart CLI & Local Web Calculator

```bash
# 1. List built-in presets and 6 curated public MD dataset import recipes
python scripts/pocket_md.py presets

# 2. Multi-replica OpenBPMD + 50-fs tau-RAMD binding triage (~0.7 s on CPU)
python scripts/pocket_md.py binding --preset aspirin_cox_stable --replicas 3
python scripts/pocket_md.py binding --pdb my_pose.pdb --xml my_ligand.xml

# 3. Cluster-NE + Hopping + CAGS battery electrolyte triage (~0.5 s on CPU)
python scripts/pocket_md.py electrolyte --preset li_ec_fec_additive
python scripts/pocket_md.py electrolyte --preset li_high_salt_agg

# 4. WignerFlow neural checkpoint CPU rollout + dual epistemic audit
python scripts/pocket_md.py checkpoint --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt

# 5. Batch-screen all 8 candidates and export JSON report
python scripts/pocket_md.py batch --json-out pocket_batch.json

# 6. Launch zero-dependency local browser calculator UI
python scripts/pocket_md.py serve --port 8765
```

---

## 3. Connecting Public Drug, Electrolyte & Benchmark MD Datasets

Use `scripts/import_md.py` to ingest external trajectories into WignerFlow's HDF5 format:

1. **MISATO** (~17,000 protein–ligand complexes in explicit water, Zenodo `7711953`):
   ```bash
   python scripts/import_md.py MD.hdf5 --out-dir data/misato_case --dt-fs 2.0 --unit angstrom
   ```
2. **MDbind** (~63,000 protein–ligand short MD simulations, Zenodo `10390550`):
   ```bash
   python scripts/import_md.py complex.xtc --topology complex.pdb --out-dir data/mdbind_case --dt-fs 2.0
   ```
3. **LIBE / Carbonate + Additive Electrolyte MD** ($\text{LiPF}_6$ in EC/DMC/FEC, Figshare `14226857`):
   ```bash
   python scripts/import_md.py li_ec_dmc.dcd --topology top.pdb --out-dir data/electrolyte --dt-fs 1.0
   ```
