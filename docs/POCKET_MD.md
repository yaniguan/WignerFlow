# PocketMD: Fast CPU Pre-Screening Calculator

Before spending GPU hours on explicit-solvent MD, free-energy perturbation (RB-FEP), or *ab initio* MD (AIMD), you can run a quick check on a laptop CPU:

> **"Does this docked pose, electrolyte additive, or neural checkpoint pass basic physical and uncertainty checks?"** (`GO` / `HOLD` / `REJECT`)

`equitraj.pocket` and `scripts/pocket_md.py` combine WignerFlow's coarse-stride `FastRollout` engine, built-in PDB/OpenFF-XML parsers, and standard MD observables into a fast CPU pre-screening tool.

---

## 1. Three Screening Modes

### A. Protein–Ligand Binding Pose Triage (`binding`)
Static docking scores often rank strained or weakly anchored poses near the top. PocketMD runs short multi-replica agitation checks and combines **OpenBPMD** pose stability (`Clark et al., JCTC 2016`; `Lukauskis et al., JCIM 2022`), **PoseBusters** steric checks (`Buttenschoen et al., Chem. Sci. 2024`), and a **50-fs $\tau$-RAMD** residence probe (`Kokh et al., JCTC 2018`):

$$\text{CompScore} = \text{PoseScore} - 5 \times \text{PersScore}$$

| Observable | `GO` (Run RB-FEP) | `HOLD` (Short Equilibration Check) | `REJECT` (Skip Production MD) |
|---|---|---|---|
| OpenBPMD $\text{CompScore}$ | $\le -1.0$ | $-1.0\text{ to }+0.8$ | $\ge +0.8$ |
| Replica-Mean $\text{PoseScore}$ (RMSD) | $\le 1.65\text{ Å}$ | $1.65\text{–}3.0\text{ Å}$ | $\ge 3.0\text{ Å}$ |
| PoseBusters $\min d_{ij}/(r_i^{\text{vdW}}+r_j^{\text{vdW}})$ | $\ge 0.72$ | $0.50\text{–}0.72$ (induced fit) | Strong recoil + unbinding |
| 50-fs $\tau$-RAMD Residence Blocks | $\ge 24$ blocks | Intermediate | Rapid escape |

Also reports the ligand recoil temperature ratio ($\max T_{\text{lig}}/T_0$) and multi-replica pose spread ($\mathcal{H}_{\text{pose}}$).

---

### B. Battery Electrolyte and Additive Screening (`electrolyte`)
Standard Nernst–Einstein conductivity ($\sigma_{\text{NE}}$) overpredicts conductivity in concentrated or low-dielectric electrolytes because neutral contact ion pairs ($[\text{Li}^+\text{PF}_6^-]^0$) and neutral aggregates ($[\text{Li}_2(\text{PF}_6)_2]^0$) diffuse together while carrying zero net charge.

PocketMD computes **Cluster Nernst–Einstein conductivity ($\sigma_{\text{cNE}}$)**, **ionicity ($\mathcal{I} = \sigma_{\text{cNE}}/\sigma_{\text{NE}}$)**, and **cluster transference ($t_+^{\text{cNE}}$)** (`France-Lanord & Grossman, PRL 2019`; `Fong, Self & Persson, ACS Energy Lett. 2019`), along with a **hopping vs. vehicular index ($\chi_{\text{hop}}$)** (`Self et al., 2019`) and **additive solvation enrichment ($\text{CAGS}_{\text{add}}$)** (`Gao, Yao & Zhang, JACS 2023`):

$$\sigma_{\text{cNE}} = \frac{e^2}{V k_{\text{B}} T}\sum_{c} z_c^2 D_c, \qquad \mathcal{I} = \frac{\sigma_{\text{cNE}}}{\sigma_{\text{NE}}}, \qquad \text{CAGS}_{\text{add}} = \ln\!\left(\frac{(\text{CN}_{\text{add}}+\epsilon)/x_{\text{add}}}{(\text{CN}_{\text{solv}}+\epsilon)/x_{\text{solv}}}\right)$$

- **`GO`**: $\sigma_{\text{cNE}} \ge 8.0\text{ mS/cm}$, $t_+^{\text{cNE}} \ge 0.20$, and $\mathcal{I} \ge 0.60$ (for example, EC/DMC + 10% FEC with $\text{CAGS}_{\text{add}} > 0$).
- **`HOLD`**: Good ionicity and additive coordination ($\text{CAGS}_{\text{add}} > 0$), but high viscosity lowers $\sigma_{\text{cNE}} < 8.0\text{ mS/cm}$ — lower the additive fraction or add a linear carbonate co-solvent before long MD.
- **`REJECT`**: Heavy neutral `CIP/AGG` clustering drops $\sigma_{\text{cNE}}$ well below $\sigma_{\text{NE}}$ ($\mathcal{I} < 0.35$).

---

### C. WignerFlow Neural Checkpoint Audit (`checkpoint`)
Runs a `.pt` WignerFlow checkpoint on CPU with `FastRollout` and checks:
1. **Physical stability** (`Fu et al., TMLR 2023`): minimum van der Waals distance ratio, temperature drift, $g(r)$ $L_1$ error, coordination number, self-diffusivity $D$, and center-of-mass kinetic energy leakage ($\max_t K_{\text{COM}}/K_{\text{tot}}$, `Bussi 2007`).
2. **Single-model uncertainty** (`Vita & Schwalbe-Koda 2023`): reuses one `encode()` pass to measure second-order **Heun–Euler ODE curvature** ($\delta_{\text{ODE}}$) and **multi-seed flow spread** ($\sigma_{\text{flow}}$) without needing a multi-model ensemble.

---

## 2. Quickstart CLI and Local Studio Server

```bash
# 1. List built-in presets and public MD dataset import commands
python scripts/pocket_md.py presets

# 2. Screen a protein-ligand binding pose on CPU (~0.7 s)
python scripts/pocket_md.py binding --preset aspirin_cox_stable --replicas 3
python scripts/pocket_md.py binding --pdb my_pose.pdb --xml my_ligand.xml

# 3. Screen a battery electrolyte formulation on CPU
python scripts/pocket_md.py electrolyte --preset li_ec_fec_additive
python scripts/pocket_md.py electrolyte --preset li_high_salt_agg

# 4. Audit a WignerFlow neural checkpoint on CPU
python scripts/pocket_md.py checkpoint --ckpt checkpoints/argon_flow_v0.pt

# 5. Batch-screen all built-in presets and save a JSON report
python scripts/pocket_md.py batch --json-out pocket_batch.json

# 6. Launch the local interactive browser UI
python scripts/pocket_md.py serve --port 8765
```

---

## 3. Importing Public MD Datasets

Use `scripts/import_md.py` to convert external trajectories into WignerFlow's HDF5 format:

1. **MISATO** (~17,000 protein–ligand complexes in explicit water, Zenodo `7711953`):
   ```bash
   python scripts/import_md.py MD.hdf5 --out-dir data/misato_case --dt-fs 2.0 --unit angstrom
   ```
2. **MDbind** (~63,000 short protein–ligand MD trajectories, Zenodo `10390550`):
   ```bash
   python scripts/import_md.py complex.xtc --topology complex.pdb --out-dir data/mdbind_case --dt-fs 2.0
   ```
3. **LIBE / Carbonate Electrolyte MD** ($\text{LiPF}_6$ in EC/DMC/FEC, Figshare `14226857`):
   ```bash
   python scripts/import_md.py li_ec_dmc.dcd --topology top.pdb --out-dir data/electrolyte --dt-fs 1.0
   ```
