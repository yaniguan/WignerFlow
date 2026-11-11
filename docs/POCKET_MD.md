# PocketMD: WignerFlow Fast CPU Triage Calculator ("随身科学计算器")

Before spending **50–500 GPU-hours** on full explicit-solvent MD, RB-FEP free-energy perturbations, or *ab initio* MD (AIMD), computational chemists and battery scientists need a 1-second "market calculator" check on a laptop CPU:

> **"Does this docked pose / electrolyte additive survive thermal fluctuations, or should we reject it before Big MD?"** (`GO` / `HOLD` / `REJECT`)

`equitraj.pocket` and `scripts/pocket_md.py` package WignerFlow's coarse-stride `FastRollout` engine, OpenFF XML/PDB parsers, and trajectory observables into a zero-OpenMM, CPU-friendly pre-screening application.

---

## 1. Three Built-In Workflows

### A. Drug Small-Molecule + Protein Pocket Pose Triage (`binding`)
 Docking scores frequently rank sterically strained or poorly anchored false-positive decoys near the top. Short thermal relaxation separates locked native poses from unstable decoys:

| Observable | `GO` (Promote to RB-FEP) | `HOLD` (5 ns Equil. Check) | `REJECT` (Skip Big MD) |
|---|---|---|---|
| Pocket-aligned Ligand RMSD | $\le 1.6\text{ Å}$ | $1.6\text{–}3.0\text{ Å}$ | $\ge 3.0\text{ Å}$ |
| Contact Persistence $R$ | $\ge 0.58$ | $0.35\text{–}0.58$ | $< 0.35$ |
| Pocket Unbinding Event | `None` | `None` | Drift $\ge 4.5\text{ Å}$ |

### B. Battery Electrolyte + New Additive Screening (`electrolyte`)
Screening a new film-forming or solvation-tuning small molecule (e.g. FEC, VC, fluorinated ethers) requires checking both **primary $\text{Li}^+$ solvation sheath participation** (`SCN`, `Additive CN`, `Anion CN`, `SSIP / CIP / AGG`) and **ionic transport** via Einstein MSD and Nernst–Einstein conductivity:

$$\sigma_{\text{NE}} = \frac{e^2}{V k_{\text{B}} T}\left(N_+ z_+^2 D_+ + N_- z_-^2 D_-\right), \qquad t_+^{\text{NE}} = \frac{N_+ z_+^2 D_+}{N_+ z_+^2 D_+ + N_- z_-^2 D_-}$$

- **`GO`**: Low ion-cluster trapping ($\text{AGG} < 0.35$, $\text{CIP}+\text{AGG} < 0.65$), $\sigma_{\text{NE}} \ge 8.0\text{ mS/cm}$, $t_+^{\text{NE}} \ge 0.20$.
- **`HOLD`**: Good solvation shell (additive enters primary $\text{Li}^+$ sheath) but viscosity lowers $\sigma_{\text{NE}} < 8.0\text{ mS/cm}$ — lower additive wt% or add DMC co-solvent before 50 ns MD.
- **`REJECT`**: Severe `CIP/AGG` ion clustering traps $\text{Li}^+$.

### C. WignerFlow Neural Checkpoint Rollout (`checkpoint`)
Runs any `.pt` WignerFlow release checkpoint on CPU with Verlet neighbor-graph reuse (`FastRollout`) and automatically checks kinetic temperature stability, radial distribution function ($g(r)$ $L_1$ error & coordination number), minimum pair distance, and self-diffusion $D$.

---

## 2. Quickstart CLI & Local Web Calculator

```bash
# 1. List built-in presets and public MD dataset import commands
python scripts/pocket_md.py presets

# 2. Screen a docked drug pose (~0.2 s on CPU) or your own PDB + OpenFF XML
python scripts/pocket_md.py binding --preset aspirin_cox_stable
python scripts/pocket_md.py binding --pdb my_pose.pdb --xml my_ligand.xml

# 3. Screen a battery electrolyte + additive formulation (~0.5 s on CPU)
python scripts/pocket_md.py electrolyte --preset li_ec_fec_additive

# 4. Run WignerFlow neural checkpoint CPU rollout triage
python scripts/pocket_md.py checkpoint --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt

# 5. Batch-screen all candidates and estimate GPU-hours saved
python scripts/pocket_md.py batch --json-out pocket_batch.json

# 6. Launch zero-dependency local browser calculator UI
python scripts/pocket_md.py serve --port 8765
```

---

## 3. Connecting Real Public Drug & Electrolyte MD Datasets

Use `scripts/import_md.py` to ingest external trajectories into WignerFlow's HDF5 format for surrogate training or direct observable triage:

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
