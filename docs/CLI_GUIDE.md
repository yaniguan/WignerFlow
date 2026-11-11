# `wignerflow-pocket` Unified Command-Line Guide

`wignerflow-pocket` turns WignerFlow's coarse-stride neural propagator and pure-PyTorch `PocketMicroEngine` into a **zero-boilerplate terminal calculator** that runs in `< 1 s` on any laptop CPU without requiring OpenMM or a GPU.

---

## 1. Installation

Install in editable mode from the repository root:

```bash
pip install -e .
```

This registers both the `wignerflow-pocket` console binary and the `python -m equitraj` module entrypoint.

---

## 2. Subcommand Reference

| Subcommand | Domain | What It Computes on CPU (`< 1 s`) | Decision Output |
| :--- | :--- | :--- | :--- |
| `wignerflow-pocket doctor` | Environment & Speed | Verifies Python, PyTorch, `e3nn`, `numpy`, `scipy`, optional `openmm`/`rdkit`, built-in assets, and runs a 192-step live CPU BAOAB benchmark (`steps/s`, `ns/day`). | `GO` / `REJECT` |
| `wignerflow-pocket binding` | Drug Discovery | Multi-replica OpenBPMD `CompScore = PoseScore - 5 * PersScore`, PoseBusters `min_vdw_ratio`, strain-recoil spike, pose entropy, and 50-fs direction-retention $\tau$-RAMD residence probe. | `GO` / `HOLD` / `REJECT` |
| `wignerflow-pocket electrolyte` | Battery Electrolytes | Periodic $\text{Li}^+$ solvation speciation (`SSIP`/`CIP`/`AGG`), France-Lanord Cluster Nernst-Einstein $\sigma_{\text{cNE}}$, Inverse Haven ionicity $\sigma_{\text{cNE}}/\sigma_{\text{NE}}$, hopping index $\chi_{\text{hop}}$, and additive `CAGS`. | `GO` / `HOLD` / `REJECT` |
| `wignerflow-pocket checkpoint` | Neural MD Propagators | Fast CPU Verlet rollout of `.pt` checkpoints with Fu-2023 structural/thermal stability (`min_vdw_ratio`, `rdf_l1_err_A`, `com_kinetic_leakage_ratio`) + committee-free dual epistemic gate ($\delta_{\text{ODE}}, \sigma_{\text{flow}}$). | `GO` / `HOLD` / `REJECT` |
| `wignerflow-pocket rollout` | Pure Pocket MD | Standalone CPU BAOAB trajectory rollout from built-in presets (`ethanol`, `aspirin`, `ala2`, `li_ec_pf6`) or custom `.xyz` / `.pdb` (`+ .xml`) with automatic covalent bond/angle inference and `.npz`/`.xyz` export. | `GO` / `HOLD` / `REJECT` |
| `wignerflow-pocket catalog` | Public Benchmarks | Prints curated public drug-binding (`MISATO`, `MDbind`, `ATLAS`), electrolyte (`LIBE`, `EDP`), and coarse-stride (`MD17`/`rMD17`) datasets with one-line import hints. | Dataset catalog |

---

## 3. Quick-Start Examples

### 3.1 Environment & Live CPU Speed Check
```bash
wignerflow-pocket doctor
```

### 3.2 Screen Protein-Ligand Docked Poses Before 50-GPU-Hour RB-FEP
```bash
# Built-in COX-1 Aspirin anchor pose vs. flipped clash decoy
wignerflow-pocket binding --preset aspirin_cox_stable
wignerflow-pocket binding --preset aspirin_decoy_clash

# Custom docked complex PDB (+ optional OpenMM XML force field) saved as JSON
wignerflow-pocket binding --pdb my_complex.pdb --xml my_complex.xml --replicas 3 --json --out binding_report.json
```

### 3.3 Screen Battery Electrolyte Additives Before 50 ns MD / AIMD
```bash
# Compare sacrificial SEI + fast-transport FEC vs. neutral CIP/AGG pairing mirage
wignerflow-pocket electrolyte --preset li_ec_fec_additive
wignerflow-pocket electrolyte --preset li_high_salt_agg --json
```

### 3.4 Audit a WignerFlow `.pt` Checkpoint on CPU
```bash
wignerflow-pocket checkpoint --steps 24 --sampling-steps 3 --neighbor verlet
wignerflow-pocket checkpoint --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt --thermostat --json
```

### 3.5 Standalone Pocket MD Trajectory Rollout & `.xyz` / `.npz` Export
```bash
# Zero-file 9-atom ethanol rollout or OpenFF-2.2.1 aspirin rollout
wignerflow-pocket rollout --preset aspirin --frames 80 --dt-fs 1.5 --out-xyz traj_aspirin.xyz --out-npz traj_aspirin.npz

# Roll out any single-frame molecule.xyz with automatic covalent bond/angle inference
wignerflow-pocket rollout --xyz ligand.xyz --temp-k 300 --frames 60 --out-xyz ligand_rollout.xyz
```
