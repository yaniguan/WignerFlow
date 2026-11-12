# `wignerflow-pocket` command-line guide

`wignerflow-pocket` is a small command-line tool. It wraps the CPU pocket MD engine
(`equitraj/pocket.py`) and the WignerFlow checkpoint rollout so you can run quick
checks from the terminal. It does not need a GPU or OpenMM.

These are fast screening checks on small toy systems. They are not a replacement
for full MD, free-energy methods, or lab data.

## 1. Install

From the repository root:

```bash
pip install -e .
```

This adds the `wignerflow-pocket` command. You can also run `python -m equitraj`.

## 2. Subcommands

| Subcommand | What it does | Output |
| :--- | :--- | :--- |
| `doctor` | Checks Python, PyTorch, `e3nn`, `numpy`, `scipy`, optional `openmm` and `rdkit`, and runs a short CPU speed test. | `GO` / `REJECT` |
| `binding` | Runs a few short replicas of a ligand in a protein pocket. Reports a pose drift score, contact survival, atom overlap, and a short random-push (tau-RAMD style) unbinding time. | `GO` / `HOLD` / `REJECT` |
| `electrolyte` | Runs a small periodic Li+ electrolyte box. Reports free ions vs. ion pairs vs. clusters, and simple conductivity estimates. | `GO` / `HOLD` / `REJECT` |
| `checkpoint` | Rolls out a WignerFlow `.pt` checkpoint on CPU and checks atom overlap, RDF error, and center-of-mass drift. | `GO` / `HOLD` / `REJECT` |
| `rollout` | Runs a CPU Langevin (BAOAB) trajectory from a built-in preset (`ethanol`, `aspirin`, `ala2`, `li_ec_pf6`) or your own `.xyz` / `.pdb` file. Saves `.npz` or `.xyz`. | `GO` / `HOLD` / `REJECT` |
| `catalog` | Lists public datasets (for example MISATO, ATLAS, LIBE, MD17) with a one-line hint on how to get them. | List |

Add `--json` to any subcommand to print a JSON report, and `--out FILE` to save it.

## 3. Examples

Check your setup:

```bash
wignerflow-pocket doctor
```

Compare a stable ligand pose with a clashing decoy (built-in toy presets):

```bash
wignerflow-pocket binding --preset aspirin_cox_stable
wignerflow-pocket binding --preset aspirin_decoy_clash

# Your own complex (PDB, plus an optional OpenMM XML force field)
wignerflow-pocket binding --pdb my_complex.pdb --xml my_complex.xml --replicas 3 --json --out binding_report.json
```

Compare two built-in electrolyte presets:

```bash
wignerflow-pocket electrolyte --preset li_ec_fec_additive
wignerflow-pocket electrolyte --preset li_high_salt_agg --json
```

Check a WignerFlow checkpoint on CPU:

```bash
wignerflow-pocket checkpoint --steps 24 --sampling-steps 3 --neighbor verlet
wignerflow-pocket checkpoint --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt --thermostat --json
```

Run a short trajectory and save it:

```bash
wignerflow-pocket rollout --preset aspirin --frames 80 --dt-fs 1.5 --out-xyz traj_aspirin.xyz --out-npz traj_aspirin.npz
wignerflow-pocket rollout --xyz ligand.xyz --temp-k 300 --frames 60 --out-xyz ligand_rollout.xyz
```

For `.xyz` input, bonds and angles are guessed from covalent radii.
