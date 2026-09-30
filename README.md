# WignerFlow

An autoregressive E(3)-equivariant model that generates molecular dynamics trajectories directly. Given the last k frames (positions and velocities), it predicts the frame Δt = n·δt later, without a numerical integrator.

> **Status: planning.** No code has been written yet. This README describes the intended design. See [PLAN.md](PLAN.md) for the full plan and [RESULTS.md](RESULTS.md) for results once they exist.

## The question

Autoregressive models in continuous space accumulate errors. After many steps, the generated states drift away from the training distribution and the rollout can blow up.

This project tests whether an equivariant model can stay physically reasonable after 10^4+ autoregressive steps. It checks four things:

- no crashes (broken bonds, overlapping atoms);
- correct radial distribution functions (RDF);
- correct velocity autocorrelation (VACF) and vibrational density of states (VDOS);
- a reasonable total energy, measured with the reference force field.

Every design choice aimed at preventing blow-up is tested with an ablation.

## Approach

- **Data.** OpenMM simulations of three systems, easiest first:
  1. Lennard-Jones argon (256 atoms, periodic box);
  2. alanine dipeptide in vacuum (22 atoms, where E(3) symmetry is exact);
  3. one Li⁺ in ethylene carbonate (1001 atoms, periodic box).

  Production runs use NVE. Train, validation and test sets are split by trajectory, not by frame.
- **Model.**
  - Per-frame spatial encoder: equivariant graph attention, eSCN-style SO(2) convolutions with Wigner-D edge alignment, l_max = 2.
  - Causal attention over the past k frames for each atom. Attention weights come only from invariants, so the layer stays equivariant.
  - The model predicts displacements and new velocities, not absolute positions.
- **Two output heads.**
  - (A) Deterministic regression (MSE).
  - (B) Equivariant conditional flow matching in displacement space.
- **Training against error accumulation**, added one step at a time:
  1. teacher forcing;
  2. input noise;
  3. unrolled (pushforward) training with a curriculum over the unroll length;
  4. optional constraint projection at inference, reported both on and off.
- **Evaluation.**
  - The model alone generates ≥10^4 steps.
  - Metrics: stability, RDF, VACF/VDOS, Li⁺ solvation-shell dynamics, energy drift, and momentum drift in vacuum.
  - The same metrics are reported as the stride n varies.
  - Baselines: a non-equivariant transformer, MACE with a Verlet integrator, and TrajCast.

### Limits to keep in mind

- Under periodic boundary conditions, the box breaks global rotation symmetry. Equivariance holds only locally, and total angular momentum is not conserved.
- The model has no Hamiltonian. Energies are computed afterwards with the reference force field, only as a check.
- Classical OpenMM is already fast, so this is a proof of concept for the method. The practical payoff would come from replacing ab initio MD.

## Repository layout

```
equitraj/
  units.py        unit conversions (OpenMM nm/ps/kJ/mol <-> model Å/fs)
  systems.py      OpenMM systems for argon, alanine dipeptide, Li+/EC
  simulate.py     equilibration and NVE production, written to HDF5
  dataset.py      trajectory-split datasets with stride and context windows
  neighbors.py    minimum-image neighbor list
  so3.py          Wigner-D matrices and edge alignment
  layers.py       SO(2) convolution, equivariant attention, temporal attention
  model.py        full model and the two output heads
  baselines.py    non-equivariant transformer, MACE wrapper
  train.py        training loop: teacher forcing, noise, unrolling
  rollout.py      autoregressive generation
  analysis.py     RDF, CN, VACF, VDOS, MSD, residence time, energy, momentum
  plots.py
configs/          one YAML file per experiment
scripts/          command-line entry points
tests/            pytest suite (equivariance, permutation, PBC, neighbors, ...)
results/          one folder per experiment: config, metrics.json, figures
```

## How to run

### Install

OpenMM and the OpenFF toolkit are most reliable from conda-forge, so the environment is managed with conda. PyTorch also comes from conda-forge: the pip wheel ships its own OpenMP library and crashes when imported next to conda-forge OpenMM.

```bash
git clone https://github.com/yaniguan/WignerFlow.git
cd WignerFlow
conda env create -f environment.yml
conda activate wignerflow
pip install -e .
```

On a CUDA cluster, use the same file but ask for a CUDA build of PyTorch (`pytorch=*=cuda*`).

Force-field files for the OpenFF-parametrized systems (Li⁺/EC variants, alanine dipeptide, aspirin) are already in `systems/`. To rebuild them:

```bash
python -m equitraj.build_systems
```

### Test

```bash
pytest tests/
```

### Ground-truth data

```bash
python scripts/gen_data.py --config configs/data/argon.yaml             # all trajectories of one system
python scripts/gen_data.py --config configs/data/li_ec_q08.yaml --traj 3  # one trajectory (one cluster job)
python scripts/gen_data.py --config configs/data/argon.yaml --benchmark   # OpenMM speed only
python scripts/analyze_gt.py --data data/argon --out results/argon/ground_truth
```

### Training

```bash
python scripts/make_configs.py argon          # one YAML per experiment in configs/argon/
python -m equitraj.train --config configs/argon/B_S3a.yaml
```

Training resumes from `ckpt.pt` in the output folder if it exists. `best.pt` is the checkpoint with the best short validation rollout.

### Cluster jobs

```bash
sbatch -J B_S3a -t 24:00:00 jobs/expanse_gpu.sbatch python -m equitraj.train --config configs/argon/B_S3a.yaml   # Expanse (Slurm)
qsub -N B_S3a -l gpu,A100,cuda=1,h_rt=24:00:00,h_data=32G jobs/hoffman2_gpu.sh python -m equitraj.train --config configs/argon/B_S3a.yaml   # Hoffman2 (UGE)
bash jobs/expanse_submit_md.sh configs/data/li_ec_q08.yaml    # one MD job per trajectory
```

Trajectory data (HDF5, tens of GB) and checkpoints are not stored in git.

## Results

None yet. See [RESULTS.md](RESULTS.md) once the first stage is done.

## Related work

The closest prior work is TrajCast (Thiemann et al., 2025) and FlashMD (Bigi et al., NeurIPS 2025). Generative approaches include Timewarp, ITO, TITO and MDGen. See [docs/related_work.md](docs/related_work.md) for a comparison and for what this project does and does not add.

## License

Not chosen yet.
