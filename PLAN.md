# EquiTraj plan

Status: v5, approved 2026-09-30. R0 in progress. No code until approved.

Question: can an autoregressive E(3)-equivariant model roll out 10^4+ steps and still give correct RDF/VACF without blowing up?

Changes in v5: the main GPU resource is now Expanse (V100, group allocation cla175), with Hoffman2 for profiling and overflow. Compute is no longer the bottleneck, so all options are tested on Li⁺/EC again (small model). Colab is a last-resort backup.

Changes in v4: compute is limited to Colab Pro (100 compute units per month, no extra spending). All MD and all small-system training move to the Mac; Colab is used only for Li⁺/EC training.

Changes in v3:
- decided all the open questions (see "Decisions");
- every alternative within a choice is now tested (see "Options to test");
- added a Colab A100 plan;
- related work is in [docs/related_work.md](docs/related_work.md), which led to a TrajCast baseline and two new inference switches.

## Code principles

- KISS over DRY. Repeat a few lines rather than add an abstraction.
- Plain PyTorch. No framework on top (no Lightning, no Hydra).
- One file per idea. Each file should be readable top to bottom.
- Comments explain the physics and the non-obvious math, not the Python.
- Few dependencies. Each one is listed below with the reason it is needed.
- Every experiment is one command plus one YAML file.

## Decisions

| Question | Decision |
|---|---|
| Hardware | Expanse V100 32 GB (Slurm, 48 h jobs, group allocation) for Li⁺/EC MD and all GPU training. Hoffman2 (UGE, 24 h campus jobs; group L40S node g19) for Stage 4 (bf16/tf32 need A100/L40S/H200) and overflow. The Mac for development, tests, and MD of the small systems. Colab Pro as last-resort backup. See "Compute". |
| Order | Full pipeline on Argon first, then alanine dipeptide and aspirin, then Li⁺/EC. |
| Default constraints | HBonds with 2 fs. No constraints with 0.5 fs is also tested. |
| System 0b | Both alanine dipeptide and aspirin, both with OpenFF Sage 2.2.1 (changed from ff14SB during R0: Sage builds both from SMILES, with no PDB templates needed). |
| Li⁺ charge | 0.8 by default; 1.0 also tested. |
| Neutralization | Uniform background by default; a PF₆⁻ counterion also tested. |
| Temperature (Li⁺/EC) | 313 K (EC melts at 36.4 °C). |
| Success criteria | VDOS compared against ground truth subsampled to the same Δt; RDF/CN errors judged against the ground truth's own spread over segments of the same length. |
| Stride | One model per n by default; one n-conditioned model also tested. |
| Logging | TensorBoard. Results are copied back to the Mac with rsync into `results/`. |
| Git | Repo at github.com/yaniguan/WignerFlow. The user makes all commits; Claude only leaves changes in the working tree. Package name `equitraj`. |
| Neighbor list | Brute force and cell list both implemented and checked against each other; the faster one is used. |
| Job size limit | ≤24 h per job, checkpoint every 20 min, resumable. I report at every checkpoint before starting the next stage. |

## Things that change the design or the success criteria

1. **With velocities as input, the true dynamics are deterministic.** (x_t, v_t) -> (x_t+Δt, v_t+Δt) is a Hamiltonian flow. A generative head is still useful for two reasons:
   - when Δt approaches the Lyapunov time, the map is so sensitive that a finite model sees an effectively random target;
   - rollouts drift off the training distribution.

   Tested with a velocity-input vs positions-only ablation.
2. **VDOS is limited by Nyquist.** Output is sampled every Δt = n·δt. With n=10 and δt=2 fs, the Nyquist frequency is ~834 cm⁻¹, so the C=O stretch cannot be resolved. Compare against ground truth subsampled to the same Δt, and require peaks below Nyquist to line up. Argon is unaffected.
3. **Rollout vs ground truth is a statistical comparison only.** Trajectories decorrelate after ~1 ps. Compare distributions and correlation functions. Report per-frame error only for the first ~10–50 steps.
4. **One Li⁺ gives poor statistics.** At n=10, 10^4 steps is 200 ps. Exchange may not happen at all. Judge errors against the ground truth's own spread, and run 2 long sparse ground-truth trajectories for residence time.
5. **Constraints.** A C–H stretch has a ~11 fs period. HBonds constraints remove it (default). The unconstrained alternative is also tested.
6. **Forces are saved** for the MACE baseline and for energy rescaling (S5).
7. **Closest prior work** is TrajCast (deterministic, single-step training, CSVR thermostat during rollout) and FlashMD (not equivariant, deterministic, energy rescaling). See docs/related_work.md.

## Options to test

How options are tested:
- One factor at a time around a default configuration. Interactions are checked only for the two most important factors.
- On Argon and ala2, every model and training option runs at full size.
- On Li⁺/EC, every option runs with a small model (64 channels, shorter training). The best configuration is then trained at full size.
- Inference switches need no retraining. They are applied to the same trained models.

| Group | Options (**default** in bold) | Where |
|---|---|---|
| Constraints and step | **HBonds, 2 fs** / none, 0.5 fs | ala2, Li⁺/EC |
| Small molecule | **alanine dipeptide** / aspirin (both OpenFF Sage) | 0b |
| Li⁺ charge | **0.8** / 1.0 | Li⁺/EC |
| Neutralization | **background** / PF₆⁻ | Li⁺/EC |
| Output head | A regression / **B flow matching** | all |
| Flow sampler | **Euler** / Heun; K ∈ {1, 2, 4, **10**, 20} | all (inference only) |
| Denoiser | **plain** / geometry-aware (edges from candidate positions) | Ar, ala2, Li⁺/EC |
| Target scaling | **per-element standard deviation** / mass-scaled (FlashMD: Δx·√m, v·√m) | ala2, aspirin (mixed masses) |
| Equivariant layer | **SO(2) convolution (eSCN)** / full CG tensor product (e3nn) | Ar, ala2, Li⁺/EC |
| l_max | 1 / **2** / 3 | Ar, ala2, Li⁺/EC |
| History k | 1 / **3** / 6 | Ar, ala2, Li⁺/EC |
| Inputs | **positions + velocities** / positions only | Ar, ala2, Li⁺/EC, both heads |
| Stride handling | **one model per n** / n-conditioned model | Ar, ala2, Li⁺/EC |
| Stride n | 1 / 5 / **10** / 50 | all, both heads |
| Input noise σ | 0 / 0.5 / **1** / 2 / 4 % of displacement std | Ar, ala2, Li⁺/EC |
| Unrolling | none / **truncated** / full backprop | Ar, ala2, Li⁺/EC, both heads |
| Max unroll m | 4 / **16** | Ar, ala2, Li⁺/EC |
| S4 constraint projection (RATTLE) | **off** / on | ala2, aspirin, Li⁺/EC (inference only) |
| S5 energy rescaling, FlashMD style | **off** / on | all (inference only) |
| S6 CSVR thermostat, TrajCast style | **off** / on | all (inference only) |
| Neighbor list | brute force / cell list | speed and equality test |
| Precision | **fp32** / tf32 / bf16 (l ≥ 1 kept in fp32) | Stage 4, Li⁺/EC (inference only) |
| Compilation and kernels | eager / torch.compile / + cuEquivariance | Stage 4 (inference only) |
| VACF source | **model velocities** / finite differences of positions | all (analysis only) |

Model-only results always use the defaults for S4–S6 (off). Any result with a switch on is labeled as such.

## Repo layout

```
WignerFlow/
  PLAN.md  RESULTS.md  README.md  environment.yml
  docs/related_work.md
  systems/              # small committed files: OpenMM system XML + starting PDB per system
  equitraj/
    units.py            # all unit conversions in one place
    build_systems.py    # Mac only (needs OpenFF): writes systems/*.xml
    simulate.py         # equilibrate -> NVE production -> HDF5 (needs only OpenMM)
    dataset.py          # loads HDF5, splits by trajectory, makes (context, target) windows
    neighbors.py        # minimum-image neighbor list: brute force and cell list
    so3.py              # Wigner-D matrices, rotate an edge onto the z axis
    layers.py           # SO(2) convolution, CG tensor product layer, attention, temporal attention
    model.py            # full model plus the two heads
    constraints.py      # RATTLE projection (used by S4 and by the MACE Verlet baseline)
    thermostat.py       # CSVR (S6) and energy rescaling (S5)
    baselines.py        # non-equivariant transformer, MACE + Verlet, TrajCast wrapper
    train.py            # training loop: teacher forcing, noise, unrolling; resumable
    rollout.py          # autoregressive generation; resumable
    analysis.py         # RDF, CN, VACF, VDOS, MSD, residence time, energy, momentum
    plots.py
  configs/              # one YAML per experiment
  jobs/                 # Slurm (Expanse) and UGE (Hoffman2) job templates
  scripts/              # thin command-line entry points
  tests/
  results/<system>/<exp_id>/   # config.yaml, metrics.json, figs/
```

## Dependencies

| Package | Why | Where |
|---|---|---|
| openff-toolkit, openff-interchange | parametrize EC, Li⁺, PF₆⁻, aspirin | Mac only (conda) |
| openmm | MD (Mac, OpenCL on the Apple GPU) and reference energies | both |
| torch | model | both |
| e3nn | Wigner-D, spherical harmonics, reference CG tensor product | both |
| h5py, numpy, matplotlib, pyyaml, pytest, tensorboard | basics | both |
| mdtraj | topology, bond lists | both |
| mace-torch | MLIP baseline | both |
| trajcast (from GitHub, Apache-2.0) | closest-work baseline | both |
| cuequivariance-torch | Stage 4 only | Hoffman2 |

Force-field XML files are built once on the Mac and committed. On the Mac, anything on MPS runs in float32; the float64 tests run on CPU because MPS has no float64. On the clusters, a conda env is built from the same `environment.yml`.

## Conventions

- Raw HDF5 files use OpenMM units (nm, ps, kJ/mol). The model uses Å and fs. Conversion happens only in `units.py`.
- One HDF5 file per trajectory with `positions` (unwrapped), `velocities`, `forces`, `potential_energy`, `kinetic_energy`, `box`, `time`, plus metadata attributes.
- Split by trajectory, about 70/15/15. The test set is never used for tuning.
- Stride n is counted in MD steps.
- Rollout keeps positions in float64 and removes the center-of-mass part of Δx and v every step.
- Metrics use ≥5 seeds × ≥3 test starting conditions. Report median and IQR.

## Stage 0: data (OpenMM)

| | 0a Argon | 0b-1 Alanine dipeptide | 0b-2 Aspirin | 0c Li⁺ in EC |
|---|---|---|---|---|
| Atoms | 256 | 22 | 21 | 1001 (+7 with PF₆⁻) |
| Force field | LJ, ε=0.996 kJ/mol, σ=3.405 Å, switched 8.0–8.5 Å | OpenFF Sage 2.2.1 | OpenFF Sage 2.2.1 | EC: Sage 2.2; Li⁺: Sage ions (Joung–Cheatham LJ); PF₆⁻: Sage |
| Box | cubic PBC, L≈23.1 Å, cutoff 8.5 Å | vacuum | vacuum | cubic PBC, L≈22.3 Å after NPT; PME, 9 Å cutoff |
| Temperature | 94.4 K | 300 K | 300 K | 313 K |
| Default step | 5 fs, no constraints | HBonds, 2 fs | HBonds, 2 fs | HBonds, 2 fs |
| Equilibration | NVT 200 ps | NVT 200 ps from C7eq, C7ax, αR | NVT 200 ps | NPT 2 ns, then NVT 1 ns |
| NVE production | 10 × 0.5 ns | 10 × 2 ns | 10 × 1 ns | 12 × 0.5 ns, plus 2 × 20 ns saved every 1 ps |
| Saved every | 1 step | 1 step | 1 step | 5 steps (forces every 50), plus 4 × 100 ps saved every step for n=1 |
| Disk | ~9 GB | ~8 GB | ~4 GB | ~18 GB |

Variant datasets (smaller, used by one-factor tests):
- ala2 without constraints at 0.5 fs, 5 × 1 ns;
- Li⁺/EC without constraints at 0.5 fs;
- Li⁺/EC with charge 1.0;
- Li⁺/EC with PF₆⁻.

Each Li⁺/EC variant is 6 × 0.5 ns, about 9 GB. Use mixed precision so the NVE ground truth has low drift.

Report: energy drift per trajectory, OpenMM ns/day on the Mac, ground-truth RDF/VACF/MSD, and how Li⁺ charge and counterion change the Li–O RDF and exchange rate.

Done when:
- Argon: first g(r) peak near 3.7 Å with height ~3.0; D within 15% of ~2.4e-5 cm²/s; VACF has a negative dip near 0.3 ps.
- ala2: φ/ψ covers C7eq and C7ax.
- Li⁺/EC: density within 3% of 1.32 g/cm³; first Li–O(carbonyl) peak at ~1.9–2.1 Å; CN≈4.

## Stage 1: model

### Inputs and outputs

- Per atom: atom type (scalar), velocity (vector), displacement since the previous frame (vector), for each of the last k frames.
- Edges: minimum-image relative vectors within a cutoff (5 Å for Li⁺/EC, 7 Å for Ar, 6 Å for molecules).
- Outputs: Δx and v_new per atom, scaled as set by the "target scaling" option.
- Under PBC the box breaks global rotation symmetry. Equivariance holds only locally (or exactly, if the box vectors are rotated too). Angular momentum is not conserved under PBC.

### Architecture

```
for each of the last k frames:
    spatial encoder (2 equivariant attention layers, l_max = 2) -> per-atom features
    (cache these; during rollout only the new frame is encoded)
temporal attention per atom, causal over the k frames:
    attention weights come only from invariants (l=0 features and norms)
    values are mixed per irrep, which keeps equivariance
fusion (2 equivariant attention layers on the current frame) -> condition features c
head A: linear -> (Δx, v)
head B: flow matching denoiser -> sample (Δx, v)
```

- SO(2) convolution (eSCN): rotate each edge onto z with Wigner-D, apply a small linear map per m, rotate back.
- Reflections: use only irreps with parity (-1)^l, and set the weights that mix m with -m antisymmetrically to zero. That term flips sign under a mirror through the z axis. This should make the layer O(3)-equivariant; the reflection test will check it. The full CG tensor product layer is implemented anyway as an option to compare against.
- Starting size: 128 channels (64 for small Li⁺/EC runs), k=3, ~2–5M parameters.

### Head B: flow matching

- Target y = scaled (Δx, v_new). Interpolate y_τ = (1-τ)·ε + τ·y and train the network to predict y - ε.
- Noise ε: isotropic Gaussian per atom with the center of mass removed. This distribution is rotation invariant.
- Denoiser: 2–3 small equivariant layers taking (y_τ, τ, c). Optional geometry-aware variant.
- The encoder runs once per rollout step; the denoiser runs K times.

### Tests

| Test | Checks |
|---|---|
| test_wigner | D is orthogonal; D(R)Y(r) = Y(Rr); edge rotation maps r onto z |
| test_equivariance | random rotation, reflection, and translation; both layer types and both heads. float64 to 1e-10, float32 to 1e-5 relative. Head B uses fixed, identically rotated noise. |
| test_permutation | permuting atoms permutes outputs |
| test_pbc | shifting one atom by a box vector changes nothing; rotating atoms and box together is equivariant |
| test_neighbors | brute force and cell list give the same pairs (cubic box, triclinic box, small box) |
| test_causal | changing a future frame does not change past outputs |
| test_com | no center-of-mass velocity after a rollout step |
| test_constraints | RATTLE restores bond lengths and removes velocity along bonds |
| test_thermostat | CSVR preserves the canonical kinetic energy distribution on a toy system |
| test_dataset | windows never cross trajectories; stride indexing is correct |
| test_resume | a job killed mid-training resumes to the same state |
| test_overfit | the model fits 16 frames to near-zero loss |

Done when: all tests pass, and on Argon the one-step error beats the constant-velocity guess x + vΔt.

## Stage 2: training against error accumulation

Strategies, added one at a time:

| ID | Strategy |
|---|---|
| S1 | one-step teacher forcing |
| S2 | + Gaussian noise on context frames |
| S3a | + unrolled training, truncated backprop |
| S3b | + unrolled training, full backprop with gradient checkpointing; curriculum m = 1 → 4 → 16 |

- For head B, unrolled steps use few-step samples (K=1–2) as inputs.
- Models are selected by stability and RDF error over 1000-step validation rollouts, not by one-step loss.
- Relative velocity MAE is tracked as an early warning (TrajCast reports instability above ~2–4%).
- The other options from "Options to test" run one at a time around the best recipe.

## Stage 3: evaluation (the model replaces the integrator)

- Rollout: ≥10^4 steps from each test starting condition. No OpenMM integration. Rollouts are batched.
- Crash criteria:
  - Hard: a bond deviates more than 50%; any pair is closer than 0.5 Å (0.7σ for Ar); a NaN appears; temperature exceeds 3 × T₀.
  - Soft: windowed RDF L1 exceeds 3× the ground-truth spread; temperature drifts more than 10%; bond RMS deviation exceeds 10%.
  - Report the distribution of steps before a crash (Kaplan–Meier curve).
- Structure: g(r) per atom pair; first Li–O peak position, height, CN; RDF L1 with ground-truth error bars.
- Dynamics: VACF (model velocities and finite differences), VDOS against subsampled ground truth, MSD and diffusion.
- Solvation: CN(t), and residence time from the long ground-truth runs and from rollouts.
- Energy: the reference force field computes potential energy for generated frames, and model velocities give kinetic energy. This is an external check; the model has no Hamiltonian. Constraint violations are reported separately.
- Vacuum: drift in linear and angular momentum.
- Speed: wall time per ns, model vs OpenMM, on the same GPU with the same batching.
- Baselines:
  - non-equivariant transformer with the same parameter count and rotation augmentation;
  - MACE trained on forces, run with our own velocity Verlet + RATTLE at 2 fs;
  - **TrajCast** trained on our data, with and without its CSVR thermostat;
  - head A vs head B.
- Diagnosis: per time window, find what goes wrong first: bond lengths, angles, temperature, intermolecular RDF, or the Li⁺ shell.
- Output: `results/.../metrics.json` and figures; `RESULTS.md` is generated by a script.

## Stage 4: profiling (no custom CUDA)

- torch.profiler and Nsight Systems on Hoffman2: time per step, split into neighbor list, Wigner rotation, SO(2) linear, attention, and denoiser × K.
- Compare eager vs torch.compile vs cuEquivariance, and fp32 vs tf32 vs bf16. Check whether lower precision changes rollout stability.
- Report where a custom kernel would help and a rough upper bound on the gain. Do not implement it.

## Compute

Checked on 2026-09-29/30:

| Resource | GPUs usable | Limits | Notes |
|---|---|---|---|
| Expanse (SDSC) | ~200 V100 32 GB (4 per node) | 48 h (`gpu`, `gpu-shared`), 7 days preemptible (`gpu-preempt`) | group allocation cla175: ~155,000 GPU SUs left, shared with the group. The A100 nodes are reserved and not usable. ~130 GPU jobs were queued. |
| Hoffman2 (UCLA) | campus: 8 A100 40 GB, 4 A100 80 GB, 4 H200; idle group nodes: L40S, H100; own group: g19 with 2 L40S (14-day highp, shared with labmates) | 24 h campus, 14 days on g19 | home quota full: use /u/project/sautet or $SCRATCH. The campus GPU queue is busy. |
| Mac M3 Pro | Apple GPU (OpenCL for OpenMM, MPS for torch) | — | 107 GB free disk |
| Colab Pro | 100 units/month (~18 A100-h) | no background execution | backup only |

Estimate: v3 needed ~250–350 A100-hours. V100 is ~2–3× slower for these models, so ~500–1,000 V100-hours, under 1% of the group allocation. The limit is queue time, not budget. Many small jobs can run in parallel on `gpu-shared` (1 GPU each).

Where things run:
- Mac: code, tests, Argon/ala2/aspirin MD, small smoke runs.
- Expanse: Li⁺/EC MD, all training, all evaluation rollouts.
- Hoffman2: Stage 4 profiling (A100/L40S/H200 for tf32, bf16, cuEquivariance), overflow.

Workflow: I write code on the Mac, push to GitHub, pull on the cluster, and submit Slurm/UGE jobs over SSH. I copy results back with rsync into `results/`. Raw trajectories stay on the cluster (Expanse Lustre); small systems also stay on the Mac.

## Checkpoints (stop and report at each)

1. R0: repo skeleton, environments (Mac, Expanse), Argon ground truth data and analysis, OpenMM speed on the Mac and on a V100.
2. R1: model and tests pass; one-step accuracy on Argon.
3. R2: full pipeline and all options on Argon; measured per-run cost. Main go/no-go point.
4. R3: ala2 and aspirin results, including momentum, angular momentum, and the constraint variant.
5. R4: Li⁺/EC data (all variants) and ground-truth analysis.
6. R5: Li⁺/EC option tests and stride sweep.
7. R6: Li⁺/EC final evaluation, baselines, verdict on the success criteria.
8. R7: profiling report; RESULTS.md final.

Each report covers what was done, numbers, plots, problems, and next steps. Missed targets are reported as missed, with a diagnosis.

## Risks

| Risk | Mitigation |
|---|---|
| Parity trick in SO(2) conv fails | the CG tensor product layer is already implemented |
| Generative head still blows up | geometry-aware denoiser, unrolled training, switches S4–S6 (reported separately) |
| Regression head blurs structure at large n | expected; measure it through RDF peak height |
| Too few Li⁺ statistics | bootstrap error bars, several starting conditions |
| No exchange events inside a rollout | n=50 (1 ns rollouts); report honestly |
| Overlap with TrajCast/FlashMD | position the work as a systematic study; TrajCast as a baseline |
| Long cluster queues | many 1-GPU jobs on `gpu-shared`; Hoffman2 as overflow; `gpu-preempt` for resumable jobs |

## Implementation notes (deviations found while building)

- NVE production uses a custom velocity Verlet integrator. OpenMM's VerletIntegrator is leapfrog, so its velocities lag the positions by half a step.
- Before NVE, velocities are rescaled so that the starting total energy equals the NVT average energy. Without this, the NVE temperature of 256-atom argon scattered between 86 and 96 K (mean 92.4 K instead of 94.4 K).
- The CMMotionRemover is removed from OpenFF systems because it edits velocities every step, which breaks NVE. The center-of-mass velocity is removed once before production instead.
- Li⁺/EC: each trajectory runs its own NPT (2 ns) → NVT (1 ns) → NVE, so trajectories are fully independent. Each one therefore has a slightly different box.
- Temporal attention uses only the newest frame as the query. The model only predicts the next frame, so causality holds by construction, and the planned `test_causal` is not needed.
- The equivariant RMS norm uses eps=1e-3. With 1e-6, atoms with near-zero l>0 features amplified float32 noise, giving up to 4e-3 equivariance error.
- float32 equivariance error (untrained model, 40 random inputs): median ~4e-6, worst ~2e-5. This is slightly above the 1e-5 target in the worst case. float64 error is below 1e-10.

## Remaining questions

None blocking. Once you approve this plan, I start R0.
