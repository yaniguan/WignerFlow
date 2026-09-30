# EquiTraj plan

Status: draft, waiting for approval. No code until approved.

Question: can an autoregressive E(3)-equivariant model roll out 10^4+ steps and still give correct RDF/VACF without blowing up?

## Code principles

- KISS over DRY. Repeat a few lines rather than add an abstraction.
- Plain PyTorch. No framework on top (no Lightning, no Hydra).
- One file per idea. Each file should be readable top to bottom.
- Comments explain the physics and the non-obvious math, not the Python.
- Few dependencies. Each one is listed below with the reason it is needed.
- Every experiment is one command plus one YAML file.

## Things that change the design or the success criteria

1. **With velocities as input, the true dynamics are deterministic.** (x_t, v_t) -> (x_t+Δt, v_t+Δt) is a Hamiltonian flow. A generative head is still useful for two reasons:
   - when Δt approaches the Lyapunov time (~0.1–1 ps in liquids), the map is so sensitive that a finite model sees an effectively random target, and a regression head will predict the mean;
   - rollouts drift off the training distribution.

   Plan: add an ablation with velocity input vs positions only, for both heads.
2. **VDOS is limited by Nyquist.** Model output is sampled every Δt = n·δt. With n=10 and δt=2 fs, the Nyquist frequency is ~834 cm⁻¹, so the C=O stretch (~1800 cm⁻¹) cannot be resolved. Proposal: compare against ground truth subsampled to the same Δt, and require peaks below Nyquist to line up. Argon is unaffected (all modes are below ~70 cm⁻¹).
3. **Rollout vs ground truth is a statistical comparison only.** Trajectories from the same start decorrelate after ~1 ps. Compare distributions and correlation functions. Report per-frame error only for the first ~10–50 steps.
4. **One Li⁺ gives poor statistics.** At n=10, 10^4 steps is 200 ps. Ligand exchange may not happen at all, so CN≈4 passes trivially. Proposal: judge RDF and CN errors against the ground truth's own spread over segments of the same length (bootstrap). Run 2 long sparse ground-truth trajectories (20 ns) to estimate residence time.
5. **Constraints.** A C–H stretch has a ~11 fs period. Without constraints, n=10 would step over it. Plan: HBonds constraints with 2 fs steps, the standard setup. The optional physics correction (Stage 2.4) then becomes SHAKE/RATTLE projection.
6. **EC melts at 36.4 °C.** Simulate at 313 K.
7. **Forces are needed** to train the MACE baseline, so forces are saved too.
8. **Close prior work** (details to verify): TrajCast (Thiemann et al. 2025), FlashMD (Bigi et al. 2025), Timewarp, ITO, MDGen. We need to state how this work differs.

## Repo layout

```
WignerFlow/
  PLAN.md  RESULTS.md  README.md
  environment.yml
  equitraj/
    units.py            # all unit conversions in one place
    systems.py          # builds the OpenMM System for argon, ala2, li_ec
    simulate.py         # minimize -> equilibrate -> NVE production -> HDF5
    dataset.py          # loads HDF5, splits by trajectory, makes (context, target) windows
    neighbors.py        # minimum-image neighbor list
    so3.py              # Wigner-D matrices, rotate an edge onto the z axis
    layers.py           # SO(2) convolution, equivariant attention, temporal attention
    model.py            # full model plus the two heads
    baselines.py        # non-equivariant transformer, MACE wrapper
    train.py            # training loop: teacher forcing, noise, unrolling
    rollout.py          # autoregressive generation, no OpenMM integration
    analysis.py         # RDF, CN, VACF, VDOS, MSD, residence time, energy, momentum
    plots.py
  configs/              # one YAML per experiment
  tests/
  scripts/              # thin command-line entry points
  results/<system>/<exp_id>/   # config.yaml, metrics.json, figs/
```

If a file grows past ~500 lines, split it. Until then, keep it in one file.

## Dependencies

| Package | Why |
|---|---|
| openmm, openff-toolkit, openff-interchange | MD data and reference energies. Install from conda-forge. |
| torch | model |
| e3nn | Wigner-D and spherical harmonics only; tested code for the tricky math |
| h5py | trajectory storage |
| mdtraj | topology, bond lists |
| numpy, matplotlib, pyyaml, pytest | basics |
| mace-torch, openmm-torch | MLIP baseline only |
| tensorboard | logging (or wandb, see Q8) |
| cuequivariance | Stage 4 only, CUDA only |

Local machine: M3 Pro, 18 GB, no CUDA. Enough for Argon, ala2, and tests. Li⁺/EC training needs a CUDA machine.

## Conventions

- Raw HDF5 files use OpenMM units (nm, ps, kJ/mol). The model uses Å and fs. Conversion happens only in `units.py`.
- One HDF5 file per trajectory with `positions` (unwrapped), `velocities`, `forces`, `potential_energy`, `kinetic_energy`, `box`, `time`, plus metadata attributes.
- Split by trajectory, about 70/15/15. The test set is never used for tuning.
- Stride n is counted in MD steps. One model per n.
- Rollout keeps positions in float64 (x += Δx) and removes the center-of-mass part of Δx and v every step.
- Metrics use ≥5 seeds × ≥3 test starting conditions. Report median and IQR.

## Stage 0: data (OpenMM)

| | 0a Argon | 0b Alanine dipeptide, vacuum | 0c Li⁺ in EC |
|---|---|---|---|
| Atoms | 256 | 22 | 1001 (1 Li⁺ + 100 EC) |
| Force field | LJ, ε=0.996 kJ/mol, σ=3.405 Å | amber ff14SB (built into OpenMM) | EC: OpenFF Sage 2.2; Li⁺: Sage ion parameters (Joung–Cheatham LJ) |
| Box | cubic PBC, L≈23.1 Å | none | cubic PBC, L≈22.3 Å after NPT; PME, 9 Å cutoff |
| Temperature | 94.4 K | 300 K | 313 K |
| Constraints, step | none, 5 fs | HBonds, 2 fs | HBonds, 2 fs |
| Equilibration | NVT 200 ps | NVT 200 ps, starting from C7eq, C7ax, αR | NPT 2 ns, then NVT 1 ns |
| NVE production | 10 × 0.5 ns | 10 × 2 ns | 12 × 1 ns, plus 2 × 20 ns saved every 1 ps |
| Saved every | 1 step | 1 step | 5 steps (forces every 50), plus 4 × 200 ps saved every step for n=1 |
| Disk | ~9 GB | ~8 GB | ~40 GB |

- Li⁺ charge scaled to 0.8 by default. Full charges overbind Li⁺ and slow exchange. One full-charge ground-truth run is kept for reference.
- The single Li⁺ makes the box net +1. PME adds a neutralizing background. Adding a PF₆⁻ counterion is optional (Q5).
- Use mixed precision so the NVE ground truth has low drift.

Report: energy drift per trajectory, OpenMM ns/day, ground-truth RDF/VACF/MSD.

Done when:
- Argon: first g(r) peak near 3.7 Å with height ~3.0; D within 15% of ~2.4e-5 cm²/s; VACF has a negative dip near 0.3 ps.
- Ala2: φ/ψ covers C7eq and C7ax.
- Li⁺/EC: density within 3% of 1.32 g/cm³; first Li–O(carbonyl) peak at ~1.9–2.1 Å; CN≈4.

## Stage 1: model

### Inputs and outputs

- Per atom: atom type (scalar), velocity (vector), displacement since the previous frame (vector), for each of the last k frames.
- Edges: minimum-image relative vectors within a cutoff (5 Å for Li⁺/EC, 7 Å for Ar, 6 Å for ala2).
- Outputs: Δx and v_new per atom, scaled by the per-element standard deviation from training data.
- Under PBC the box breaks global rotation symmetry. Equivariance holds only locally (or exactly, if the box vectors are rotated too). Angular momentum is not conserved under PBC. The README will say this.

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

- SO(2) convolution (eSCN): rotate each edge onto z with Wigner-D, apply a small linear map per m, rotate back. Cost O(L³) instead of a full Clebsch–Gordan product.
- Reflections: eSCN is only SO(3)-equivariant. Plan: use only irreps with parity (-1)^l and set the weights that mix m with -m antisymmetrically to zero. That term flips sign under a mirror through the z axis. This should make the layer O(3)-equivariant, but it is not yet verified. The reflection test will check it. If it fails, fall back to e3nn's full tensor product at l_max ≤ 2.
- Starting size: 64–128 channels, k=3, ~2–5M parameters.

### Head B: flow matching (main approach)

- Target y = normalized (Δx, v_new). Interpolate y_τ = (1-τ)·ε + τ·y and train the network to predict y - ε.
- Noise ε: isotropic Gaussian per atom, with the center of mass removed. This distribution is rotation invariant.
- Denoiser: 2–3 small equivariant layers taking (y_τ, τ, c). Optional variant: build edges from the candidate positions x + Δx_τ so the denoiser can see overlaps.
- Sampling: Euler with K steps; ablate K ∈ {1, 2, 4, 10, 20}. The encoder runs once per rollout step and the denoiser runs K times.

### Tests

| Test | Checks |
|---|---|
| test_wigner | D is orthogonal; D(R)Y(r) = Y(Rr); edge rotation maps r onto z |
| test_equivariance | random rotation, reflection, and translation; outputs transform as vectors. float64 to 1e-10, float32 to 1e-5 relative. Head B uses fixed, identically rotated noise. |
| test_permutation | permuting atoms permutes outputs |
| test_pbc | shifting one atom by a box vector changes nothing; rotating atoms and box together is equivariant |
| test_neighbors | neighbor list matches brute force (cubic box, triclinic box, small box) |
| test_causal | changing a future frame does not change past outputs |
| test_com | no center-of-mass velocity after a rollout step |
| test_dataset | windows never cross trajectories; stride indexing is correct |
| test_overfit | the model fits 16 frames to near-zero loss |

Done when: all tests pass, and on Argon the one-step error beats the constant-velocity guess x + vΔt.

## Stage 2: training against error accumulation

Strategies are added one at a time. Each one is measured by rollout stability. Main ablation: Li⁺/EC at n=10. Argon runs the full grid.

| ID | Strategy |
|---|---|
| S1 | one-step teacher forcing |
| S2 | + Gaussian noise on context frames, σ ∈ {0.5, 1, 2, 4}% of the displacement std |
| S3a | + unrolled training, truncated backprop (no gradient through earlier steps) |
| S3b | + unrolled training, full backprop with gradient checkpointing; curriculum m = 1 → 4 → 16 |
| S4 | SHAKE/RATTLE projection at inference, reported both on and off |

- For head B, unrolled steps use few-step samples (K=1–2) as inputs.
- Models are selected by stability and RDF error over 1000-step validation rollouts, not by one-step loss.
- Stride sweep: n ∈ {1, 5, 10, 50} with the best recipe, both heads. This gives the stability-vs-n plot.
- Extra ablations: velocity input vs positions only; l_max ∈ {1, 2, 3}; k ∈ {1, 3, 6}.
- Li⁺/EC runs: about 17 runs × 6–12 GPU·h ≈ 100–200 GPU·h.

## Stage 3: evaluation (the model replaces the integrator)

- Rollout: ≥10^4 steps from each test starting condition. No OpenMM integration. Rollouts are batched.
- Crash criteria:
  - Hard: a bond deviates more than 50%; any pair is closer than 0.5 Å (0.7σ for Ar); a NaN appears; temperature exceeds 3 × T₀.
  - Soft: windowed RDF L1 exceeds 3× the ground-truth spread; temperature drifts more than 10%; bond RMS deviation exceeds 10%.
  - Report the distribution of steps before a crash (Kaplan–Meier curve).
- Structure: g(r) per atom pair; first Li–O peak position, height, CN; RDF L1 distance with ground-truth error bars.
- Dynamics: VACF from model velocities and from finite differences; VDOS against ground truth subsampled to the same Δt; MSD and diffusion.
- Solvation: CN(t), and residence time from the ground-truth long runs and from rollouts (if any exchange happens).
- Energy: the reference force field computes potential energy for generated frames, and model velocities give kinetic energy. Compare the total-energy drift with the ground-truth NVE drift. This is an external check; the model has no Hamiltonian. Constraint violations are reported separately, because constrained bond terms are not in the potential energy.
- Vacuum: drift in total linear and angular momentum, with no correction applied.
- Speed: wall time to generate 1 ns with the model vs OpenMM, on the same GPU with the same batching. Classical OpenMM is already fast; the real payoff would be replacing AIMD.
- Baselines:
  - non-equivariant transformer with the same parameter count and rotation augmentation;
  - MACE trained on forces, run with the same integrator and constraints via openmm-torch;
  - head A vs head B.
- Diagnosis: per time window, find what goes wrong first: bond lengths, angles, temperature, intermolecular RDF, or the Li⁺ shell.
- Output: `results/.../metrics.json` and figures; `RESULTS.md` is generated by a script.

## Stage 4: profiling (no custom CUDA)

- PyTorch profiler and Nsight: time per step, split into neighbor list, Wigner rotation, SO(2) linear, attention, and denoiser × K.
- Try torch.compile, bf16 (keep l ≥ 1 in fp32 and check rollout stability), fewer sampling steps, and cuEquivariance where it applies.
- Report where a custom kernel would help and a rough upper bound on the gain. Do not implement it.

## Compute estimate (1× A100; H100 is ~0.6×)

| Part | GPU·h |
|---|---|
| Stage 0 MD | 5–10 |
| Stage 1 tests and smoke runs | ≤5 |
| Argon, full pipeline | 15–25 |
| Ala2 | 20–40 |
| Li⁺/EC ablations and stride sweep | 100–200 |
| Baselines and all evaluation rollouts | 40–80 |
| Profiling | ≤5 |
| **Total** | **~200–400** |

Disk: ~60 GB data plus ~20 GB checkpoints and rollouts. The table will be updated with measured numbers once Argon is done.

## Checkpoints (stop and report at each)

1. R0: environment and Argon data.
2. R1: model and tests pass; one-step accuracy on Argon.
3. R2: full pipeline on Argon; updated compute estimate. Main go/no-go point.
4. R3: ala2 results, including momentum and angular momentum.
5. R4: Li⁺/EC data and ground-truth analysis.
6. R5: Li⁺/EC ablations and stride sweep.
7. R6: Li⁺/EC final evaluation, baselines, verdict on the success criteria.
8. R7: profiling report; RESULTS.md final.

Each report covers what was done, numbers, plots, problems, and next steps. Missed targets are reported as missed, with a diagnosis.

## Risks

| Risk | Mitigation |
|---|---|
| Parity trick in SO(2) conv fails | fall back to e3nn full tensor product |
| Generative head still blows up | geometry-aware denoiser, unrolled training, S4 (reported separately) |
| Regression head blurs structure at large n | expected; measure it through RDF peak height |
| Too few Li⁺ statistics | bootstrap error bars, several starting conditions |
| No exchange events inside a rollout | use n=50 (1 ns rollouts); report honestly |
| OpenFF charges for EC differ from the literature | the goal is to reproduce the reference force field, not experiment |
| No local CUDA | see Q1 |

## Questions for you

1. Hardware: is there a CUDA machine (GPU model, count, CUDA version)? What is the GPU·h limit N per job without asking (suggest 10)? What is the total budget?
2. Order: OK to run the full pipeline on Argon first, then ala2, then Li⁺/EC?
3. Constraints: HBonds with 2 fs (recommended), or no constraints with 0.5–1 fs?
4. System 0b: alanine dipeptide with ff14SB (recommended), or aspirin with GAFF/OpenFF?
5. Li⁺/EC: charge scaling 0.8 or full? Neutralizing background or a PF₆⁻ counterion? 313 K OK?
6. Success criteria: OK to compare VDOS against subsampled ground truth, and to judge RDF/CN errors against the ground truth's own spread?
7. Stride: one model per n (recommended), or one model conditioned on n?
8. Tools: TensorBoard or wandb? Should I `git init` and commit once per stage? Package name `equitraj` OK?
9. Neighbor list: for ≤1000 atoms, an O(N²) minimum-image search on the GPU is simplest and fast enough. OK to start with that and add a cell list only if profiling shows it matters? (The original spec asks for a cell list.)
10. Prior work: should I write a short comparison with TrajCast and FlashMD before R1?
