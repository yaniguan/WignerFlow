# Review round 1 notes

An external code review raised specific questions about the v0 release and evaluation pipeline. Every point that can be checked in code was verified and addressed on `main`.

## Confirmed in the code and updated on `main`

| Point | Check | Update on `main` |
|---|---|---|
| The released v0 checkpoint (seed 1, step 10,000) and the seed 4 checkpoint (step 10,000) saw no multi-step unrolled training | `unroll_length` returns `m = 1` for steps 0–9,999 when the schedule switches at step 10,000 | Recorded as an erratum below |
| Taking softmax over neighbors and multiplying by the cutoff envelope afterwards causes a step jump when an atom crosses `r_c` | `tests/test_review_fixes.py` and `results/bench/cutoff_continuity.json` | `GraphAttention` on `main` places `log(envelope)` inside softmax with a zero-logit sink (`attention_version = 'v1'`, #52); legacy v0 checkpoints emit a one-time warning |
| Molecule temperature calls ignored rigid bond constraints (`ala2`: 12, `aspirin`: 8) | `temperature()` defaulted to `0` constraints | `n_constraints` is read from the HDF5 attributes and used when `train.constrained_dof: true` (#78, opt-in so v0 argon behavior stays identical) |
| The last validation rollout start had only 1 frame of ground truth after it | `validation_rollout` spaced start indices across all windows | `validation_starts()` (#78) selects only windows followed by `n_steps` of ground truth |
| `use_velocity=False` leaked the true velocity through the residual output path (`vel[:, -1]`) | `loss()` and `predict_with_grad()` used `vel[:, -1]` | `EquiTraj.base_velocity()` (#78) estimates the coasting velocity from positions when `use_velocity=False` |
| No reference-energy check on generated argon trajectories | `scripts/evaluate.py` did not compute potential + kinetic energy | `equitraj/energy.py` (switched LJ potential matching OpenMM) and an `energy` block in `scripts/evaluate.py` |

## Additional diagnostics

- `scripts/kinetic_diagnostics.py` splits one-step kinetic energy changes into cross, squared, and flow-variance terms across multiple samples from the same context.
- `scripts/bench_api.py` and `results/cpu_api/` measure how rollout temperature and RDF accuracy vary with the number of Euler flow steps (`S = 1, 2, 4, 10`).

## Erratum for v0.1.0

The released `argon_flow_v0.pt` checkpoint stopped at step 10,000, while the `B_S3a` unroll schedule switches from `m = 1` to `m = 2` at step 10,000. Thus that specific checkpoint was trained with teacher forcing and input noise only, not multi-step unrolling. Comparing teacher-forcing-only runs (`B_S1`, `B_S2`) against unrolled runs (`B_S3a` beyond step 10,000) isolates the effect of unrolled training.

## Open items for future work

- Full observation-completeness matrix (positions + velocities vs positions only, deterministic vs flow, `k = 1` vs `k = 3`).
- Calibrated-noise control for the regression head against the flow head.
- Unroll horizon scaled by the system's Lyapunov time rather than a fixed 200 fs window.
- Green–Kubo diffusion alongside MSD diffusion, plus time-reversal checks.
- External force-field and trajectory-generator baselines (MACE + Verlet, TrajCast, FlashMD) on equal hardware.
