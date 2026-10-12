# WignerFlow v0: argon

v0 answers one question on the simplest system: does the full pipeline run end to end and give numbers we can trust? It covers ground-truth MD → equivariant model → training → 10⁴-step rollouts with no integrator → evaluation.

## What is in v0

- Ground truth (OpenMM, NVE, velocity Verlet):
  - Lennard-Jones argon: 256 atoms, 10 × 0.5 ns;
  - alanine dipeptide in vacuum: 10 × 2 ns;
  - aspirin in vacuum: 10 × 1 ns.
- Model:
  - per-frame spatial encoder: eSCN SO(2) graph attention, O(3)-equivariant (rotations and reflections), l_max = 2;
  - causal temporal attention over the last k = 3 frames;
  - flow-matching head (B) and regression head (A);
  - targets are residuals relative to constant-velocity motion.
- Training: teacher forcing, input noise, and truncated unrolled training with the unroll horizon capped at 200 fs.
- Evaluation (`scripts/evaluate.py`): 3 test starts × 5 seeds × 10⁴ steps (0.5 ns at a 50 fs stride). It reports stability, temperature, RDF, VACF/VDOS at the same stride, MSD/diffusion, short-range error, and speed.
- Tests: symmetry (float64 < 1e-10), neighbor lists, integrator, analysis, thermostat.

## v0 result (argon, stride 50 fs = 10 MD steps, model only)

Flow-matching head, checkpoint picked by validation rollouts (step 10000 of 40000):

| Criterion | Target | Result | |
|---|---|---|---|
| No crash in 10⁴ steps | 15/15 | 15/15 | pass |
| Temperature | within 5% | 89.2 K vs 94.9 K (−6.0%) | **fail** |
| RDF first peak position | < 0.05 Å | 0.025 Å | pass |
| RDF first peak height | < 10% | 0.1% | pass |
| VACF first zero crossing | < 10% | 0.6% (306 vs 307 fs) | pass |
| Diffusion coefficient | < 20% | −11% | pass |

The temperature is stable over 0.5 ns (no drift), but it settles about 6% below the ground truth. The lower diffusion coefficient follows from the lower temperature.

With the CSVR thermostat switch on (500 fs, reported separately, not model only): T error 0.09%, D error −2.8%.

This is one training seed. Multi-seed runs, the regression (A) head, and the direct-vs-residual ablation were still running when v0 was committed; their numbers go into RESULTS.md as they finish.

Run the checkpoint: `python scripts/demo.py --ckpt checkpoints/argon_flow_v0.pt` (see README, "Quickstart").

## What we learned on the way

1. **Direct velocity targets cool the system.** Predicting v_new directly made an imperfect network shrink velocities by ~2.5% per step, so rollouts froze to ~3 K within 200 steps. Predicting v_new − v and Δx − vΔt removed the systematic cooling.
2. **Unrolled training has a horizon limit in a chaotic liquid.** Unrolling 16 steps × 50 fs (0.8 ps) diverged (loss ~1e5) because the model trajectory and the ground truth decorrelate within ~1 ps. Capping the unroll at 200 fs fixed this.
3. **Validation rollouts swing a lot between checkpoints.** Across checkpoints, 1000-step validation rollouts ranged from ~0.6 K to ~250 K. The model has no energy conservation built in, so which checkpoint is chosen matters. Selecting by validation rollout, not by loss, is essential.
4. **Speed.** On a V100 a rollout runs at ~7 ns/day (111 ns/day for 15 rollouts in parallel). Classical OpenMM runs argon at 2000 ns/day on a laptop CPU. This PoC tests the method; any speed benefit would come from replacing ab initio MD, not classical MD.

## Not in v0

- Li⁺/EC (ground-truth jobs submitted);
- the stride sweep;
- the 26-run ablation grid;
- baselines (non-equivariant transformer, MACE + Verlet, TrajCast);
- profiling (Stage 4).

## Reproduce

See README, "How to run". The numbers above come from `results/argon/B_S3a_seed1/eval_step10000/metrics.json`.
