# Argon full-data run (argon_full_oct29): diagnosis and stable fine-tune

## What the A100 run measured (Drive `WignerFlow_runs/argon_full_oct29/train/log.jsonl`, `training.log`)

Config `B_S3a_seed1` (batch 8, LR 3e-4 cosine over 40k, grad clip 1.0, truncated unroll 1 → 2 at 10k → 4 at 20k,
loss on the last unroll step only, no prior). Validation: 4 starts × 1000 steps on the val trajectory, T_ref = 95 K.

| step | m | crash frac | steps before crash | val RDF L1 | val T_end (K) |
|---:|---:|---:|---:|---:|---:|
| 2000 | 1 | 0 | 1000 | 2.520 | 0.28 |
| 4000 | 1 | 0 | 1000 | 2.399 | 0.25 |
| 6000 | 1 | 1.0 | 865 | — | — |
| 8000 | 1 | 0 | 1000 | 0.276 | 123.4 |
| 10000 | 1→2 | 0.25 | 1000 | 0.542 | 236.8 |
| 12000 | 2 | 0 | 1000 | 1.716 | 0.96 |
| 14000 | 2 | 1.0 | 317.5 | — | — |
| 16000 | 2 | 0 | 1000 | 0.283 | 134.1 |
| 18000 | 2 | 0 | 1000 | 0.707 | 34.0 |
| 20000 | 2 | 0 | 1000 | 0.433 | 54.6 |
| 22000 | 4 | 1.0 | 272.5 | — | — |

Train loss: ~0.10–0.13 at m=2; at the first logged m=4 window (step 20050) 11.8 (18.8 after the resume), up to 26.0,
settling at 0.4–0.8 (3–6× the m=2 level). The session ended at step ~22400. No final evaluation was run.

## Diagnosis

1. **Curriculum jump.** m goes 2 → 4 in one step at full LR (1.55e-4). In truncated mode the loss is taken only on the
   last step, whose context is three model predictions made with 2 flow steps each. The Δx target is relative to the
   drifted context, so its normalised size, and the loss, jump by ~100×. Clipping at 1.0 bounds the step size but
   every update is then a full-norm step along the gradient of those outlier windows. Single-step accuracy degrades
   and validation crashes 2000 steps later (272 steps before crash).
2. **Intrinsic instability even at m ≤ 2.** Crashes at steps 6000 and 14000, and T_end swinging between 0.3 K
   (frozen) and 237 K, show the model is not stable from step to step. Short 4-start validation is noisy, so the
   checkpoint choice matters.
3. **Selection.** This run used the legacy `best.pt` (100·crash + RDF) on 4 × 1000-step rollouts. That picks a
   short-horizon winner (best.pt = step 8000, RDF 0.276, T_end 123 K vs 95 K).

## Fix (`configs/argon/B_S3a_stable_ft.yaml`, `colab/argon_stable_ft.ipynb`)

- Fine-tune from the most stable m ≤ 2 snapshot (`scripts/rank_snapshots.py`: 8 starts × 2000 steps, with the
  repulsion prior on, ranked by `selection_score`). `train.init_from` loads the weights non-strictly, only for the
  prior buffer. The initial weights are evaluated at step 0, so `best_stable.pt` is never worse than the start.
- Gradual unroll curriculum 2 → 3 (3k) → 4 (8k) with `rewarmup_steps: 1000`: LR 0.1× → 1× after each switch.
- `unroll_loss_weights: [1,1,1,1]`: the loss is on every unroll step, not only the last, so the m=4 loss stays on
  the scale of the m=2 loss.
- `loss_spike_factor: 5`: updates with a non-finite loss, or a loss above 5× the running median at the current
  m, are skipped and counted (`skipped_updates` in the log). Non-finite gradient norms are skipped too.
- LR 1e-4 (warmup 300, cosine over 16k), grad clip 0.5.
- Stability options (docs/STABILITY.md): repulsion prior (0.95 r_min, 2 iterations), noise annealing
  0.02 → 0.005, overlap penalty 0.1.
- Selection: `best_stable.pt` (crash fraction, then steps survived, then RDF + |T−T_ref|/T_ref) on 8 × 2000-step
  selection rollouts.
- Final evaluation: `scripts/eval_multiseed.py` on v0 and `best_stable.pt`, test trajectory, 8 starts × 3 seeds ×
  1000 steps. Then `scripts/compare_v0.py` writes `v0_vs_v1.json` with a pass/fail on all four metrics
  (temperature_rel_err, rdf_l1, vacf_zero_rel_err, D_rel_err; lower is better) and on crash fraction.
