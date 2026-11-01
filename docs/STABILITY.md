# Rollout stability options

Motivation: the argon retrain crashes about 865 steps into validation rollouts (`val_crash_frac` 1.0 at
step 6000). The target is stable 20 ns rollouts (400,000 steps at the 50 fs stride). Everything below is
**opt-in**. A config without these keys trains, evaluates and checkpoints as before. Two additions apply
to every run: the extra `diagnostics` entry in the rollout output and the `best_stable.pt` file.
`configs/argon/B_S3a_stable.yaml` is an example that enables them.

## 1. Short-range repulsion prior (`equitraj/prior.py`)

```yaml
model:
  repulsion_prior:
    enabled: true        # false or absent: no prior, state dict unchanged
    strength: 0.5        # fraction of the wall push per iteration (1 = full projection of an isolated pair)
    rmin_scale: 1.0      # wall distance r0 = rmin_scale * r_min
    width: 0.1           # soft-wall width in Å; push = strength * o^2 / (o + width), o = r0 - r; 0 = hard
    n_iter: 1            # correction iterations (several help in dense overlaps)
    damp_velocity: 0.0   # fraction of the approaching relative velocity removed for overlapping pairs
    percentile: 0.1      # percentile (%) of training nearest-neighbor distances used as r_min
    rmin: null           # optional fixed r_min in Å (scalar or n_types x n_types); skips estimation
```

- `estimate_rmin(ds, percentile, n_frames=200)` collects nearest-neighbor distances for each type pair
  over sampled training frames, using minimum-image distances. It returns their low percentile as a
  symmetric `(n_types, n_types)` matrix. Type pairs that never occur get 0, which disables the prior
  for them. `train.py` calls it once at the start of a fresh run unless `rmin` is given.
- `r_min` is the buffer `prior.rmin`, so it is saved in checkpoints and restored by
  `evaluation.load_model`. A checkpoint trained without the prior can still be loaded into a model
  that has it. The missing buffer is tolerated, and `r_min` keeps its constructed value (0 or the
  fixed `rmin`). Set it with `model.prior.rmin.copy_(estimate_rmin(ds))` before rolling out.
- Applied in `EquiTraj.predict` / `predict_with_grad` after the center-of-mass removal. The prior is
  used in rollouts and unrolled-training contexts, but not in the teacher-forced flow-matching loss.
  Each pair closer than `r0` is pushed apart along its axis, and the push is split by inverse mass.
  This is a mass-weighted gradient step on the soft-wall energy `0.5 Σ (r0 - r)_+²`. Properties:
  - E(3) equivariant (only relative minimum-image vectors are used)
  - preserves the center of mass exactly (and momentum when `damp_velocity > 0`)
  - exactly zero when no pair is closer than `r0`
  - differentiable
- `damp_velocity` is inelastic and removes kinetic energy. Use it as a last resort and report it.

## 2. Training options (`train:`)

| key | default | meaning |
|---|---|---|
| `noise_schedule` | absent | `[[step, sigma], ...]`, linear interpolation (held constant outside). Replaces `noise_sigma` when set (noise annealing). |
| `noise_v_ratio` | 1.0 | velocity noise level relative to the position noise level |
| `noise_type_scale` | absent | per-element noise multiplier: `{Ar: 1.0}` or a list in sorted-element order |
| `unroll_loss_weights` | absent | weights per unroll step, e.g. `[0.5, 1, 2]`, truncated or padded with the last value. When set, the loss is evaluated at **every** unroll step, including truncated mode on detached contexts, and the weighted mean is used. This is more compute in truncated mode. |
| `overlap_penalty` | absent | `{weight, rmin_scale: 1.0, percentile: 0.1}`. Adds `weight * mean_systems Σ_pairs ((r0 - r)_+ / r0)²` on predicted positions. In full backprop this covers every unrolled prediction. In both modes it also covers one extra differentiable prediction from the final context. `r0 = rmin_scale * r_min`, where `r_min` is the prior's buffer if the prior is enabled, otherwise estimated from the training data. |

Input noise was already per element, scaled by `dx_std` (positions) and `v_std` (velocities).
`noise_type_scale` adds a multiplier on top of that.

## 3. Stability monitor (`equitraj/monitor.py`)

`rollout()` now builds a `StabilityMonitor` and returns `out["diagnostics"]`. Its signature, crash
criteria and `crash_step` semantics are unchanged. New optional arguments:
`energy_fn(pos, box) -> (B,)` gives the potential energy in kJ/mol, `history` (default 20) sets how
many checks crash reports keep, and `monitor` accepts a custom instance.

At every check it records each system's temperature, kinetic energy, smallest pair distance, largest
|v|, total momentum, the largest bond deviation (if bonds are given) and the potential energy (if
`energy_fn` is given). `diagnostics` contains:

- `crash_step`, `crash_reason`: per system; reasons are `nan` > `overlap` > `temperature` > `bond` in priority order
- `crashes[b]`: `None` or `{step, reason, reasons, atoms, atom_types, value, history}`, plus `pairs`
  `[i, j, d]` for overlaps and `bonds` `[i, j, rel_dev]` for bond failures. Offending atoms are those
  with non-finite values for `nan`, the pairs below `min_dist` for `overlap` (closest first), the 5
  highest-kinetic-energy atoms for `temperature`, and the broken bonds for `bond`. `history` holds the
  last `history` checks of all statistics.
- `T_drift_K_per_ns`, `KE_drift_kJ_mol_per_ns` (least-squares slopes vs time = step × stride_fs),
  `KE_rel_drift`; `PE_drift_kJ_mol_per_ns`, `E_drift_kJ_mol_per_ns` when `energy_fn` is set
- `min_pair_distance`, `max_speed`, `max_momentum` (extremes before the crash)
- `series`: all recorded statistics per check

Per-system summaries use only checks before that system's crash.

## 4. Checkpoint selection (`eval:`)

- `best.pt` is unchanged and uses `legacy_score` = `100 * crash_frac + rdf_l1`.
- `best_stable.pt` uses `selection_score`, a tuple compared lexicographically:
  (crash fraction, −mean steps before crash, RDF L1 + relative temperature error).
  Its value is logged as `stability_score` and stored as `best_stable` in `ckpt.pt` for resuming.
- `rollout_steps_select` (absent = off) runs an additional, longer validation rollout at each
  evaluation. `rollout_starts_select` sets the number of starts and defaults to `rollout_starts`. Its
  metrics are logged with the prefix `sel_`, and `selection_score` uses them when present.
- New scalar validation metrics: `val_crash_{nan,overlap,temperature,bond}` (counts),
  `val_min_pair_dist`, `val_abs_T_drift_K_per_ns`, `val_T_rel_err`.
- Validation rollouts that crash append their diagnostics, without `series`, to
  `<out_dir>/crash_diagnostics.jsonl`.

## Suggested recipe for argon

1. Enable the prior with `rmin_scale` about 0.9–0.95 and `n_iter: 2`. Retrain or fine-tune.
2. Anneal the noise from 0.03 to 0.005 and add `overlap_penalty.weight` about 0.1.
3. Set `rollout_steps_select` to several thousand steps and pick `best_stable.pt`.
4. Check `crash_diagnostics.jsonl`: `overlap` crashes point to the prior or the penalty. `temperature`
   crashes with a large `T_drift_K_per_ns` point to energy drift, so try the thermostat switch S6 or
   longer unrolls.
