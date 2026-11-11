# Resumable Chunked Rollout, Flow Uncertainty, and Hybrid Fallback

This document describes the production long-timescale rollout utilities in `equitraj/rollout.py` and `equitraj/hybrid.py`. All additions are opt-in and preserve existing `rollout()` return keys and signatures.

## 1. Bit-Exact Resumable Chunked Rollout (`chunked_rollout`)

Multi-nanosecond rollouts on preemptible or wall-time-capped GPUs require mid-rollout checkpointing without losing `float64` coordinate accumulation precision or breaking reproducibility of flow-matching noise and CSVR thermostat draws.

- `rollout(..., generator=..., return_state=True, step_offset=...)` snapshots:
  - `context_positions` in **`float64`** (`(B, k, N, 3)`), avoiding `float64 -> float32 -> float64` truncation drift across restarts,
  - `context_velocities` in `float32`,
  - `rng_state` from the dedicated `torch.Generator` driving both `com_free_noise` and `csvr`.
- `chunked_rollout` wraps `rollout` in `chunk_steps` blocks and writes atomic checkpoints (`save_rollout_state` via temp-file + `os.replace`) safe for `torch.load(..., weights_only=True)`:

```python
from equitraj.hybrid import chunked_rollout

out = chunked_rollout(
    model, pos, vel, types, box, masses,
    total_steps=20_000,
    chunk_steps=1_000,
    state_path="runs/argon_long/rollout_state.pt",
    seed=42,
    save_every=10,
    thermostat={"t_target": 94.4, "tau_fs": 500.0, "dt_fs": 50.0},
)
```

Re-running the identical call after preemption automatically resumes from `rollout_state.pt` and produces saved frames bit-for-bit identical (`torch.equal`) to an uninterrupted run, even when `chunk_steps` is not a multiple of `save_every`.

## 2. $O(3)$-Invariant Flow Ensemble Uncertainty (`predict_with_uncertainty`)

Because `EquiTraj` separates its expensive $k$-frame spatiotemporal encoder (`model.encode`) from its lightweight conditional flow denoiser (`model.denoiser`), drawing $E$ independent flow samples from the same conditioning representation adds minimal overhead while exposing local epistemic/aleatoric spread:

```python
from equitraj.hybrid import predict_with_uncertainty

dx_mean, v_mean, unc = predict_with_uncertainty(
    model, pos, vel, types, box, masses, n_ensembles=4, n_steps=10
)
# unc["dx_std"]: (B, N) per-atom O(3)-invariant RMS vector spread in Å
# unc["dx_std_max"]: (B,) worst-atom spread per system
```

For each atom $i$, $\sigma_{\Delta x, i} = \sqrt{\frac{1}{E}\sum_{e=1}^E \|\Delta\mathbf{x}_i^{(e)} - \overline{\Delta\mathbf{x}}_i\|_2^2}$ is strictly invariant under $O(3)$ rotations, reflections, and translations, while $(\overline{\Delta\mathbf{x}}, \overline{\mathbf{v}})$ remain $O(3)$-equivariant and center-of-mass-free.

## 3. Hybrid Reference-Physics Fallback (`HybridRollout` & `OpenMMStepper`)

Instead of terminating a trajectory when a rare high-energy collision or tail flow sample triggers an unphysical proposal, `HybridRollout` rejects the offending model step for any flagged trajectory $b$, rolls back to `(x[b, -1], v[b, -1])`, advances that trajectory with `fallback_stepper` for `n_fallback_strides` coarse strides (e.g. `model.k` strides so the entire $k$-frame context buffer is refilled with relaxed physics), and hands control back to WignerFlow:

- Supported per-step fallback triggers:
  - `nan`: non-finite positions or velocities,
  - `overlap`: minimum pair distance `< min_dist` (respecting `excluded_pairs` and PBC),
  - `temperature`: instantaneous kinetic temperature `> t_max`,
  - `bond`: relative bond stretch `|r - r0| / r0 > max_bond_dev`,
  - `uncertainty`: flow ensemble spread `dx_std_max > max_dx_std`,
  - `energy_jump`: potential energy jump `|U(x_cand) - U(x_prev)| > max_energy_jump`.

- `OpenMMStepper(system, dt_fs, stride_fs)` wraps `equitraj.simulate.velocity_verlet` (`CustomIntegrator`, avoiding OpenMM's leapfrog half-step velocity lag) and converts automatically between WignerFlow (`Å`, `Å/fs`) and OpenMM (`nm`, `nm/ps`) units:

```python
from equitraj.hybrid import HybridRollout, OpenMMStepper

stepper = OpenMMStepper("system.xml", dt_fs=1.0, stride_fs=model.stride_fs, platform_name="CPU")
hybrid = HybridRollout(
    model,
    stepper,
    n_fallback_strides=model.k,
    min_dist=2.6,
    t_max=500.0,
    max_dx_std=0.35,
    uncertainty_ensembles=4,
)
res = hybrid.rollout(pos, vel, types, box, masses, n_steps=10_000, save_every=10)
print(res["fallback_fraction"], res["switches"])
```
