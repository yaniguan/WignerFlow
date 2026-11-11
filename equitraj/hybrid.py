"""Resumable chunked rollouts, flow-ensemble uncertainty, and hybrid physics fallback.

Long coarse-stride MD rollouts face two practical bottlenecks:
1. Preemptible/time-limited GPU jobs need atomic mid-rollout checkpointing that preserves
   both the float64 position accumulator and the flow/thermostat RNG stream bit-for-bit.
2. Rare out-of-distribution collisions or tail flow samples can trigger an atomic overlap,
   temperature spike, bond stretch, energy jump, or high flow-ensemble spread; handing the
   flagged trajectory back to a fine-step reference integrator (e.g. OpenMM velocity Verlet)
   for a few strides heals the state without discarding a multi-nanosecond run.
"""

import os
import tempfile
import time

import torch

from equitraj.model import remove_com
from equitraj.monitor import StabilityMonitor
from equitraj.rollout import max_bond_deviation, min_pair_distance, rollout, temperature
from equitraj.thermostat import csvr
from equitraj.units import NM_PER_PS_TO_A_PER_FS, NM_TO_A, PS_TO_FS


# ---------------------------------------------------------------- resumable chunked rollout

def save_rollout_state(state, path):
    """Atomically write a tensor-only rollout state dict to `path`."""
    path = os.fspath(path)
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".rollout_state_", suffix=".pt", dir=parent)
    os.close(fd)
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def load_rollout_state(path):
    """Load a rollout state dict saved by `save_rollout_state` (`weights_only=True`)."""
    return torch.load(os.fspath(path), map_location="cpu", weights_only=True)


@torch.no_grad()
def chunked_rollout(model, pos, vel, types, box, masses, total_steps, chunk_steps, *,
                    state_path=None, seed=0, save_every=1, on_chunk=None, **rollout_kwargs):
    """Run `total_steps` in chunks of `chunk_steps`, optionally checkpointing after every chunk.

    Because `state['context_positions']` is stored in float64 and `state['rng_state']` snapshots
    the dedicated `torch.Generator`, a multi-chunk rollout (including stop/resume from `state_path`)
    produces saved positions and velocities bit-for-bit identical (`torch.equal`) to a single
    uninterrupted `rollout(..., generator=torch.Generator(device).manual_seed(seed))`.
    """
    if min(total_steps, chunk_steps, save_every) < 1:
        raise ValueError("total_steps, chunk_steps, and save_every must be positive")
    for disallowed in ("generator", "return_state", "step_offset"):
        if disallowed in rollout_kwargs:
            raise ValueError(f"chunked_rollout manages {disallowed!r} internally")

    device = next(model.parameters()).device
    gen = torch.Generator(device=device).manual_seed(seed)

    if state_path is not None and os.path.exists(state_path):
        saved = load_rollout_state(state_path)
        cur_pos = saved["context_positions"]
        cur_vel = saved["context_velocities"]
        cur_box = saved["box"]
        types = saved["types"]
        masses = saved["masses"]
        steps_done = int(saved["steps_done"])
        resumed_from = steps_done
        if saved.get("rng_state") is not None:
            gen.set_state(saved["rng_state"])
        saved_x = [] if saved["positions"] is None else [saved["positions"]]
        saved_v = [] if saved["velocities"] is None else [saved["velocities"]]
        crash = saved["crash_step"].clone()
        wall_s = float(saved.get("wall_s", 0.0))
    else:
        cur_pos, cur_vel, cur_box = pos, vel, box
        steps_done = 0
        resumed_from = 0
        saved_x, saved_v = [], []
        crash = torch.full((pos.shape[0],), -1, dtype=torch.long)
        wall_s = 0.0

    chunks_run = 0
    last_diag = None
    while steps_done < total_steps and not (crash >= 0).all():
        n_this = min(chunk_steps, total_steps - steps_done)
        out = rollout(
            model, cur_pos, cur_vel, types, cur_box, masses, n_this,
            save_every=save_every, generator=gen, return_state=True,
            step_offset=steps_done, **rollout_kwargs,
        )
        chunks_run += 1
        wall_s += out["wall_s"]
        last_diag = out["diagnostics"]
        if out["positions"] is not None:
            saved_x.append(out["positions"])
            saved_v.append(out["velocities"])
        new_crash = out["crash_step"]
        crash = torch.where((crash < 0) & (new_crash >= 0), new_crash, crash)
        st = out["state"]
        cur_pos = st["context_positions"]
        cur_vel = st["context_velocities"]
        steps_done = int(st["steps_done"])

        full_pos = torch.cat(saved_x, dim=0) if saved_x else None
        full_vel = torch.cat(saved_v, dim=0) if saved_v else None
        ckpt_state = {
            **st,
            "positions": full_pos,
            "velocities": full_vel,
            "crash_step": crash.clone(),
            "wall_s": wall_s,
        }
        if state_path is not None:
            save_rollout_state(ckpt_state, state_path)
        if on_chunk is not None:
            on_chunk({"chunks_run": chunks_run, "steps_done": steps_done, "crash_step": crash.clone()})

    positions = torch.cat(saved_x, dim=0) if saved_x else None
    velocities = torch.cat(saved_v, dim=0) if saved_v else None
    return {
        "positions": positions,
        "velocities": velocities,
        "crash_step": crash,
        "steps_done": steps_done,
        "wall_s": wall_s,
        "diagnostics": last_diag,
        "chunks_run": chunks_run,
        "resumed_from_step": resumed_from,
        "state": {
            "context_positions": cur_pos.detach().cpu().to(torch.float64),
            "context_velocities": cur_vel.detach().cpu().to(torch.float32),
            "box": None if cur_box is None else cur_box.detach().cpu(),
            "types": types.detach().cpu(),
            "masses": masses.detach().cpu(),
            "steps_done": steps_done,
            "rng_state": gen.get_state().clone().cpu(),
        },
    }


# ---------------------------------------------------------------- flow ensemble uncertainty

@torch.no_grad()
def predict_with_uncertainty(model, pos, vel, types, box, masses, n_ensembles=4, n_steps=10,
                             generator=None, noises=None):
    """Encode the k-frame context once and integrate `n_ensembles` flow trajectories.

    Returns `(dx_mean, v_mean, unc)` where:
      - `dx_mean`, `v_mean`: `(B, N, 3)` ensemble mean displacement and velocity (COM-free).
      - `unc['dx_std']`, `unc['v_std']`: `(B, N)` O(3)-invariant RMS vector spread
            sqrt(mean_e ||u^(e)_i - u_mean_i||^2) per atom in Å and Å/fs.
      - `unc['dx_std_max']`, `unc['dx_std_mean']`, `unc['v_std_max']`: `(B,)` system summaries.
    """
    if n_ensembles < 1 or n_steps < 1:
        raise ValueError("n_ensembles and n_steps must be >= 1")
    b, _, n, _ = pos.shape
    if model.head_type == "regression":
        dx, v = model.predict(pos, vel, types, box, masses, n_steps=n_steps)
        zeros_bn = torch.zeros(b, n, dtype=dx.dtype, device=dx.device)
        zeros_b = torch.zeros(b, dtype=dx.dtype, device=dx.device)
        return dx, v, {
            "dx_std": zeros_bn, "v_std": zeros_bn,
            "dx_std_max": zeros_b, "dx_std_mean": zeros_b, "v_std_max": zeros_b,
        }

    if noises is not None and len(noises) != n_ensembles:
        raise ValueError("len(noises) must equal n_ensembles")

    cond, graph = model.encode(pos, vel, types, box)
    y_kw = {"masses": masses} if getattr(model, "target_scaling", "element_std") == "mass" else {}
    dx_list, v_list = [], []
    for e in range(n_ensembles):
        y = (model.com_free_noise(b, types, masses, generator=generator, like=cond)
             if noises is None else noises[e])
        for i in range(n_steps):
            tau = torch.full((b * n,), i / n_steps, dtype=cond.dtype, device=cond.device)
            y = y + model.denoiser(cond, graph, y, tau) / n_steps
        dx_e, v_e = model.y_to_targets(y, types, vel[:, -1], **y_kw)
        dx_e, v_e = remove_com(dx_e, masses), remove_com(v_e, masses)
        if model.prior is not None:
            dx_e, v_e = model.prior(pos[:, -1], dx_e, v_e, types, box, masses)
        dx_list.append(dx_e)
        v_list.append(v_e)

    dx_stack = torch.stack(dx_list, dim=0)  # (E, B, N, 3)
    v_stack = torch.stack(v_list, dim=0)
    dx_mean = remove_com(dx_stack.mean(dim=0), masses)
    v_mean = remove_com(v_stack.mean(dim=0), masses)
    dx_std = ((dx_stack - dx_mean[None]).pow(2).sum(dim=-1).mean(dim=0)).sqrt()
    v_std = ((v_stack - v_mean[None]).pow(2).sum(dim=-1).mean(dim=0)).sqrt()
    return dx_mean, v_mean, {
        "dx_std": dx_std,
        "v_std": v_std,
        "dx_std_max": dx_std.max(dim=1).values,
        "dx_std_mean": dx_std.mean(dim=1),
        "v_std_max": v_std.max(dim=1).values,
    }


# ---------------------------------------------------------------- hybrid fallback rollout

class OpenMMStepper:
    """Reference fallback stepper wrapping OpenMM's explicit velocity Verlet integrator.

    Advances one coarse model stride (`stride_fs`) using `round(stride_fs / dt_fs)` fine
    velocity Verlet substeps (`dt_fs`), converting cleanly between WignerFlow units (Å, Å/fs)
    and OpenMM units (nm, nm/ps) without leapfrog half-step velocity lag.
    """

    def __init__(self, system, dt_fs, stride_fs, *, platform_name="CPU", precision="mixed"):
        import openmm
        import openmm.unit as u

        from equitraj.simulate import make_context, velocity_verlet

        self._openmm = openmm
        self._u = u
        if isinstance(system, (str, os.PathLike)):
            text = open(system).read() if os.path.exists(system) else str(system)
            system = openmm.XmlSerializer.deserialize(text)
        else:
            system = openmm.XmlSerializer.clone(system)
        if dt_fs <= 0 or stride_fs <= 0:
            raise ValueError("dt_fs and stride_fs must be positive")
        n_sub = int(round(stride_fs / dt_fs))
        if n_sub < 1 or abs(n_sub * dt_fs - stride_fs) > 1e-6 * stride_fs:
            raise ValueError(f"stride_fs={stride_fs} must be a positive integer multiple of dt_fs={dt_fs}")
        self.system = system
        self.dt_fs = float(dt_fs)
        self.stride_fs = float(stride_fs)
        self.n_substeps = n_sub
        self.integrator = velocity_verlet(self.dt_fs / PS_TO_FS)
        self.context, self.precision_used = make_context(system, self.integrator, platform_name, precision)

    def step(self, pos, vel, box=None, types=None, masses=None):
        """Advance single-system `pos` (N, 3) Å and `vel` (N, 3) Å/fs by one `stride_fs`."""
        u = self._u
        device = pos.device
        vel_dtype = vel.dtype
        if box is not None and self.system.usesPeriodicBoundaryConditions():
            b_nm = box.detach().cpu().double().numpy() / NM_TO_A
            self.context.setPeriodicBoxVectors(
                b_nm[0] * u.nanometer, b_nm[1] * u.nanometer, b_nm[2] * u.nanometer
            )
        pos_nm = pos.detach().cpu().double().numpy() / NM_TO_A
        vel_nm_ps = vel.detach().cpu().double().numpy() / NM_PER_PS_TO_A_PER_FS
        self.context.setPositions(pos_nm * u.nanometer)
        self.context.setVelocities(vel_nm_ps * (u.nanometer / u.picosecond))
        if self.system.getNumConstraints() > 0:
            self.context.applyConstraints(1e-6)
            self.context.applyVelocityConstraints(1e-6)
        self.integrator.step(self.n_substeps)
        state = self.context.getState(getPositions=True, getVelocities=True)
        x_next = torch.from_numpy(
            state.getPositions(asNumpy=True).value_in_unit(u.nanometer) * NM_TO_A
        ).to(device=device, dtype=torch.float64)
        v_next = torch.from_numpy(
            state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond) * NM_PER_PS_TO_A_PER_FS
        ).to(device=device, dtype=vel_dtype)
        if masses is not None:
            v_next = remove_com(v_next[None], masses.to(device=device, dtype=vel_dtype))[0]
        return x_next, v_next


class HybridRollout:
    """Autoregressive WignerFlow rollout with automatic reference-integrator fallback.

    At each coarse step, the model proposes `(x_cand, v_cand)`. Any trajectory in the batch
    that violates a physical/uncertainty guard (NaN, pair overlap < `min_dist`, temperature >
    `t_max`, relative bond deviation > `max_bond_dev`, flow ensemble spread > `max_dx_std`,
    or potential energy jump |Delta U| > `max_energy_jump`) rejects the model proposal, rolls
    back to its pre-step state, and runs `fallback_stepper` for `n_fallback_strides` coarse
    strides (e.g. `model.k` strides to flush the rolling k-frame context with relaxed physics)
    before handing control back to the model.
    """

    def __init__(self, model, fallback_stepper, *, n_fallback_strides=1, min_dist=0.5,
                 t_max=None, bonds=None, bond_r0=None, max_bond_dev=0.5, max_dx_std=None,
                 uncertainty_ensembles=1, energy_fn=None, max_energy_jump=None,
                 excluded_pairs=None, predictor=None):
        if n_fallback_strides < 1 or uncertainty_ensembles < 1:
            raise ValueError("n_fallback_strides and uncertainty_ensembles must be >= 1")
        if max_dx_std is not None and uncertainty_ensembles < 2:
            raise ValueError("max_dx_std requires uncertainty_ensembles >= 2")
        if max_energy_jump is not None and energy_fn is None:
            raise ValueError("max_energy_jump requires energy_fn")
        self.model = model
        self.predictor = model if predictor is None else predictor
        self.fallback_stepper = fallback_stepper
        self.n_fallback_strides = int(n_fallback_strides)
        self.min_dist = min_dist
        self.t_max = t_max
        self.bonds = None if bonds is None else torch.as_tensor(bonds, dtype=torch.long)
        self.bond_r0 = None if bond_r0 is None else torch.as_tensor(bond_r0, dtype=torch.float64)
        self.max_bond_dev = max_bond_dev
        self.max_dx_std = max_dx_std
        self.uncertainty_ensembles = int(uncertainty_ensembles)
        self.energy_fn = energy_fn
        self.max_energy_jump = max_energy_jump
        self.excluded_pairs = excluded_pairs

    def _call_fallback(self, pos_b, vel_b, box_b, types, masses):
        if hasattr(self.fallback_stepper, "step"):
            return self.fallback_stepper.step(pos_b, vel_b, box_b, types, masses)
        return self.fallback_stepper(pos_b, vel_b, box_b, types, masses)

    def _detect_triggers(self, x_prev, x_cand, v_cand, box, masses, n_constraints, unc):
        b = x_cand.shape[0]
        reasons = [None] * b
        nan_mask = ~torch.isfinite(x_cand).all(dim=(1, 2)) | ~torch.isfinite(v_cand).all(dim=(1, 2))
        for idx in torch.nonzero(nan_mask, as_tuple=False).flatten().tolist():
            reasons[idx] = "nan"

        finite = ~nan_mask
        if finite.any() and self.min_dist is not None:
            dmin = min_pair_distance(x_cand, box, excluded_pairs=self.excluded_pairs)
            for idx in torch.nonzero(finite & (dmin < self.min_dist), as_tuple=False).flatten().tolist():
                if reasons[idx] is None:
                    reasons[idx] = "overlap"
        if finite.any() and self.t_max is not None:
            temp = temperature(v_cand, masses, n_constraints=n_constraints)
            for idx in torch.nonzero(finite & (temp > self.t_max), as_tuple=False).flatten().tolist():
                if reasons[idx] is None:
                    reasons[idx] = "temperature"
        if finite.any() and self.bonds is not None and self.bond_r0 is not None:
            bdev = max_bond_deviation(
                x_cand, self.bonds.to(x_cand.device), self.bond_r0.to(x_cand.device, x_cand.dtype)
            )
            for idx in torch.nonzero(finite & (bdev > self.max_bond_dev), as_tuple=False).flatten().tolist():
                if reasons[idx] is None:
                    reasons[idx] = "bond"
        if finite.any() and self.max_dx_std is not None and unc is not None:
            for idx in torch.nonzero(finite & (unc["dx_std_max"] > self.max_dx_std), as_tuple=False).flatten().tolist():
                if reasons[idx] is None:
                    reasons[idx] = "uncertainty"
        if finite.any() and self.max_energy_jump is not None and self.energy_fn is not None:
            u_prev = self.energy_fn(x_prev, box)
            u_cand = self.energy_fn(x_cand, box)
            jump = (u_cand - u_prev).abs()
            bad_e = ~torch.isfinite(u_cand) | (jump > self.max_energy_jump)
            for idx in torch.nonzero(finite & bad_e, as_tuple=False).flatten().tolist():
                if reasons[idx] is None:
                    reasons[idx] = "energy_jump"
        return reasons

    @torch.no_grad()
    def rollout(self, pos, vel, types, box, masses, n_steps, *, sampling_steps=10, save_every=1,
                check_every=10, thermostat=None, constraint_projector=None, history=20,
                generator=None):
        """Run `n_steps` coarse strides with hybrid fallback, returning frames and switch stats."""
        if min(n_steps, save_every, check_every, sampling_steps) < 1:
            raise ValueError("step counts and intervals must be positive")
        device = next(self.model.parameters()).device
        b = pos.shape[0]
        x = pos.to(device, torch.float64)
        v = vel.to(device, torch.float32)
        types, masses = types.to(device), masses.to(device)
        box = None if box is None else box.to(device)

        n_constraints = 0 if constraint_projector is None else constraint_projector.n_constraints
        monitor = StabilityMonitor(
            types, masses, box=box, min_dist=self.min_dist if self.min_dist is not None else 0.0,
            t_max=self.t_max, bonds=self.bonds, bond_r0=self.bond_r0, max_bond_dev=self.max_bond_dev,
            energy_fn=self.energy_fn, history=history, stride_fs=getattr(self.model, "stride_fs", 1.0),
            excluded_pairs=self.excluded_pairs,
        )
        monitor.n_constraints = n_constraints

        fallback_remaining = [0] * b
        switches = []
        model_traj_steps = 0
        fallback_traj_steps = 0
        wall_model_s = 0.0
        wall_fallback_s = 0.0
        crash = torch.full((b,), -1, dtype=torch.long, device=device)
        saved_x, saved_v = [], []
        t0 = time.time()

        for step in range(1, n_steps + 1):
            need_model = any(rem == 0 for rem in fallback_remaining)
            unc = None
            x_cand = x[:, -1].clone()
            v_cand = v[:, -1].clone()
            reasons = [None] * b

            if need_model:
                tm0 = time.time()
                if self.uncertainty_ensembles >= 2:
                    dx, v_pred, unc = predict_with_uncertainty(
                        self.model, x.float(), v, types, box, masses,
                        n_ensembles=self.uncertainty_ensembles, n_steps=sampling_steps,
                        generator=generator,
                    )
                elif (generator is not None and getattr(self.model, "head_type", None) == "flow"
                      and hasattr(self.model, "com_free_noise")):
                    noise = self.model.com_free_noise(b, types, masses, generator=generator, like=v)
                    dx, v_pred = self.predictor.predict(
                        x.float(), v, types, box, masses, noise=noise, n_steps=sampling_steps
                    )
                else:
                    dx, v_pred = self.predictor.predict(
                        x.float(), v, types, box, masses, n_steps=sampling_steps
                    )
                x_cand = x[:, -1] + dx.double()
                v_cand = v_pred
                wall_model_s += time.time() - tm0

                reasons = self._detect_triggers(x[:, -1], x_cand, v_cand, box, masses, n_constraints, unc)
                for idx, reason in enumerate(reasons):
                    if fallback_remaining[idx] == 0 and reason is not None:
                        fallback_remaining[idx] = self.n_fallback_strides
                        switches.append({"step": step, "system": idx, "reason": reason})

            x_new = x_cand.clone()
            v_new = v_cand.clone()
            for idx in range(b):
                if fallback_remaining[idx] > 0:
                    tf0 = time.time()
                    box_b = None if box is None else box[idx]
                    xb, vb = self._call_fallback(x[idx, -1], v[idx, -1], box_b, types, masses)
                    wall_fallback_s += time.time() - tf0
                    x_new[idx] = xb.to(device=device, dtype=torch.float64)
                    v_new[idx] = vb.to(device=device, dtype=torch.float32)
                    fallback_remaining[idx] -= 1
                    fallback_traj_steps += 1
                else:
                    model_traj_steps += 1

            if thermostat is not None:
                v_new = csvr(
                    v_new, masses, thermostat["t_target"], thermostat["tau_fs"], thermostat["dt_fs"],
                    dof=3 * v_new.shape[1] - 3 - n_constraints, generator=generator,
                )
            if constraint_projector is not None:
                x_new, v_new, _ = constraint_projector(x_new, v_new, masses, box)

            x = torch.cat([x[:, 1:], x_new[:, None]], dim=1)
            v = torch.cat([v[:, 1:], v_new[:, None]], dim=1)
            if step % save_every == 0:
                saved_x.append(x_new.float().cpu())
                saved_v.append(v_new.cpu())
            if step % check_every == 0 or step == n_steps:
                bad = monitor.check(step, x_new, v_new)
                crash = torch.where((crash < 0) & bad, torch.full_like(crash, step), crash)
                if (crash >= 0).all():
                    break

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_traj_steps = max(1, model_traj_steps + fallback_traj_steps)
        return {
            "positions": torch.stack(saved_x) if saved_x else None,
            "velocities": torch.stack(saved_v) if saved_v else None,
            "crash_step": crash.cpu(),
            "steps_done": step,
            "wall_s": time.time() - t0,
            "wall_model_s": wall_model_s,
            "wall_fallback_s": wall_fallback_s,
            "model_steps": model_traj_steps,
            "fallback_steps": fallback_traj_steps,
            "fallback_fraction": fallback_traj_steps / total_traj_steps,
            "switches": switches,
            "diagnostics": monitor.diagnostics(),
        }
