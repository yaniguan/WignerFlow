"""Autoregressive rollout: the model replaces the integrator. No OpenMM here.

Positions are accumulated in float64 (x_new = x_last + dx) so rounding does not build up
over 10^4+ steps; the model itself runs in float32.
"""

import time

import torch

from equitraj.neighbors import minimum_image
from equitraj.pair_masks import exclusion_mask
from equitraj.thermostat import csvr
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB


def temperature(vel, masses, n_constraints=0):
    """Instantaneous temperature (K) per system. vel: (B, N, 3) Å/fs."""
    ke = 0.5 * torch.einsum("n,bnd->b", masses.double(), vel.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    dof = 3 * vel.shape[1] - n_constraints - 3
    return 2 * ke / (dof * KB)


def min_pair_distance(pos, box, excluded_pairs=None):
    """Smallest interatomic distance per system (minimum image). pos: (B, N, 3), box: (B, 3, 3) or None."""
    dr = pos[:, :, None, :] - pos[:, None, :, :]
    if box is not None:
        dr = minimum_image(dr, box[:, None].to(dr.dtype))
    d = torch.linalg.norm(dr, dim=-1)
    n = pos.shape[1]
    mask = exclusion_mask(excluded_pairs, n, d.device) | torch.eye(n, dtype=torch.bool, device=d.device)
    d = d.masked_fill(mask, float("inf"))
    return d.flatten(1).min(dim=1).values


def max_bond_deviation(pos, bonds, r0):
    """Largest relative bond-length deviation |r - r0| / r0 per system. bonds: (n_bonds, 2)."""
    r = torch.linalg.norm(pos[:, bonds[:, 0]] - pos[:, bonds[:, 1]], dim=-1)
    return ((r - r0) / r0).abs().max(dim=1).values


@torch.no_grad()
def rollout(model, pos, vel, types, box, masses, n_steps, sampling_steps=10, save_every=1,
            min_dist=0.5, t_max=None, bonds=None, bond_r0=None, max_bond_dev=0.5, check_every=10,
            thermostat=None, energy_fn=None, history=20, monitor=None, predictor=None,
            constraint_projector=None, excluded_pairs=None, *, generator=None, return_state=False,
            step_offset=0):
    """Roll out B systems in parallel.

    pos, vel: (B, k, N, 3) context (Å, Å/fs). box: (B, 3, 3) or None. Returns a dict with saved frames (n_saved, B, N, 3),
    crash_step (B,) = first step at which a crash criterion fired (-1 if never), timing, and
    'diagnostics' (StabilityMonitor.diagnostics(): crash reasons, offending atoms, T/KE drift, ...).
    Crash criteria: NaN, a pair closer than min_dist Å, temperature above t_max, a bond off by more
    than max_bond_dev (relative). Crashed systems keep being integrated, but their frames after the
    crash should be ignored.
    thermostat: None (model only) or dict(t_target, tau_fs, dt_fs) for the CSVR switch S6.
    energy_fn: optional callable(pos (B, N, 3), box) -> (B,) potential energy in kJ/mol, tracked by the monitor.
    history: number of checks kept in crash reports. monitor: a custom StabilityMonitor (then the
    crash-criterion arguments above are ignored).
    predictor: optional object with predict(); defaults to the model. Off unless passed in.
    constraint_projector: optional DistanceProjector. Off by default. Applied after the thermostat.
    excluded_pairs: optional symmetric boolean (N, N) mask. True skips that pair. Off by default.
    generator: optional torch.Generator for deterministic flow noise and CSVR thermostat sampling.
    return_state: when True, include a resumable 'state' dict (float64 context positions, float32
        context velocities, global step count, and RNG state if generator is given).
    step_offset: starting global step index before this call (0 for fresh rollouts).
    """
    from equitraj.monitor import StabilityMonitor  # monitor.py imports this module

    if min(n_steps, save_every, check_every, sampling_steps) < 1 or step_offset < 0:
        raise ValueError("step counts and intervals must be positive and step_offset >= 0")
    if predictor is None:
        predictor = model
    base_model = getattr(predictor, "model", predictor)
    device = next(model.parameters()).device
    b = pos.shape[0]
    x = pos.to(device, torch.float64)
    v = vel.to(device, torch.float32)
    types, masses = types.to(device), masses.to(device)
    box = None if box is None else box.to(device)
    if monitor is None:
        monitor = StabilityMonitor(types, masses, box=box, min_dist=min_dist, t_max=t_max, bonds=bonds,
                                   bond_r0=bond_r0, max_bond_dev=max_bond_dev, energy_fn=energy_fn,
                                   history=history, stride_fs=getattr(model, "stride_fs", 1.0),
                                   excluded_pairs=excluded_pairs)
    projection_diagnostics = []
    n_constraints = 0 if constraint_projector is None else constraint_projector.n_constraints
    if constraint_projector is not None:
        monitor.n_constraints = n_constraints
    crash = torch.full((b,), -1, dtype=torch.long, device=device)
    saved_x, saved_v = [], []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        global_step = step_offset + step
        if (generator is not None and getattr(base_model, "head_type", None) == "flow"
                and hasattr(base_model, "com_free_noise")):
            noise = base_model.com_free_noise(b, types, masses, generator=generator, like=v)
            dx, v_new = predictor.predict(x.float(), v, types, box, masses, noise=noise, n_steps=sampling_steps)
        else:
            dx, v_new = predictor.predict(x.float(), v, types, box, masses, n_steps=sampling_steps)
        if thermostat is not None:
            v_new = csvr(v_new, masses, thermostat["t_target"], thermostat["tau_fs"], thermostat["dt_fs"],
                         dof=3 * v_new.shape[1] - 3 - n_constraints, generator=generator)
        x_new = x[:, -1] + dx.double()
        if constraint_projector is not None:
            x_new, v_new, stats = constraint_projector(x_new, v_new, masses, box)
            projection_diagnostics.append({"step": global_step, **stats})
        x = torch.cat([x[:, 1:], x_new[:, None]], dim=1)
        v = torch.cat([v[:, 1:], v_new[:, None]], dim=1)
        if global_step % save_every == 0:
            saved_x.append(x_new.float().cpu())
            saved_v.append(v_new.cpu())
        if global_step % check_every == 0 or step == n_steps:
            bad = monitor.check(global_step, x_new, v_new)
            crash = torch.where((crash < 0) & bad, torch.full_like(crash, global_step), crash)
            if (crash >= 0).all():
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    diagnostics = monitor.diagnostics()
    if constraint_projector is not None:
        diagnostics["constraint_projection"] = projection_diagnostics
    out = {
        "positions": torch.stack(saved_x) if saved_x else None,
        "velocities": torch.stack(saved_v) if saved_v else None,
        "crash_step": crash.cpu(),
        "steps_done": step,
        "wall_s": time.time() - t0,
        "diagnostics": diagnostics,
    }
    if return_state:
        out["state"] = {
            "context_positions": x.detach().cpu(),
            "context_velocities": v.detach().cpu(),
            "box": None if box is None else box.detach().cpu(),
            "types": types.detach().cpu(),
            "masses": masses.detach().cpu(),
            "steps_done": step_offset + step,
            "rng_state": None if generator is None else generator.get_state().clone().cpu(),
        }
    return out
