"""Autoregressive rollout: the model replaces the integrator. No OpenMM here.

Positions are accumulated in float64 (x_new = x_last + dx) so rounding does not build up
over 10^4+ steps; the model itself runs in float32.
"""

import time

import torch

from equitraj.neighbors import minimum_image
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB


def temperature(vel, masses, n_constraints=0):
    """Instantaneous temperature (K) per system. vel: (B, N, 3) Å/fs."""
    ke = 0.5 * torch.einsum("n,bnd->b", masses.double(), vel.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    dof = 3 * vel.shape[1] - n_constraints - 3
    return 2 * ke / (dof * KB)


def min_pair_distance(pos, box):
    """Smallest interatomic distance per system (minimum image). pos: (B, N, 3), box: (B, 3, 3) or None."""
    dr = pos[:, :, None, :] - pos[:, None, :, :]
    if box is not None:
        dr = minimum_image(dr, box[:, None].to(dr.dtype))
    d = torch.linalg.norm(dr, dim=-1)
    n = pos.shape[1]
    d = d + torch.eye(n, dtype=d.dtype, device=d.device) * 1e9
    return d.flatten(1).min(dim=1).values


def max_bond_deviation(pos, bonds, r0):
    """Largest relative bond-length deviation |r - r0| / r0 per system. bonds: (n_bonds, 2)."""
    r = torch.linalg.norm(pos[:, bonds[:, 0]] - pos[:, bonds[:, 1]], dim=-1)
    return ((r - r0) / r0).abs().max(dim=1).values


@torch.no_grad()
def rollout(model, pos, vel, types, box, masses, n_steps, sampling_steps=10, save_every=1,
            min_dist=0.5, t_max=None, bonds=None, bond_r0=None, max_bond_dev=0.5, check_every=10):
    """Roll out B systems in parallel.

    pos, vel: (B, k, N, 3) context (Å, Å/fs). box: (B, 3, 3) or None. Returns a dict with saved frames (n_saved, B, N, 3),
    crash_step (B,) = first step at which a crash criterion fired (-1 if never), and timing.
    Crash criteria: NaN, a pair closer than min_dist Å, temperature above t_max, a bond off by more
    than max_bond_dev (relative). Crashed systems keep being integrated, but their frames after the
    crash should be ignored.
    """
    device = next(model.parameters()).device
    b = pos.shape[0]
    x = pos.to(device, torch.float64)
    v = vel.to(device, torch.float32)
    types, masses = types.to(device), masses.to(device)
    box = None if box is None else box.to(device)
    crash = torch.full((b,), -1, dtype=torch.long, device=device)
    saved_x, saved_v = [], []
    t0 = time.time()
    for step in range(1, n_steps + 1):
        dx, v_new = model.predict(x.float(), v, types, box, masses, n_steps=sampling_steps)
        x_new = x[:, -1] + dx.double()
        x = torch.cat([x[:, 1:], x_new[:, None]], dim=1)
        v = torch.cat([v[:, 1:], v_new[:, None]], dim=1)
        if step % save_every == 0:
            saved_x.append(x_new.float().cpu())
            saved_v.append(v_new.cpu())
        if step % check_every == 0 or step == n_steps:
            bad = ~torch.isfinite(x_new).all(dim=(1, 2)) | ~torch.isfinite(v_new).all(dim=(1, 2))
            bad |= min_pair_distance(x_new, box) < min_dist
            if t_max is not None:
                bad |= temperature(v_new, masses) > t_max
            if bonds is not None:
                bad |= max_bond_deviation(x_new, bonds.to(device), bond_r0.to(device)) > max_bond_dev
            crash = torch.where((crash < 0) & bad, torch.full_like(crash, step), crash)
            if (crash >= 0).all():
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {
        "positions": torch.stack(saved_x) if saved_x else None,
        "velocities": torch.stack(saved_v) if saved_v else None,
        "crash_step": crash.cpu(),
        "steps_done": step,
        "wall_s": time.time() - t0,
    }
