"""Baseline: a learned equivariant force field plus a velocity Verlet integrator.

EquiFF has the same encoder blocks as EquiTraj (SO(2) graph attention, l_max = 2) but a different job: it predicts one scalar
energy per atom from a single frame, E = sum of atomic energies, and the forces are F = -dE/dx, so they are conservative and
equivariant by construction. It is trained on forces only. A rollout then integrates with velocity Verlet using small
substeps, i.e. the standard "machine-learned force field + integrator" pipeline.

This is the comparison that matters for a trajectory generator: one model call per stride (EquiTraj) against
`substeps` force evaluations per stride (EquiFF + Verlet).

Units: positions Å, velocities Å/fs, energies kJ/mol, forces kJ/mol/Å, masses amu.
"""

import time

import torch
import torch.nn as nn

from equitraj.layers import MLP
from equitraj.model import EquiTraj, build_graph, remove_com
from equitraj.rollout import max_bond_deviation, min_pair_distance, temperature
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL


class EquiFF(nn.Module):
    def __init__(self, n_types, lmax=2, channels=64, heads=8, n_rbf=32, cutoff=6.0, n_spatial=3, parity=True,
                 neighbor_method="brute_force", **_unused):
        super().__init__()
        # reuse the encoder pieces of EquiTraj, then drop what a force field does not need
        net = EquiTraj(n_types, lmax=lmax, channels=channels, heads=heads, n_rbf=n_rbf, cutoff=cutoff, k=1,
                       n_spatial=n_spatial, n_fusion=0, head="regression", parity=parity, neighbor_method=neighbor_method)
        for name in ["temporal", "fusion", "readout", "y_out"]:
            delattr(net, name)
        self.net = net
        self.lmax, self.cutoff, self.neighbor_method = lmax, cutoff, neighbor_method
        self.energy = MLP([channels, channels, 1])
        self.register_buffer("force_scale", torch.ones(1))  # std of the training forces, kJ/mol/Å

    def atomic_energy(self, pos, types, box):
        """pos: (B, N, 3). Returns the energy of each system (B,), in units of force_scale * Å."""
        b, n, _ = pos.shape
        graph = build_graph(pos, box, self.cutoff, self.lmax, self.neighbor_method)
        tf = types.repeat(b)
        zero = torch.zeros(b * n, 3, dtype=pos.dtype, device=pos.device)
        x = self.net.embed(tf, zero, zero, graph, b * n)
        for blk in self.net.spatial:
            x = blk(x, graph)
        return self.energy(x[:, 0, :]).view(b, n).sum(1)

    def forces(self, pos, types, box, create_graph=False):
        """F = -dE/dx in kJ/mol/Å, shape (B, N, 3)."""
        pos = pos if pos.requires_grad else pos.detach().requires_grad_(True)
        e = self.atomic_energy(pos, types, box).sum()
        (g,) = torch.autograd.grad(e, pos, create_graph=create_graph)
        return -g * self.force_scale


def verlet_rollout(model, pos, vel, types, box, masses, n_steps, substeps, dt_fs, save_every=1, min_dist=0.5, t_max=None,
                   bonds=None, bond_r0=None, max_bond_dev=0.5, check_every=10, n_constraints=0):
    """Velocity Verlet with the learned forces. One output step = `substeps` Verlet steps of dt_fs (= one stride).

    pos, vel: (B, N, 3) the current frame (not a context window). Returns the same dict as rollout.rollout so that
    scripts/evaluate.py can analyse it unchanged. Positions are accumulated in float64.
    """
    param = next(model.parameters())
    device, dtype = param.device, param.dtype
    b = pos.shape[0]
    x = pos.to(device, torch.float64)
    v = vel.to(device, torch.float64)
    types, masses = types.to(device), masses.to(device).double()
    box = None if box is None else box.to(device)
    inv_m = 1.0 / (masses * AMU_A2_PER_FS2_TO_KJ_PER_MOL)  # Å/fs^2 per (kJ/mol/Å)

    def acc(xx):
        return model.forces(xx.to(dtype), types, None if box is None else box.to(dtype)).double() * inv_m[None, :, None]

    crash = torch.full((b,), -1, dtype=torch.long, device=device)
    saved_x, saved_v = [], []
    a = acc(x)
    t0 = time.time()
    for step in range(1, n_steps + 1):
        for _ in range(substeps):
            v = v + 0.5 * dt_fs * a
            x = x + dt_fs * v
            a = acc(x)
            v = v + 0.5 * dt_fs * a
        if step % save_every == 0:
            saved_x.append(x.float().cpu())
            saved_v.append(v.float().cpu())
        if step % check_every == 0 or step == n_steps:
            bad = ~torch.isfinite(x).all(dim=(1, 2)) | ~torch.isfinite(v).all(dim=(1, 2))
            bad |= min_pair_distance(x.float(), box) < min_dist
            if t_max is not None:
                bad |= temperature(v.float(), masses.float(), n_constraints) > t_max
            if bonds is not None:
                bad |= max_bond_deviation(x.float(), bonds.to(device), bond_r0.to(device)) > max_bond_dev
            crash = torch.where((crash < 0) & bad, torch.full_like(crash, step), crash)
            if (crash >= 0).all():
                break
    if device.type == "cuda":
        torch.cuda.synchronize()
    return {"positions": torch.stack(saved_x), "velocities": torch.stack(saved_v), "crash_step": crash.cpu(),
            "steps_done": step, "wall_s": time.time() - t0}
