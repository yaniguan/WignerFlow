"""Short-range repulsion prior: a physics guard rail applied after the network's prediction.

The learned model never saw two argon atoms closer than ~3 Å, so once a rollout drifts into an
overlap its predictions are extrapolations and the run crashes (argon retrain: ~865 steps). The prior
keeps rollouts inside the region covered by the training data:

    r_min[a, b]   low percentile of nearest-neighbor distances between types a and b in the
                  training data (`estimate_rmin`), stored as a buffer so it travels with checkpoints
    r0 = rmin_scale * r_min

For every pair closer than r0 the proposed new positions x' = x + Δx are pushed apart along the pair
axis by strength * s(r0 - r), where s(o) = o^2 / (o + width) is a soft wall (C1 at o = 0, ~o for deep
overlaps; width = 0 gives the hard projection s(o) = o). The push is split by inverse mass
(atom i moves m_j / (m_i + m_j) of it), i.e. a mass-weighted gradient step on the soft-wall energy
0.5 * sum (r0 - r)_+^2. Hence:
    - it depends only on relative vectors: translation, rotation and reflection equivariant
    - the mass-weighted sum of the correction is exactly zero: the center of mass does not move
    - pairs with r >= r0 are untouched, so the correction is exactly zero for well-separated systems
    - it is differentiable (used inside unrolled training)
Optionally (damp_velocity > 0) the approaching component of the relative velocity of overlapping
pairs is removed, again momentum conserving. This is inelastic and removes kinetic energy.

Units: Å, Å/fs, amu.
"""

import numpy as np
import torch
import torch.nn as nn

from equitraj.neighbors import minimum_image


def close_pairs(pos, box, types, r0):
    """Pairs i < j closer than r0[type_i, type_j] (minimum image).

    pos: (B, N, 3). box: (B, 3, 3) or None. types: (N,). r0: (T, T) in Å (0 disables a type pair).
    Returns bi, i, j (E,) and vec = pos[bi, i] - pos[bi, j] (E, 3), differentiable w.r.t. pos.
    """
    n = pos.shape[1]
    with torch.no_grad():
        dr = pos[:, :, None, :] - pos[:, None, :, :]
        if box is not None:
            dr = minimum_image(dr, box[:, None].to(dr.dtype))
        d = torch.linalg.norm(dr, dim=-1)
        cut = r0.to(d.dtype)[types[:, None], types[None, :]]
        upper = torch.ones(n, n, dtype=torch.bool, device=pos.device).triu(1)
        bi, i, j = torch.nonzero((d < cut) & upper, as_tuple=True)
    vec = pos[bi, i] - pos[bi, j]
    if box is not None and bi.numel() > 0:
        bx = box[bi].to(vec.dtype)
        with torch.no_grad():
            shift = torch.round(torch.einsum("ed,edk->ek", vec, torch.linalg.inv(bx)))
        vec = vec - torch.einsum("ek,ekd->ed", shift, bx)
    return bi, i, j, vec


def _length(vec):
    return torch.sqrt((vec * vec).sum(-1) + 1e-12)  # finite gradient even for coincident atoms


def overlap_penalty(pos, box, types, r0):
    """Mean over systems of sum_pairs ((r0 - r)_+ / r0)^2. Differentiable; 0 if nothing overlaps."""
    bi, i, j, vec = close_pairs(pos, box, types, r0)
    if bi.numel() == 0:
        return pos.new_zeros(())
    rr = r0.to(vec.dtype)[types[i], types[j]]
    o = ((rr - _length(vec)) / rr).clamp_min(0)
    return pos.new_zeros(pos.shape[0]).index_add(0, bi, o * o).mean()


@torch.no_grad()
def estimate_rmin(ds, percentile=0.1, n_frames=200, seed=0):
    """Per-type-pair r_min (T, T) in Å from training data.

    For sampled frames, the nearest-neighbor distance of every atom of type a to atoms of type b is
    collected (both directions for a != b); r_min[a, b] is the given percentile (in %, 0 = minimum)
    of those. Type pairs that never occur get 0 (prior inactive for them).
    ds needs: pos (list of (frames, N, 3) arrays, Å), boxes, periodic, types (N,), elements.
    """
    rng = np.random.default_rng(seed)
    n_types = len(ds.elements)
    types = torch.as_tensor(ds.types)
    per_traj = max(1, n_frames // len(ds.pos))
    vals = {(a, b): [] for a in range(n_types) for b in range(a, n_types)}
    for t in range(len(ds.pos)):
        for f in rng.choice(len(ds.pos[t]), size=min(per_traj, len(ds.pos[t])), replace=False):
            x = torch.as_tensor(np.array(ds.pos[t][f]), dtype=torch.float64)
            dr = x[:, None, :] - x[None, :, :]
            if ds.periodic:
                dr = minimum_image(dr, torch.as_tensor(ds.boxes[t], dtype=torch.float64))
            d = torch.linalg.norm(dr, dim=-1)
            d.fill_diagonal_(float("inf"))
            for a in range(n_types):
                for b in range(n_types):
                    sub = d[types == a][:, types == b]
                    if sub.numel() == 0:
                        continue
                    nn_d = sub.min(dim=1).values
                    vals[(min(a, b), max(a, b))].append(nn_d[torch.isfinite(nn_d)])
    rmin = torch.zeros(n_types, n_types)
    for (a, b), chunks in vals.items():
        v = torch.cat(chunks).numpy() if chunks else np.zeros(0)
        if v.size:
            rmin[a, b] = rmin[b, a] = float(np.percentile(v, percentile))
    return rmin


class RepulsionPrior(nn.Module):
    """Soft-wall correction of predicted displacements (and optionally velocities). See module docstring.

    strength:      fraction of the soft-wall push applied per iteration (1 = full projection for an isolated pair)
    rmin_scale:    r0 = rmin_scale * r_min
    width:         soft-wall width in Å (0 = hard projection)
    n_iter:        correction iterations (several help in dense overlaps)
    damp_velocity: fraction of the approaching relative velocity removed for overlapping pairs (0 = off)
    percentile:    used by train.py when estimating r_min from data
    rmin:          optional fixed r_min (scalar or (T, T)); otherwise estimated by train.py
    """

    def __init__(self, n_types, strength=0.5, rmin_scale=1.0, width=0.1, n_iter=1, damp_velocity=0.0,
                 percentile=0.1, rmin=None):
        super().__init__()
        self.strength, self.rmin_scale, self.width = float(strength), float(rmin_scale), float(width)
        self.n_iter, self.damp_velocity, self.percentile = int(n_iter), float(damp_velocity), float(percentile)
        self.rmin_fixed = rmin is not None
        r = torch.zeros(n_types, n_types)
        if rmin is not None:
            r = r + torch.as_tensor(rmin, dtype=torch.float32)
        self.register_buffer("rmin", r)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # checkpoints trained without the prior have no r_min: keep the current value (set it afterwards)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if prefix + "rmin" in missing_keys:
            missing_keys.remove(prefix + "rmin")

    def r0(self):
        return self.rmin * self.rmin_scale

    def correction(self, pos, box, types, masses):
        """Displacement (B, N, 3) that pushes overlapping pairs of `pos` apart; zero mass-weighted sum."""
        r0 = self.r0()
        bi, i, j, vec = close_pairs(pos, box, types, r0)
        if bi.numel() == 0:
            return torch.zeros_like(pos)
        r = _length(vec)
        o = (r0.to(r.dtype)[types[i], types[j]] - r).clamp_min(0)
        push = self.strength * (o if self.width <= 0 else o * o / (o + self.width))
        unit = vec / r[:, None]
        mi, mj = masses[i].to(r.dtype), masses[j].to(r.dtype)
        n = pos.shape[1]
        flat = torch.zeros_like(pos).reshape(-1, 3)
        flat = flat.index_add(0, bi * n + i, (push * mj / (mi + mj))[:, None] * unit)
        flat = flat.index_add(0, bi * n + j, -(push * mi / (mi + mj))[:, None] * unit)
        return flat.view_as(pos)

    def damp(self, pos, vel, box, types, masses):
        """Remove the fraction damp_velocity of the approaching relative velocity of overlapping pairs."""
        bi, i, j, vec = close_pairs(pos, box, types, self.r0())
        if bi.numel() == 0:
            return vel
        unit = vec / _length(vec)[:, None]
        approach = -((vel[bi, i] - vel[bi, j]) * unit).sum(-1).clamp_max(0) * self.damp_velocity
        mi, mj = masses[i].to(vel.dtype), masses[j].to(vel.dtype)
        n = vel.shape[1]
        flat = vel.reshape(-1, 3)
        flat = flat.index_add(0, bi * n + i, (approach * mj / (mi + mj))[:, None] * unit)
        flat = flat.index_add(0, bi * n + j, -(approach * mi / (mi + mj))[:, None] * unit)
        return flat.view_as(vel)

    def forward(self, pos_last, dx, vel, types, box, masses):
        """pos_last, dx, vel: (B, N, 3). Returns the corrected (dx, vel)."""
        corr = torch.zeros_like(dx)
        for _ in range(self.n_iter):
            corr = corr + self.correction(pos_last + dx + corr, box, types, masses)
        dx = dx + corr
        if self.damp_velocity > 0:
            vel = self.damp(pos_last + dx, vel, box, types, masses)
        return dx, vel
