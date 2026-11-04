"""Adjust pair distances and velocities using particle masses (Å, Å/fs, amu).

Corrections keep the center of mass and total momentum fixed. With a box,
use the nearest periodic image; it must be the intended bond image.
These corrections do not replace integration using forces.
"""
import torch
from equitraj.neighbors import minimum_image


class DistanceProjector:
    """Set pair distances with (i, j, distance_A) constraints.

    Particle masses must be positive. Raise RuntimeError if corrections fail
    to reach the tolerances. Leave inputs unchanged.
    """
    def __init__(self, constraints, *, tolerance=1e-8, velocity_tolerance=1e-10,
                 max_iterations=100):
        rows = torch.as_tensor(constraints, dtype=torch.float64)
        if rows.numel() == 0:
            rows = rows.reshape(0, 3)
        if rows.ndim != 2 or rows.shape[1] != 3 or not torch.isfinite(rows).all():
            raise ValueError("constraints must be finite (C,3)")
        idx = rows[:, :2]
        if (idx < 0).any() or (idx != idx.floor()).any() or (idx[:, 0] == idx[:, 1]).any() or (rows[:, 2] <= 0).any():
            raise ValueError("invalid constraint indices or distances")
        pairs = [tuple(sorted(pair)) for pair in idx.long().tolist()]
        if len(set(pairs)) != len(pairs):
            raise ValueError("duplicate constraint pairs")
        if tolerance <= 0 or velocity_tolerance <= 0 or max_iterations < 1:
            raise ValueError("positive tolerances and iteration limit required")
        self.rows = rows
        self.tolerance, self.velocity_tolerance = tolerance, velocity_tolerance
        self.max_iterations = max_iterations

    @property
    def n_constraints(self):
        return len(self.rows)

    @torch.no_grad()
    def __call__(self, positions, velocities, masses, box=None):
        if positions.ndim != 3 or positions.shape[-1] != 3 or velocities.shape != positions.shape:
            raise ValueError("positions/velocities must match (B,N,3)")
        x, v = positions.double().clone(), velocities.double().clone()
        m = torch.as_tensor(masses, device=x.device, dtype=x.dtype)
        if m.shape != (x.shape[1],) or not torch.isfinite(m).all() or (m <= 0).any():
            raise ValueError("finite positive per-particle masses required")
        if not torch.isfinite(x).all() or not torch.isfinite(v).all():
            raise ValueError("cannot project nonfinite state")
        rows = self.rows.to(x.device)
        if len(rows) and rows[:, :2].max() >= x.shape[1]:
            raise ValueError("constraint index outside particle array")
        bx = None if box is None else box.to(x.device, x.dtype)
        for iteration in range(1, self.max_iterations + 1):
            for i0, j0, target in rows:
                i, j = int(i0), int(j0)
                dr = x[:, i] - x[:, j]
                if bx is not None:
                    dr = minimum_image(dr[:, None], bx)[:, 0]
                length = dr.norm(dim=-1)
                if (length < 1e-15).any():
                    raise RuntimeError("undefined constraint direction for coincident particles")
                unit = dr / length[:, None]
                wi, wj = 1 / m[i], 1 / m[j]
                shift = ((length - target) / (wi + wj))[:, None] * unit
                x[:, i] -= wi * shift
                x[:, j] += wj * shift
            # Remove relative velocity along each corrected bond.
            for i0, j0, _ in rows:
                i, j = int(i0), int(j0)
                dr = x[:, i] - x[:, j]
                if bx is not None:
                    dr = minimum_image(dr[:, None], bx)[:, 0]
                unit = dr / dr.norm(dim=-1)[:, None]
                wi, wj = 1 / m[i], 1 / m[j]
                shift = (((v[:, i] - v[:, j]) * unit).sum(-1) / (wi + wj))[:, None] * unit
                v[:, i] -= wi * shift
                v[:, j] += wj * shift
            distance_errors = []
            speed_errors = []
            for i0, j0, target in rows:
                i, j = int(i0), int(j0)
                dr = x[:, i] - x[:, j]
                if bx is not None:
                    dr = minimum_image(dr[:, None], bx)[:, 0]
                length = dr.norm(dim=-1)
                distance_errors.append(float((length - target).abs().max()))
                relative_speed = ((v[:, i] - v[:, j]) * dr).sum(-1)
                relative_speed = relative_speed / length.clamp_min(1e-30)
                speed_errors.append(float(relative_speed.abs().max()))
            distance = max(distance_errors, default=0.)
            speed = max(speed_errors, default=0.)
            if distance <= self.tolerance and speed <= self.velocity_tolerance:
                return x, v.to(velocities.dtype), dict(
                    iterations=iteration, distance_residual_A=distance,
                    velocity_residual_A_per_fs=speed,
                    max_position_correction_A=float((x - positions).norm(dim=-1).max()),
                    max_velocity_correction_A_per_fs=float((v - velocities).norm(dim=-1).max()))
        raise RuntimeError(f"constraint projection did not converge: distance={distance}, velocity={speed}")
