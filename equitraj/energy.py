"""Reference energy of the argon ground-truth force field for checking rollout energy drift.

The ground truth uses Lennard-Jones argon in OpenMM:
    E = 4 * eps * [(sigma / r)^12 - (sigma / r)^6] * S(x)
with switching function S(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5 for x = (r - r_switch) / (r_cut - r_switch)
between r_switch and r_cut (0 beyond), minimum image, and no dispersion correction (equitraj/systems.py).

Units: positions in Å, energies in kJ/mol, masses in amu, velocities in Å/fs.
"""

import torch

from equitraj.neighbors import minimum_image
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL

EPSILON = 0.996  # kJ/mol
SIGMA = 3.405  # Å
R_SWITCH = 8.0  # Å
R_CUT = 8.5  # Å


def lj_potential(pos, box):
    """Potential energy (kJ/mol) of each frame. pos: (F, N, 3) in Å, box: (3, 3) in Å."""
    pos = pos.double()
    n = pos.shape[1]
    bx = box.to(device=pos.device, dtype=torch.float64)
    dr = minimum_image(pos[:, :, None, :] - pos[:, None, :, :], bx)
    eye = torch.eye(n, dtype=torch.float64, device=pos.device) * 1e9
    r = torch.linalg.norm(dr, dim=-1) + eye
    sr6 = (SIGMA / r) ** 6
    e = 4.0 * EPSILON * (sr6**2 - sr6)
    x = ((r - R_SWITCH) / (R_CUT - R_SWITCH)).clamp(0.0, 1.0)
    e = e * (1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5)
    return 0.5 * e.sum(dim=(1, 2))


def kinetic(vel, masses):
    """Kinetic energy (kJ/mol) of each frame. vel: (F, N, 3) in Å/fs."""
    m = masses.to(device=vel.device, dtype=torch.float64)
    return 0.5 * torch.einsum("n,fnd->f", m, vel.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
