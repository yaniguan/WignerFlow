"""Reference energy of the argon ground-truth force field, so rollouts can be checked for energy drift.

The data were made with Lennard-Jones argon in OpenMM: E = 4 eps [(sigma/r)^12 - (sigma/r)^6] times a switching function
S(x) = 1 - 10 x^3 + 15 x^4 - 6 x^5, x = (r - r_switch) / (r_cut - r_switch), between r_switch and r_cut (0 beyond), minimum image,
no dispersion correction (equitraj/systems.py). tests/test_energy.py checks this against OpenMM.

Units: Å and kJ/mol, masses in amu, velocities in Å/fs.
"""

import torch

from equitraj.neighbors import minimum_image
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL

EPSILON = 0.996  # kJ/mol
SIGMA = 3.405  # Å
R_SWITCH = 8.0  # Å
R_CUT = 8.5  # Å


def lj_potential(pos, box):
    """Potential energy (kJ/mol) of each frame. pos: (F, N, 3) in Å, box: (3, 3) in Å (one cubic box for all frames)."""
    pos = pos.double()
    n = pos.shape[1]
    dr = minimum_image(pos[:, :, None, :] - pos[:, None, :, :], box.double())
    r = torch.linalg.norm(dr, dim=-1) + torch.eye(n, dtype=torch.float64) * 1e9  # no self term
    sr6 = (SIGMA / r) ** 6
    e = 4 * EPSILON * (sr6**2 - sr6)
    x = ((r - R_SWITCH) / (R_CUT - R_SWITCH)).clamp(0.0, 1.0)
    e = e * (1 - 10 * x**3 + 15 * x**4 - 6 * x**5)
    return 0.5 * e.sum(dim=(1, 2))  # every pair was counted twice


def kinetic(vel, masses):
    """Kinetic energy (kJ/mol) of each frame. vel: (F, N, 3) in Å/fs."""
    return 0.5 * torch.einsum("n,fnd->f", masses.double(), vel.double() ** 2) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
