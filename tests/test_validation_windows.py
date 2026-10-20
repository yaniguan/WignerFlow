"""Validation rollout starts must have enough ground truth after them, and temperature uses the constrained degrees of freedom."""

import torch

from equitraj.rollout import temperature
from equitraj.units import KB


def test_temperature_counts_constraints():
    n = 22
    vel = torch.randn(1, n, 3) * 0.003
    masses = torch.full((n,), 12.0)
    t0 = temperature(vel, masses, 0).item()
    t12 = temperature(vel, masses, 12).item()
    assert abs(t12 / t0 - (3 * n - 3) / (3 * n - 3 - 12)) < 1e-9  # fewer degrees of freedom: same energy, higher T
    ke = 0.5 * (masses[None, :, None] * vel**2).sum().item() * 1e4
    assert abs(t12 - 2 * ke / ((3 * n - 15) * KB)) < 1e-6 * t12
