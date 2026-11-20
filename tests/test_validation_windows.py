"""Validation rollout starts must have enough ground truth after them, and temperature can count constraints."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from equitraj.rollout import temperature
from equitraj.train import validation_starts
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB


def fake_ds(lengths, f, k, m=1):
    span = (k + m - 1) * f
    index = [(t, s) for t, n in enumerate(lengths) for s in range(0, n - span)]
    return SimpleNamespace(f=f, pos=[np.zeros((n, 1, 3)) for n in lengths], index=index)


def test_starts_have_enough_ground_truth():
    k, f, n_steps = 3, 2, 10
    ds = fake_ds([60, 40], f, k)
    starts = validation_starts(ds, k, 4, n_steps)
    need = (k - 1 + n_steps) * f + 1
    for i in starts:
        t, s0 = ds.index[i]
        assert s0 + need <= len(ds.pos[t])
    assert len(set(starts.tolist())) == 4


def test_no_valid_start_raises():
    ds = fake_ds([20], 2, 3)
    with pytest.raises(ValueError, match="ground truth"):
        validation_starts(ds, 3, 4, 100)


def test_temperature_counts_constraints():
    n = 22
    vel = torch.randn(1, n, 3) * 0.003
    masses = torch.full((n,), 12.0)
    t0 = temperature(vel, masses, 0).item()
    t12 = temperature(vel, masses, 12).item()
    assert abs(t12 / t0 - (3 * n - 3) / (3 * n - 3 - 12)) < 1e-9  # fewer degrees of freedom: same energy, higher T
    ke = 0.5 * (masses[None, :, None].double() * vel.double() ** 2).sum().item() * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    assert abs(t12 - 2 * ke / ((3 * n - 15) * KB)) < 1e-6 * t12


def test_temperature_default_has_no_constraints():
    vel = torch.randn(2, 256, 3) * 0.003  # argon-sized system
    masses = torch.full((256,), 39.948)
    assert torch.equal(temperature(vel, masses), temperature(vel, masses, 0))
