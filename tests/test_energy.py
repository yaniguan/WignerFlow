"""Tests for equitraj/energy.py (analytical LJ + switching checks on CPU, plus OpenMM parity when installed)."""

import numpy as np
import pytest
import torch

from equitraj.energy import EPSILON, R_CUT, R_SWITCH, SIGMA, kinetic, lj_potential
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL


def test_lj_dimer_minimum_and_cutoff_switch():
    box = torch.eye(3) * 30.0
    r_min = SIGMA * (2.0 ** (1.0 / 6.0))
    pos_min = torch.tensor([[[0.0, 0.0, 0.0], [r_min, 0.0, 0.0]]], dtype=torch.float32)
    e_min = lj_potential(pos_min, box)
    assert torch.isclose(e_min, torch.tensor([-EPSILON], dtype=torch.float64), atol=1e-7)

    # Halfway through switching region [R_SWITCH, R_CUT], S(0.5) = 0.5
    r_mid = 0.5 * (R_SWITCH + R_CUT)
    pos_mid = torch.tensor([[[0.0, 0.0, 0.0], [r_mid, 0.0, 0.0]]], dtype=torch.float32)
    sr6 = (SIGMA / r_mid) ** 6
    raw_lj = 4.0 * EPSILON * (sr6**2 - sr6)
    assert torch.isclose(lj_potential(pos_mid, box), torch.tensor([0.5 * raw_lj], dtype=torch.float64), atol=1e-9)

    # Beyond R_CUT, switched potential is strictly zero
    pos_far = torch.tensor([[[0.0, 0.0, 0.0], [R_CUT + 0.2, 0.0, 0.0]]], dtype=torch.float32)
    assert lj_potential(pos_far, box).item() == 0.0


def test_kinetic_energy_units():
    masses = torch.tensor([40.0, 20.0], dtype=torch.float32)
    vel = torch.tensor([[[0.01, 0.0, 0.0], [0.0, -0.02, 0.0]]], dtype=torch.float32)
    expected = 0.5 * (40.0 * 1e-4 + 20.0 * 4e-4) * AMU_A2_PER_FS2_TO_KJ_PER_MOL
    assert torch.isclose(kinetic(vel, masses), torch.tensor([expected], dtype=torch.float64), rtol=1e-6)


def test_pure_numpy_lj_argon_forces_match_potential_gradient():
    from equitraj.simulate import _lj_argon_energy_forces_nm

    rng = np.random.default_rng(42)
    box_nm = 2.30
    # Place pairs across repulsive, attractive, and [0.80, 0.85] nm switching regions
    pos_nm = np.array([
        [0.10, 0.20, 0.30],
        [0.46, 0.22, 0.31],
        [0.925, 0.20, 0.30],  # ~0.825 nm from atom 0 (inside switching window!)
        [1.40, 1.30, 1.20],
    ], dtype=np.float64) + rng.normal(0.0, 0.005, size=(4, 3))
    pe, forces = _lj_argon_energy_forces_nm(pos_nm, box_nm)
    pe_torch = lj_potential(torch.from_numpy(pos_nm * 10.0)[None], torch.eye(3, dtype=torch.float64) * (box_nm * 10.0))
    assert np.isclose(pe, float(pe_torch.item()), atol=1e-10)

    eps = 1.0e-6
    fd_forces = np.zeros_like(pos_nm)
    for i in range(len(pos_nm)):
        for d in range(3):
            p_plus = pos_nm.copy()
            p_minus = pos_nm.copy()
            p_plus[i, d] += eps
            p_minus[i, d] -= eps
            e_p, _ = _lj_argon_energy_forces_nm(p_plus, box_nm)
            e_m, _ = _lj_argon_energy_forces_nm(p_minus, box_nm)
            fd_forces[i, d] = -(e_p - e_m) / (2.0 * eps)
    assert np.allclose(forces, fd_forces, rtol=1e-7, atol=1e-7)


def test_lj_potential_matches_openmm():
    openmm = pytest.importorskip("openmm")
    from equitraj.systems import build_argon

    system, _, positions, _ = build_argon()
    box_nm = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(openmm.unit.nanometer)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(0)
    for _ in range(2):
        pos = np.asarray(positions) + rng.normal(scale=0.03, size=np.shape(positions))
        context.setPositions(pos)
        e_omm = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        box_a = torch.eye(3) * box_nm * 10.0
        e_mine = lj_potential(torch.from_numpy(pos * 10.0)[None], box_a).item()
        assert abs(e_mine - e_omm) < 1e-3 * abs(e_omm) + 1e-3, (e_mine, e_omm)
