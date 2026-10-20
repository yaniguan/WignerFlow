"""lj_potential reproduces the OpenMM energy of the argon system (the one the ground truth was generated with)."""

import numpy as np
import pytest
import torch

openmm = pytest.importorskip("openmm")

from equitraj.energy import lj_potential  # noqa: E402
from equitraj.systems import build_argon  # noqa: E402


def test_lj_potential_matches_openmm():
    system, _, positions, _ = build_argon()
    box_nm = system.getDefaultPeriodicBoxVectors()[0][0].value_in_unit(openmm.unit.nanometer)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    rng = np.random.default_rng(0)
    for _ in range(2):
        pos = np.asarray(positions) + rng.normal(scale=0.03, size=np.shape(positions))  # nm, a thermally disordered fcc cell
        context.setPositions(pos)
        e_omm = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(openmm.unit.kilojoule_per_mole)
        box_a = torch.eye(3) * box_nm * 10.0
        e_mine = lj_potential(torch.from_numpy(pos * 10.0)[None], box_a).item()
        assert abs(e_mine - e_omm) < 1e-3 * abs(e_omm) + 1e-3, (e_mine, e_omm)
