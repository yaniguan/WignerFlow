"""Tests for the ground-truth data pipeline: systems, integrator, HDF5 files."""

import numpy as np
import openmm
import openmm.unit as u
import pytest

from equitraj.simulate import run_trajectory, velocity_verlet
from equitraj.systems import build_argon, fcc_lattice
from equitraj.trajfile import read_traj


def test_fcc_lattice_count_and_spacing():
    pos = fcc_lattice(4, 1.0)
    assert pos.shape == (256, 3)
    # nearest-neighbor distance in fcc is a / sqrt(2)
    d = np.linalg.norm(pos[1:] - pos[0], axis=1)
    assert np.isclose(d.min(), 1 / np.sqrt(2))


def test_argon_box_matches_density():
    _, _, _, info = build_argon()
    # 256 atoms at 1.374 g/cm^3 -> L ~ 2.31 nm
    assert abs(info["box_nm"] - 2.312) < 0.005


def test_lj_minimum_energy():
    """Two argon atoms at r = 2^(1/6) sigma have energy -epsilon."""
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.NoCutoff)
    for _ in range(2):
        system.addParticle(39.948)
        nb.addParticle(0.0, 0.3405, 0.996)
    system.addForce(nb)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    r = 2 ** (1 / 6) * 0.3405
    ctx.setPositions([[0, 0, 0], [r, 0, 0]])
    e = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    assert abs(e + 0.996) < 1e-6


def test_velocity_verlet_is_time_reversible():
    """Integrate forward, flip velocities, integrate the same number of steps: we must return to the start."""
    system, _, positions, _ = build_argon(n_cells=3)  # 108 atoms; the box must be > 2 x cutoff
    integ = velocity_verlet(0.002)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(ctx)
    ctx.setVelocitiesToTemperature(94.4, 1)
    s0 = ctx.getState(getPositions=True, getVelocities=True)
    integ.step(50)
    v = ctx.getState(getVelocities=True).getVelocities(asNumpy=True)
    ctx.setVelocities(-v)
    integ.step(50)
    x1 = ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    x0 = s0.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    assert np.abs(x1 - x0).max() < 1e-6


@pytest.fixture(scope="module")
def short_traj(tmp_path_factory):
    cfg = {
        "system": "argon", "temperature_K": 94.4, "dt_fs": 5.0, "friction_per_ps": 1.0,
        "equil_ps": 2, "production_ps": 1, "save_every": 2, "force_every": 5,
        "platform": "Reference", "precision": "double", "constraint_tol": 1e-8,
    }
    path = tmp_path_factory.mktemp("traj") / "t.h5"
    run_trajectory(cfg, seed=7, out_path=str(path))
    return read_traj(str(path)), cfg


def test_traj_file_contents(short_traj):
    t, cfg = short_traj
    n_frames = int(1.0 / 0.005 / 2)
    assert t["positions"].shape == (n_frames, 256, 3)
    assert t["velocities"].shape == (n_frames, 256, 3)
    assert t["forces"].shape == ((n_frames + 4) // 5, 256, 3)
    assert np.allclose(np.diff(t["time"]), 0.010)  # 2 steps of 5 fs
    assert t["attrs"]["complete"]
    assert t["attrs"]["ensemble"] == "NVE"
    assert list(t["elements"][:2]) == ["Ar", "Ar"]


def test_traj_nve_and_com(short_traj):
    t, _ = short_traj
    e = t["potential_energy"] + t["kinetic_energy"]
    assert e.std() < 0.05  # kJ/mol over 1 ps
    p = (t["velocities"] * t["masses"][None, :, None]).sum(axis=1)
    assert np.abs(p).max() < 1e-3  # center of mass stays at rest


def test_velocities_match_positions(short_traj):
    """Velocities are synchronous with positions (not half a step off, as with leapfrog)."""
    t, _ = short_traj
    dt = t["time"][1] - t["time"][0]
    x, v = t["positions"], t["velocities"]
    central = (x[2:] - x[:-2]) / (2 * dt)
    rel = np.linalg.norm(central - v[1:-1]) / np.linalg.norm(v[1:-1])
    assert rel < 0.02
