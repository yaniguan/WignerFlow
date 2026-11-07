import numpy as np
import pytest

from equitraj.timeline import block_slices, kinetic_energy, stability_timeline


def _lattice_traj(n_frames=40, seed=0, heat_after=None):
    rng = np.random.default_rng(seed)
    g = np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) * 3.0
    L = 12.0
    pos = g[None] + 0.05 * rng.standard_normal((n_frames, len(g), 3))
    masses = np.full(len(g), 39.948)
    vel = 0.002 * rng.standard_normal((n_frames, len(g), 3))
    if heat_after is not None:
        vel[heat_after:] *= 3.0
    box = np.broadcast_to(np.eye(3) * L, (n_frames, 3, 3)).copy()
    t = np.arange(n_frames) * 1000.0  # 1 ps frames
    return pos, vel, masses, t, box


def test_block_slices():
    s = block_slices(10, 3)
    assert s[0].start == 0 and s[-1].stop == 10 and len(s) == 3
    with pytest.raises(ValueError):
        block_slices(2, 5)


def test_kinetic_energy_units():
    ke = kinetic_energy(np.ones((1, 2, 3)) * 0.01, np.array([1.0, 1.0]))
    assert np.isclose(ke[0], 0.5 * 2 * 3 * 1e-4 * 1e4)


def test_stable_run_full_time():
    rep = stability_timeline(*_lattice_traj(), n_blocks=4, max_temp_dev=1e6)
    assert rep["first_unstable_block"] is None
    assert np.isclose(rep["stable_time_ns"], 0.039)
    assert rep["blocks"][0]["rdf_l1"] == pytest.approx(0.0)


def test_heating_detected():
    rep = stability_timeline(*_lattice_traj(heat_after=20), n_blocks=4, max_temp_dev=20.0)
    assert rep["first_unstable_block"] == 2
    assert "temperature" in rep["blocks"][2]["violations"]
    assert rep["temperature_drift_K_per_ns"] > 0
    assert np.isclose(rep["stable_time_ns"], 0.020)


def test_melting_rdf_and_energy_drift():
    pos, vel, m, t, box = _lattice_traj()
    rng = np.random.default_rng(1)
    pos[30:] = rng.uniform(0, 12.0, pos[30:].shape)
    pe = np.linspace(0, 100, len(t))
    rep = stability_timeline(pos, vel, m, t, box, n_blocks=4, potential_energy=pe,
                             max_temp_dev=1e6, min_distance=0.0)
    assert rep["first_unstable_block"] == 3 and "rdf" in rep["blocks"][3]["violations"]
    assert rep["energy_drift_kJmol_per_ns_per_dof"] > 0


def test_nan_flagged():
    pos, vel, m, t, box = _lattice_traj()
    pos[-1, 0] = np.nan
    rep = stability_timeline(pos, vel, m, t, box, n_blocks=4, max_temp_dev=1e6)
    assert "nan" in rep["blocks"][-1]["violations"]
