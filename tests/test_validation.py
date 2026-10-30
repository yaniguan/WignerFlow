"""Fast CPU validation cases with analytic or statistical answers."""
import numpy as np
import pytest

from equitraj.validation import validate, report_markdown


def ballistic():
    t = np.arange(21.)
    x = np.zeros((21, 2, 3))
    x[:, 0, 0] = .01 * t
    x[:, 1, 0] = 4 - .01 * t
    v = np.zeros_like(x)
    v[:, 0, 0], v[:, 1, 0] = .01, -.01
    return dict(positions=x, velocities=v, elements=["Na", "Cl"], masses=[23, 35.45],
                box=np.eye(3) * 20, dt_fs=1)


def test_ballistic_and_identical_rdf():
    d = ballistic()
    r = validate(d, d, max_lag=10, fit_window_fs=(2, 10), n_bins=20,
                 charges={"Na": 1, "Cl": -1}, temperature_K=1200)
    assert r["status"] == "ok"
    assert set(r["rollout"]["rdf"]) == {"Na-Na", "Cl-Na", "Cl-Cl"}
    assert r["errors"]["Cl-Na"]["rdf_L1_A"] == 0
    for s in ("Na", "Cl"):
        m = r["rollout"]["species"][s]
        assert np.allclose(m["msd_A2"], .0001 * np.arange(11) ** 2)
        assert np.allclose(m["vacf_A2_per_fs2"], .0001)
        assert np.allclose(m["vacf_integral_A2_per_fs"], .0001 * np.arange(11))
        assert m["diffusion_A2_per_fs"] == pytest.approx(.0001 * 12 / 6)
    assert r["rollout"]["stable_length_ns"] == pytest.approx(20e-6)
    assert r["rollout"]["temperature_drift_K_per_ns"] == pytest.approx(0, abs=1e-5)
    assert "Cl-Na" in report_markdown(r) and "S/m" in report_markdown(r)


def test_collective_charge_conductivity_correlation():
    d = ballistic()
    # Same velocity for neutral oppositely charged ions: collective current cancels.
    d["positions"][:, 1, 0] = d["positions"][:, 0, 0] + 4
    d["velocities"][:, 1] = d["velocities"][:, 0]
    r = validate(d, d, charges={"Na": 1, "Cl": -1}, temperature_K=1000)
    c = r["rollout"]["conductivity"]
    assert c["nernst_einstein_S_per_m"] > 0
    assert c["einstein_helfand_S_per_m"] == pytest.approx(0, abs=1e-20)
    diffusion = r["rollout"]["species"]["Na"]["diffusion_A2_per_fs"]
    expected = 2 * diffusion * 1e-5 * (1.602176634e-19)**2 / (8000e-30 * 1.380649e-23 * 1000)
    assert c["nernst_einstein_S_per_m"] == pytest.approx(expected)


def test_ideal_gas_partial_rdf():
    rng = np.random.default_rng(4)
    x = rng.uniform(0, 30, (12, 160, 3))
    d = dict(positions=x, velocities=np.zeros_like(x), box=np.eye(3) * 30,
             elements=["Na"] * 80 + ["Cl"] * 80, masses=np.full(160, 30), dt_fs=1)
    r = validate(d, d, r_max=12, n_bins=12, max_lag=2)
    for g in r["reference"]["rdf"].values():
        assert np.max(np.abs(np.array(g)[4:] - 1)) < .25


def test_random_walk_diffusion():
    rng = np.random.default_rng(7)
    diffusion, dt = .002, 2
    dx = rng.normal(0, np.sqrt(2 * diffusion * dt), (180, 64, 3))
    x = np.cumsum(dx, axis=0)
    d = dict(positions=x, velocities=dx / dt, elements=["Ar"] * 64, masses=np.full(64, 40), dt_fs=dt)
    r = validate(d, d, r_max=5, n_bins=8, max_lag=20, fit_window_fs=(4, 30))
    assert r["rollout"]["species"]["Ar"]["diffusion_A2_per_fs"] == pytest.approx(diffusion, rel=.15)


def test_wrapped_msd_and_stability():
    d = ballistic()
    x = d["positions"].copy()
    x[:, 0, 0] = 19.95 + .01 * np.arange(21)
    x[:, 1, 0] = 5 + .01 * np.arange(21)
    d["positions"] = x % 20
    r = validate(d, d, wrapped=True, max_lag=5)
    assert np.allclose(r["rollout"]["species"]["Na"]["msd_A2"], .0001 * np.arange(6)**2)
    broken = dict(d, positions=d["positions"].copy())
    broken["positions"][8] = np.nan
    r = validate(broken, d, wrapped=True, max_lag=4)
    assert r["rollout"]["stable_frames"] == 8
    assert r["rollout"]["stable_length_ns"] == pytest.approx(7e-6)
    broken["positions"][0] = np.nan
    r = validate(broken, d)
    assert r["status"] == "insufficient stable frames"
    assert "insufficient" in report_markdown(r)


def test_energy_temperature_and_invalid_fit():
    d = ballistic()
    energy = lambda x, v, m, box: 2 + x[0, 0] * 3
    r = validate(d, d, energy=energy, dof=3)
    assert r["rollout"]["energy_drift_kJmol_per_ns"] == pytest.approx(30000)
    expected = (23 + 35.45) * .0001 * 1e4 / (3 * .0083144626)
    assert r["rollout"]["temperature_mean_K"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="fit window"):
        validate(d, d, fit_window_fs=(100, 200))
    with pytest.raises(ValueError, match="charges"):
        validate(d, d, charges={"Na": 1})
    with pytest.raises(ValueError, match="radius"):
        validate(d, d, r_max=11)


def test_triclinic_lattice_translation_invariance():
    cell = np.array([[10., 0, 0], [2., 9, 0], [1., 1, 8.]])
    frame = np.array([[.1, .1, .1], [.3, .1, .1], [.2, .2, .2], [.4, .2, .2]]) @ cell
    x = np.broadcast_to(frame, (5, 4, 3)).copy()
    d = dict(positions=x, velocities=np.zeros_like(x), elements=["Na", "Na", "Cl", "Cl"],
             masses=[23, 23, 35.45, 35.45], box=cell, dt_fs=1)
    translated = x.copy()
    translated[:, 0] += cell[1] - 2 * cell[2]
    r = validate(dict(d, positions=translated), d, r_max=3.5, n_bins=13)
    assert all(e["rdf_L1_A"] == 0 for e in r["errors"].values())
    assert r["rollout"]["species"]["Na"]["diffusion_A2_per_fs"] == 0


def test_distance_and_temperature_failure_prefix():
    d = ballistic()
    collision = d["positions"].copy()
    collision[6, 1] = collision[6, 0]
    r = validate(dict(d, positions=collision), d, min_distance=.5)
    assert r["rollout"]["first_unstable_frame"] == 6
    hot = d["velocities"].copy()
    hot[7] *= 100
    r = validate(dict(d, velocities=hot), d, max_temperature=10000)
    assert r["rollout"]["first_unstable_frame"] == 7
