"""Tests for analysis functions on cases with known answers."""

import numpy as np

from equitraj import analysis


def test_minimum_image():
    L = np.array([10.0, 10.0, 10.0])
    dr = np.array([[6.0, -6.0, 4.0]])
    assert np.allclose(analysis.minimum_image(dr, L), [[-4.0, 4.0, 4.0]])


def test_rdf_ideal_gas_is_one():
    rng = np.random.default_rng(0)
    L = 20.0
    pos = rng.uniform(0, L, size=(20, 500, 3))
    box = np.tile(np.eye(3) * L, (20, 1, 1))
    idx = np.arange(500)
    r, g, rho = analysis.rdf(pos, box, idx, idx, r_max=9.0, n_bins=30)
    assert np.isclose(rho, 499 / L**3)
    assert np.abs(g[5:] - 1.0).max() < 0.1


def test_coordination_number_ideal_gas():
    r = np.linspace(0.005, 5.0, 500)
    g = np.ones_like(r)
    rho = 0.03
    n = analysis.coordination_number(r, g, rho, 3.0)
    assert abs(n - rho * 4 / 3 * np.pi * 27) / n < 0.01


def test_first_peak_parabola():
    r = np.linspace(0, 10, 1001)
    g = np.exp(-((r - 3.7123) ** 2))
    pos, height = analysis.first_peak(r, g)
    assert abs(pos - 3.7123) < 1e-3 and abs(height - 1.0) < 1e-4


def test_vacf_constant_and_oscillator():
    # constant velocity: VACF is flat
    v = np.ones((100, 4, 3))
    c = analysis.vacf(v, 10)
    assert np.allclose(c, 3.0)
    # harmonic oscillator: VACF ~ cos(w t), VDOS peak at w
    dt = 1.0  # fs
    wavenumber = 500.0  # cm^-1
    freq = wavenumber * 2.99792458e10 / 1e15  # 1/fs
    t = np.arange(4000) * dt
    v = np.cos(2 * np.pi * freq * t)[:, None, None] * np.ones((1, 2, 3))
    c = analysis.vacf(v, 1000)
    k, s = analysis.vdos(c, dt)
    assert abs(k[np.argmax(s)] - wavenumber) < 10


def test_msd_ballistic_and_diffusion():
    t = np.arange(50)
    x = np.zeros((50, 3, 3))
    x[:, :, 0] = 2.0 * t[:, None]  # moving at 2 Å/frame along x
    m = analysis.msd(x, [0, 1, 5])
    assert np.allclose(m, [0, 4, 100])
    # MSD = 6 D t -> D recovered
    tf = np.linspace(0, 1000, 101)
    d, d_cm2 = analysis.diffusion_coefficient(tf, 6 * 0.01 * tf, 100, 1000)
    assert np.isclose(d, 0.01) and np.isclose(d_cm2, 0.001)


def test_energy_drift():
    t = np.linspace(0, 100, 101)  # ps
    e = 5.0 + 0.01 * t
    assert np.isclose(analysis.energy_drift(t, e, dof=10), 1.0)  # 0.01 kJ/mol/ps = 10 /ns, /10 dof
