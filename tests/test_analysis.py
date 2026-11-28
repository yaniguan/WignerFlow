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


def test_csvr_samples_canonical_kinetic_energy():
    """Repeated CSVR steps give the canonical mean temperature and the right relative fluctuation sqrt(2/dof)."""
    import torch
    from equitraj.rollout import temperature
    from equitraj.thermostat import csvr
    torch.manual_seed(0)
    n = 50
    masses = torch.full((n,), 40.0)
    v = torch.randn(200, n, 3) * 0.0001  # start very cold
    temps = []
    for i in range(400):
        v = csvr(v, masses, 100.0, tau_fs=50.0, dt_fs=10.0, dof=3 * n - 3)
        if i > 200:
            temps.append(temperature(v, masses))
    t = torch.cat(temps)
    assert abs(t.mean().item() - 100.0) < 1.5
    assert abs(t.std().item() / 100.0 - (2.0 / (3 * n - 3)) ** 0.5) < 0.02


def test_kinematics_constant_velocity_trapezoid_residual_zero():
    rng = np.random.default_rng(7)
    n_times, n_atoms, dt_fs = 40, 12, 50.0
    box_l = 18.0
    box = np.eye(3) * box_l
    masses = np.full(n_atoms, 39.948)
    v0 = rng.normal(0.0, 0.0025, size=(n_atoms, 3))
    v0 -= v0.mean(axis=0, keepdims=True)
    x0 = rng.uniform(0.0, box_l, size=(n_atoms, 3))
    t_grid = np.arange(n_times)[:, None, None] * dt_fs
    pos_unwrapped = x0[None, :, :] + v0[None, :, :] * t_grid
    pos_wrapped = np.mod(pos_unwrapped, box_l)
    vel = np.broadcast_to(v0[None, :, :], pos_unwrapped.shape).copy()

    # Exact constant-velocity motion has zero trapezoid residual both unwrapped and PBC-wrapped
    assert analysis.trapezoid_residual(pos_unwrapped, vel, dt_fs) < 1e-12
    assert analysis.trapezoid_residual(pos_wrapped, vel, dt_fs, box=box) < 1e-12
    kin = analysis.kinematic_consistency(pos_wrapped, vel, dt_fs, masses, box=box)
    assert kin["trapezoid_residual"] < 1e-12
    assert np.isclose(kin["temperature_rel_gap"], 0.0, atol=1e-12)


def test_kinematics_scaled_velocities_show_21_percent_temperature_gap():
    rng = np.random.default_rng(11)
    n_times, n_atoms, dt_fs = 50, 16, 50.0
    masses = np.full(n_atoms, 39.948)
    v_true = rng.normal(0.0, 0.0022, size=(n_atoms, 3))
    v_true -= v_true.mean(axis=0, keepdims=True)
    x0 = rng.uniform(0.0, 20.0, size=(n_atoms, 3))
    pos = x0[None, :, :] + v_true[None, :, :] * (np.arange(n_times)[:, None, None] * dt_fs)
    vel_scaled = np.broadcast_to((1.1 * v_true)[None, :, :], pos.shape).copy()

    kin = analysis.kinematic_consistency(pos, vel_scaled, dt_fs, masses)
    # Scaling velocities by 1.1 raises velocity temperature by 1.1^2 - 1 = 0.21 (21%) over displacement temperature
    assert np.isclose(kin["temperature_rel_gap"], 1.1 ** 2 - 1.0, rtol=1e-6, atol=1e-8)
    assert kin["trapezoid_residual"] > 0.09


def test_green_kubo_diffusion_units_and_exponential_vacf():
    dt_fs = 5.0
    tau_fs = 120.0
    c0 = 6.0e-6  # Å^2/fs^2
    t = np.arange(0, 2400.0, dt_fs)
    c_vacf = c0 * np.exp(-t / tau_fs)
    d_a2_fs, d_cm2_s = analysis.green_kubo_diffusion(c_vacf, dt_fs)
    expected_a2_fs = (c0 * tau_fs) / 3.0
    assert np.isclose(d_cm2_s, 0.1 * d_a2_fs, rtol=1e-12)
    assert np.isclose(d_a2_fs, expected_a2_fs, rtol=2e-3)
