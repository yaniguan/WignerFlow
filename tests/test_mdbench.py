"""Unit tests for the pure-MD coarse-stride benchmark suite (equitraj.mdbench)."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from equitraj.mdbench import (
    EquivariantPairFlowModel,
    compute_bond_stability,
    compute_hr_density,
    compute_hr_mae,
    compute_kinetic_temperature,
    compute_vdos_metrics,
    evaluate_md17_molecule,
    infer_covalent_bonds,
    masses_from_atomic_numbers,
    project_zero_com_motion,
)


def _synthetic_water_like_traj(n_frames: int = 300, dt_fs: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    z = np.array([8, 1, 1], dtype=np.int64)
    base = np.array(
        [
            [0.000, 0.000, 0.000],
            [0.958, 0.000, 0.000],
            [-0.240, 0.927, 0.000],
        ],
        dtype=np.float64,
    )
    t = np.arange(n_frames, dtype=np.float64) * dt_fs
    traj = np.repeat(base[None, :, :], n_frames, axis=0)
    # Realistic ~0.03 A OH stretch + gentle bend
    traj[:, 1, 0] += 0.028 * np.sin(0.55 * t) + rng.normal(0.0, 0.004, size=n_frames)
    traj[:, 2, 1] += 0.028 * np.cos(0.52 * t) + rng.normal(0.0, 0.004, size=n_frames)
    traj[:, 2, 0] += 0.020 * np.sin(0.18 * t)
    forces = -250.0 * (traj - base[None, :, :])
    return z, traj, forces


def test_covalent_bonds_and_com_projection() -> None:
    z, traj, _ = _synthetic_water_like_traj(20)
    bonds, b0 = infer_covalent_bonds(traj[0], z)
    assert bonds.shape == (2, 2)
    expected_b0 = np.linalg.norm(traj[0, [1, 2]] - traj[0, 0], axis=-1)
    assert np.allclose(b0, expected_b0, atol=1e-12)

    masses = masses_from_atomic_numbers(z)
    shifted_r = traj[0] + np.array([4.0, -2.5, 1.2])
    shifted_v = np.array([[0.02, -0.01, 0.005], [0.03, 0.01, -0.02], [-0.01, 0.02, 0.01]])
    pos_c, vel_c = project_zero_com_motion(shifted_r, shifted_v, masses)
    com_r = np.sum(pos_c * masses[:, None], axis=0) / np.sum(masses)
    com_v = np.sum(vel_c * masses[:, None], axis=0) / np.sum(masses)
    ang_l = np.sum(masses[:, None] * np.cross(pos_c, vel_c), axis=0)
    assert np.allclose(com_r, 0.0, atol=1e-12)
    assert np.allclose(com_v, 0.0, atol=1e-12)
    assert np.allclose(ang_l, 0.0, atol=1e-10)


def test_fu2023_stability_hr_and_vdos_metrics() -> None:
    z, traj, _ = _synthetic_water_like_traj(240)
    stab_ok = compute_bond_stability(traj, z, traj[0], dt_frame_fs=10.0)
    assert stab_ok["is_stable"] is True
    assert stab_ok["stable_fraction"] == pytest.approx(1.0)
    assert stab_ok["max_bond_dev_a"] < 0.15

    ruptured = traj.copy()
    ruptured[50:, 1, 0] += 1.25  # > 0.5 A bond rupture
    stab_bad = compute_bond_stability(ruptured, z, traj[0], dt_frame_fs=10.0)
    assert stab_bad["is_stable"] is False
    assert stab_bad["t_stab_ps"] == pytest.approx(0.50)

    r_grid, h = compute_hr_density(traj, r_max=4.0)
    dr = float(r_grid[1] - r_grid[0])
    assert float(np.sum(h) * dr) == pytest.approx(1.0, abs=1e-6)
    assert compute_hr_mae(traj, traj, r_max=4.0) == pytest.approx(0.0, abs=1e-12)

    masses = masses_from_atomic_numbers(z)
    vel = np.gradient(traj, 0.5, axis=0)
    vdos = compute_vdos_metrics(vel, vel, masses, dt_model_fs=5.0, dt_ref_fs=5.0)
    assert vdos["vdos_cosine_sim"] == pytest.approx(1.0, abs=1e-6)
    assert vdos["vdos_mae"] == pytest.approx(0.0, abs=1e-6)


def test_wigner_coarse_rollout_offline_archive(tmp_path: Path) -> None:
    # Build a synthetic offline 4,000-frame ethanol-shaped .npz cache to verify zero-network CI
    rng = np.random.default_rng(19)
    z = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1], dtype=np.int64)
    pos0 = np.array(
        [
            [0.00, 0.00, 0.00],
            [1.52, 0.00, 0.00],
            [2.05, 1.32, 0.00],
            [-0.38, 1.02, 0.00],
            [-0.38, -0.51, 0.88],
            [-0.38, -0.51, -0.88],
            [1.89, -0.54, 0.88],
            [1.89, -0.54, -0.88],
            [3.01, 1.30, 0.00],
        ],
        dtype=np.float64,
    )
    t = np.arange(4000, dtype=np.float64)[:, None, None] * 0.5
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, 9, 3))
    r_arr = pos0[None, :, :] + 0.025 * np.sin(0.22 * t + phase)
    f_arr = -180.0 * (r_arr - pos0[None, :, :])
    e_arr = np.zeros(4000, dtype=np.float64)

    np.savez_compressed(
        tmp_path / "ethanol_ccsd_t.npz",
        z=z,
        R_train=r_arr[:1000],
        F_train=f_arr[:1000],
        R_test=r_arr[1000:2000],
        F_test=f_arr[1000:2000],
    )
    res = evaluate_md17_molecule(
        "ethanol_ccsd_t", cache_dir=tmp_path, sim_horizon_ps=0.6, target_temp_k=120.0, seed=11
    )
    assert res.wf_10fs_stable is True
    assert res.wf_20fs_stable is True
    assert res.wf_10fs_hr_mae < 0.35
    assert res.wf_10fs_com_drift_a_fs < 1e-10

    # Direct check on EquivariantPairFlowModel kinetic temperature sanity
    model = EquivariantPairFlowModel.fit_from_trajectory(z, r_arr[:800], f_arr[:800])
    _, wf_v, _ = model.rollout_wigner_coarse(r_arr[0], (r_arr[1] - r_arr[0]) / 0.5, n_steps=30, dt_fs=10.0)
    temps = compute_kinetic_temperature(wf_v, model.masses)
    assert np.all(np.isfinite(temps))
