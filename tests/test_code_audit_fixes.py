"""Regression tests for core physics, dtype, PBC, thermostat, and training audit fixes."""

import sys

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("e3nn")

from equitraj import analysis
from equitraj.constraints import DistanceProjector
from equitraj.distill import distillation_loss
from equitraj.evaluation import first_zero
from equitraj.fast import FastRollout
from equitraj.forcefield import EquiFF, verlet_rollout
from equitraj.model import EquiTraj, remove_com, timestep_embedding
from equitraj.rollout import max_bond_deviation
from equitraj.thermostat import csvr
from equitraj.trajfile import TrajWriter


def test_csvr_zero_velocity_and_dof1_are_finite():
    masses = torch.tensor([12.0, 1.0, 16.0], dtype=torch.float32)
    v_zero = torch.zeros((2, 3, 3), dtype=torch.float32)
    gen = torch.Generator().manual_seed(17)
    v_out = csvr(v_zero, masses, t_target=300.0, tau_fs=100.0, dt_fs=10.0, dof=6, generator=gen)
    assert v_out.dtype == torch.float32
    assert torch.isfinite(v_out).all()
    assert (v_out.norm(dim=-1) > 0).all()
    com = (v_out.double() * masses.double()[None, :, None]).sum(dim=1)
    assert torch.allclose(com, torch.zeros_like(com), atol=1e-6)

    v64_out = csvr(v_zero.double(), masses.double(), t_target=300.0, tau_fs=100.0, dt_fs=10.0, dof=6, generator=gen)
    com64 = (v64_out * masses.double()[None, :, None]).sum(dim=1)
    assert torch.allclose(com64, torch.zeros_like(com64), atol=1e-12)

    v1 = torch.tensor([[[0.01, 0.0, 0.0]]], dtype=torch.float32)
    v1_out = csvr(v1, torch.tensor([4.0]), t_target=100.0, tau_fs=50.0, dt_fs=10.0, dof=1, generator=gen)
    assert torch.isfinite(v1_out).all()

    with pytest.raises(ValueError, match="dof"):
        csvr(v1, torch.tensor([4.0]), t_target=100.0, tau_fs=50.0, dt_fs=10.0, dof=0)


def test_distance_projector_preserves_input_dtypes():
    x32 = torch.tensor([[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]]], dtype=torch.float32)
    v32 = torch.tensor([[[0.02, 0.0, 0.0], [-0.01, 0.0, 0.0]]], dtype=torch.float32)
    m = torch.tensor([12.0, 1.0], dtype=torch.float32)
    xp, vp, _ = DistanceProjector([[0, 1, 1.1]])(x32, v32, m)
    assert xp.dtype == torch.float32
    assert vp.dtype == torch.float32

    x64 = x32.double()
    xp64, vp64, _ = DistanceProjector([[0, 1, 1.1]])(x64, v32, m)
    assert xp64.dtype == torch.float64
    assert vp64.dtype == torch.float32


def test_distillation_loss_supports_mass_scaling_and_position_only():
    pos = torch.randn(2, 3, 4, 3)
    vel = torch.randn(2, 3, 4, 3) * 0.01
    types = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    masses = torch.tensor([12.0, 1.0, 12.0, 1.0])
    kw = dict(
        n_types=2, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow",
        target_scaling="mass", use_velocity=False, stride_fs=10.0,
    )
    teacher = EquiTraj(**kw)
    student = EquiTraj(**kw)
    loss = distillation_loss(student, teacher, pos, vel, types, None, masses, teacher_steps=4, supervised_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()


def test_model_odd_embedding_large_k_bf16_prior_and_geometry_aware_fast():
    emb = timestep_embedding(torch.tensor([0.2, 0.8]), 17)
    assert emb.shape == (2, 17)

    v = torch.randn(2, 4, 3, dtype=torch.float32)
    m64 = torch.tensor([1.0, 12.0, 16.0, 14.0], dtype=torch.float64)
    vc = remove_com(v, m64)
    assert vc.dtype == torch.float32

    m_bigk = EquiTraj(
        n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, k=20,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow",
        repulsion_prior={"enabled": True, "rmin": 2.5},
    )
    pos = torch.randn(1, 20, 4, 3) * 0.5
    vel = torch.randn(1, 20, 4, 3) * 0.01
    types = torch.zeros(4, dtype=torch.long)
    masses = torch.ones(4)

    fast_bf16 = FastRollout(m_bigk, sampling_steps=2, precision="bf16", neighbor="brute")
    dx, vn = fast_bf16.predict(pos, vel, types, None, masses)
    assert torch.isfinite(dx).all() and torch.isfinite(vn).all()

    m_geom = EquiTraj(
        n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", geometry_aware=True,
    )
    with pytest.warns(RuntimeWarning, match="geometry_aware"):
        fr = FastRollout(m_geom, sampling_steps=2, neighbor="brute", cuda_graphs=True)
    assert fr.cuda_graphs is False


def test_so3_float64_wigner_cache_under_float32_default_dtype():
    from equitraj import bench, so3

    so3._Y_TO_X.clear()
    torch.set_default_dtype(torch.float32)
    model = bench.make_model("regression", torch.float64, channels=16, heads=2, n_rbf=8, cutoff=4.0,
                             n_spatial=1, n_fusion=1, n_denoiser=1)
    inp = bench.make_liquid(16, k=2, periodic=False, dtype=torch.float64)
    g = torch.Generator().manual_seed(3)
    err = bench.equivariance_error(model, inp, bench.random_orthogonal(True, generator=g))
    assert err["dx"]["max_rel"] < 1e-12 and err["v"]["max_rel"] < 1e-12


def test_analysis_and_evaluation_edge_cases_and_forcefield_nograd():
    t = np.array([0.0, 10.0, 20.0])
    assert first_zero(t, np.array([-0.5, -0.2, 0.1])) is None
    assert first_zero(t, np.array([0.0, -0.2, 0.1])) == 0.0
    assert first_zero(t, np.array([1.0, -1.0, 0.0])) == pytest.approx(5.0)

    r = np.linspace(1.0, 5.0, 10)
    g_mono = np.linspace(0.0, 2.0, 10)
    assert analysis.first_minimum_after_peak(r, g_mono) == r[-1]

    freq, spec = analysis.vdos(np.zeros(8), dt_fs=10.0)
    assert len(freq) == len(spec) and np.all(spec == 0.0)

    box = torch.eye(3)[None] * 10.0
    pos_pbc = torch.tensor([[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]]])
    dev = max_bond_deviation(pos_pbc, torch.tensor([[0, 1]]), torch.tensor([1.0]), box=box)
    assert dev.item() == pytest.approx(0.0, abs=1e-6)

    ff = EquiFF(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, n_spatial=1)
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]])
    v0 = torch.zeros_like(x0)
    with torch.no_grad():
        f_val = ff.forces(x0, torch.zeros(3, dtype=torch.long), None)
    assert f_val.shape == (1, 3, 3) and torch.isfinite(f_val).all()
    vr = verlet_rollout(ff, x0, v0, torch.zeros(3, dtype=torch.long), None, torch.ones(3), n_steps=2, substeps=2, dt_fs=0.5, save_every=5)
    assert vr["positions"] is None and vr["steps_done"] == 2


def test_train_main_populates_mass_scaling_and_chemistry_exclusions(tmp_path, monkeypatch):
    from equitraj import train as train_mod

    data_dir = tmp_path / "mol_data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        yaml.safe_dump({"split": {"train": [0], "val": [0], "test": [0]}, "dt_fs": 1.0}),
        encoding="utf-8",
    )
    n_frames, n_atoms = 16, 3
    rng = np.random.default_rng(42)
    meta = {
        "version": 1,
        "units": {"charge": "e", "distance": "nm"},
        "charges": [0.4, -0.8, 0.4],
        "ff_types": ["H_w", "O_w", "H_w"],
        "molecule_ids": [0, 0, 0],
        "original_indices": [0, 1, 2],
        "bonds": [[0, 1], [1, 2]],
        "constraints": [],
        "drude_parent_map": [],
    }
    h5_path = data_dir / "traj_000.h5"
    with TrajWriter(
        h5_path, n_atoms=n_atoms, masses=[1.008, 15.999, 1.008], elements=["H", "O", "H"],
        attrs={"dt_fs": 1.0, "save_every": 1, "periodic": False, "complete": True},
        particle_metadata=meta,
    ) as w:
        base_nm = np.array([[0.096, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.032, 0.090, 0.0]], dtype=np.float32)
        for f in range(n_frames):
            pos_nm = base_nm + 0.002 * rng.normal(size=base_nm.shape).astype(np.float32)
            vel_nm_ps = 0.2 * rng.normal(size=base_nm.shape).astype(np.float32)
            w.add(pos_nm, vel_nm_ps, np.zeros_like(pos_nm), 0.0, 1.0, np.eye(3, dtype=np.float32) * 2.0, float(f) * 0.001)

    out_dir = tmp_path / "run_out"
    cfg = {
        "seed": 0,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "stride_steps": 1,
        "k": 2,
        "model": {
            "lmax": 1, "channels": 8, "heads": 2, "n_rbf": 8, "cutoff": 4.0,
            "n_spatial": 1, "n_fusion": 1, "n_denoiser": 1, "head": "flow",
            "target_scaling": "mass", "use_charges": True, "n_ff_types": "auto", "max_bond_hops": 2,
            "repulsion_prior": {"enabled": True, "percentile": 5.0},
        },
        "train": {
            "batch_size": 2, "lr": 1e-3, "steps": 1, "warmup": 1, "workers": 0,
            "unroll": [[0, 1]], "unroll_sampling_steps": 2,
            "overlap_penalty": {"weight": 0.05, "percentile": 5.0},
        },
        "eval": {"every": 1, "val_batches": 1, "rollout_starts": 1, "rollout_steps": 2, "sampling_steps": 2},
    }
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    monkeypatch.setattr(sys, "argv", ["equitraj.train", "--config", str(cfg_path), "--device", "cpu", "--no_tensorboard"])
    train_mod.main()

    ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert ck["step"] == 1
    assert float(ck["model"]["mass_dx_std"]) != pytest.approx(1.0)
    assert float(ck["model"]["mass_v_std"]) != pytest.approx(1.0)
    assert torch.all(ck["model"]["prior.rmin"] == 0.0)
