"""Regression tests for core physics, dtype, PBC, thermostat, and training audit fixes."""

import io
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request

import h5py
import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("e3nn")

from equitraj import align, analysis, cli, pocket
from equitraj.compliance import run_compliance_audit
from equitraj.constraints import DistanceProjector
from equitraj.distill import distillation_loss
from equitraj.evaluation import first_zero
from equitraj.fast import FastRollout
from equitraj.forcefield import EquiFF, verlet_rollout
from equitraj.hybrid import predict_with_uncertainty
from equitraj.model import EquiTraj, remove_com, timestep_embedding
from equitraj.rollout import max_bond_deviation
from equitraj.studio import create_studio_server
from equitraj.thermostat import csvr
from equitraj.topology import write_metadata
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

    # dof == 1 avoids Gamma(0) crash
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

    # remove_com handles float64 masses with float32 vectors
    v = torch.randn(2, 4, 3, dtype=torch.float32)
    m64 = torch.tensor([1.0, 12.0, 16.0, 14.0], dtype=torch.float64)
    vc = remove_com(v, m64)
    assert vc.dtype == torch.float32

    # k > 16 works without IndexError in TemporalAttention.age
    m_bigk = EquiTraj(
        n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, k=20,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow",
        repulsion_prior={"enabled": True, "rmin": 2.5},
    )
    pos = torch.randn(1, 20, 4, 3) * 0.5
    vel = torch.randn(1, 20, 4, 3) * 0.01
    types = torch.zeros(4, dtype=torch.long)
    masses = torch.ones(4)

    # bf16 autocast works with RepulsionPrior index_add
    fast_bf16 = FastRollout(m_bigk, sampling_steps=2, precision="bf16", neighbor="brute")
    dx, vn = fast_bf16.predict(pos, vel, types, None, masses)
    assert torch.isfinite(dx).all() and torch.isfinite(vn).all()

    # geometry_aware=True disables cuda_graphs cleanly with a warning
    m_geom = EquiTraj(
        n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", geometry_aware=True,
    )
    with pytest.warns(RuntimeWarning, match="geometry_aware"):
        fr = FastRollout(m_geom, sampling_steps=2, neighbor="brute", cuda_graphs=True)
    assert fr.cuda_graphs is False
    dx_u, v_u, unc = predict_with_uncertainty(m_geom, pos[:, :3], vel[:, :3], types, None, masses, n_ensembles=2, n_steps=2)
    assert torch.isfinite(dx_u).all() and torch.isfinite(v_u).all() and torch.isfinite(unc["dx_std"]).all()


def test_compliance_single_atom_species_equipartition(monkeypatch):
    from equitraj import compliance

    model = EquiTraj(n_types=2, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, k=3,
                     n_spatial=1, n_fusion=1, n_denoiser=1, head="flow")
    pos = torch.tensor([[[[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [0.0, 2.2, 0.0]]]]).repeat(1, 3, 1, 1)
    vel = torch.tensor([[[[0.004, -0.002, 0.001], [-0.001, 0.002, -0.001], [-0.001, -0.001, 0.0005]]]]).repeat(1, 3, 1, 1)
    masses = torch.tensor([6.94, 16.0, 16.0])
    vel = remove_com(vel.reshape(3, 3, 3), masses).reshape(1, 3, 3, 3)

    fake_ctx = {
        "target_id": "single_li_test",
        "title": "Single Li+ Equipartition Test",
        "model": model,
        "dtype": torch.float32,
        "pos_alg": pos.clone(),
        "vel_alg": vel.clone(),
        "box_alg": None,
        "types_alg": torch.tensor([0, 1, 1], dtype=torch.long),
        "masses_alg": masses.clone(),
        "elements_alg": ["Li", "O", "O"],
        "charges_alg": None,
        "bonds_alg": None,
        "bond_r0": np.zeros(0),
        "excluded_pairs": set(),
        "pos_full": pos.clone(),
        "vel_full": vel.clone(),
        "box_full": None,
        "types_full": torch.tensor([0, 1, 1], dtype=torch.long),
        "masses_full": masses.clone(),
        "elements_full": ["Li", "O", "O"],
        "stride_fs": 10.0,
        "t_ref_K": 300.0,
        "ref_rdf": {},
    }
    monkeypatch.setattr(compliance, "_load_audit_system", lambda tid, precision="float32", seed=0: fake_ctx)
    rep = run_compliance_audit(target_id="single_li_test", rollout_steps=12, sampling_steps=2, seed=0)
    g7 = next(g for g in rep["gates"] if g["id"] == "G7")
    assert "nan" not in g7["detail"].lower() and "inf" not in g7["detail"].lower()
    assert "'Li':" in g7["detail"]


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


def test_posttrain_transport_unwraps_pbc_and_excludes_short_bonds():
    box_l = 12.0
    # 2 bonded atoms (r_bond = 1.21 A < 1.65 A) crossing periodic boundary at x = 12.0 -> 0.1
    t_frames = 6
    raw_unw = np.zeros((t_frames, 3, 3), dtype=np.float64)
    for t in range(t_frames):
        x0 = 11.7 + 0.2 * t
        raw_unw[t, 0] = [x0, 2.0, 2.0]
        raw_unw[t, 1] = [x0 + 1.21, 2.0, 2.0]
        raw_unw[t, 2] = [6.0 - 0.1 * t, 6.0, 6.0]
    wrapped = raw_unw % box_l
    vel = np.gradient(raw_unw, 10.0, axis=0)
    win_wrapped = {
        "positions": wrapped,
        "velocities": vel,
        "types": np.array([0, 1, 0], dtype=np.int64),
        "masses": np.array([12.0, 16.0, 6.94], dtype=np.float64),
        "box": np.eye(3) * box_l,
        "bonds": np.array([[0, 1]], dtype=np.int64),
        "ion_indices": np.array([0, 2], dtype=np.int64),
    }
    win_unwrapped = dict(win_wrapped, positions=raw_unw)
    model = EquiTraj(
        n_types=2, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, k=2,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", target_scaling="mass", stride_fs=10.0,
    )
    model.mass_dx_std.fill_(0.05)
    model.mass_v_std.fill_(0.005)
    res_w = align.posttrain_equitraj_transport(model, [win_wrapped], steps=2, unroll_m=2, sampling_steps=2, seed=7)
    res_u = align.posttrain_equitraj_transport(model, [win_unwrapped], steps=2, unroll_m=2, sampling_steps=2, seed=7)
    assert all(np.isfinite(h["total_loss"]) for h in res_w["history"])
    assert res_w["history"][0]["traj_loss"] == pytest.approx(res_u["history"][0]["traj_loss"], rel=1e-4)
    assert res_w["history"][0]["msd_loss"] == pytest.approx(res_u["history"][0]["msd_loss"], rel=1e-4)


def test_pocket_pbc_unwrapping_and_cli_periodic_rg_and_neighbor():
    box_l = 15.0
    t_frames = 10
    unwrapped = np.zeros((t_frames, 3, 3), dtype=np.float64)
    for t in range(t_frames):
        unwrapped[t, 0] = [14.6 + 0.25 * t, 3.0, 3.0]  # Li+ crosses boundary
        unwrapped[t, 1] = [5.0 - 0.15 * t, 5.0, 5.0]   # Anion
        unwrapped[t, 2] = [14.6 + 0.25 * t + 2.0, 3.0, 3.0]  # Bound O crosses boundary
    wrapped = unwrapped % box_l
    vel = np.zeros_like(unwrapped)

    cne_u = pocket.cluster_nernst_einstein(
        unwrapped, box_l, [0], [1], z_cation=1.0, z_anion=-1.0, dt_frame_fs=20.0, temperature_k=300.0
    )
    cne_w = pocket.cluster_nernst_einstein(
        wrapped, box_l, [0], [1], z_cation=1.0, z_anion=-1.0, dt_frame_fs=20.0, temperature_k=300.0
    )
    assert np.isclose(cne_u["sigma_cne_mS_cm"], cne_w["sigma_cne_mS_cm"], rtol=1e-6)
    assert np.isclose(cne_u["sigma_ne_mS_cm"], cne_w["sigma_ne_mS_cm"], rtol=1e-6)

    hop_u = pocket.solvation_hopping_metrics(unwrapped, vel, box_l, [0], [2], r_shell=2.55)
    hop_w = pocket.solvation_hopping_metrics(wrapped, vel, box_l, [0], [2], r_shell=2.55)
    assert hop_u == hop_w

    rep_periodic = cli.run_pocket_rollout(preset="li_ec_pf6", n_frames=8, substeps=2, dt_fs=1.0, seed=0)
    assert rep_periodic["metrics"]["radius_of_gyration_A"] is None

    parser = cli.build_parser()
    ns_brute = parser.parse_args(["checkpoint", "--neighbor", "brute", "--steps", "2"])
    assert ns_brute.neighbor == "brute"
    ns_dense = parser.parse_args(["checkpoint", "--neighbor", "dense", "--steps", "2"])
    assert ns_dense.neighbor == "dense"


def test_analysis_and_evaluation_edge_cases_and_forcefield_nograd():
    t = np.array([0.0, 10.0, 20.0])
    assert first_zero(t, np.array([-0.5, -0.2, 0.1])) is None
    assert first_zero(t, np.array([0.0, -0.2, 0.1])) == 0.0
    assert first_zero(t, np.array([1.0, -1.0, 0.0])) == pytest.approx(5.0)

    r = np.linspace(1.0, 5.0, 10)
    g_mono = np.linspace(0.0, 2.0, 10)  # peak at final bin
    assert analysis.first_minimum_after_peak(r, g_mono) == r[-1]

    freq, spec = analysis.vdos(np.zeros(8), dt_fs=10.0)
    assert len(freq) == len(spec) and np.all(spec == 0.0)

    # Periodic max_bond_deviation wraps across box
    box = torch.eye(3)[None] * 10.0
    pos_pbc = torch.tensor([[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]]])
    dev = max_bond_deviation(pos_pbc, torch.tensor([[0, 1]]), torch.tensor([1.0]), box=box)
    assert dev.item() == pytest.approx(0.0, abs=1e-6)

    # EquiFF.forces works inside torch.no_grad() and verlet_rollout handles save_every > n_steps
    ff = EquiFF(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, n_spatial=1)
    x0 = torch.tensor([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [0.0, 2.5, 0.0]]])
    v0 = torch.zeros_like(x0)
    with torch.no_grad():
        f_val = ff.forces(x0, torch.zeros(3, dtype=torch.long), None)
    assert f_val.shape == (1, 3, 3) and torch.isfinite(f_val).all()
    vr = verlet_rollout(ff, x0, v0, torch.zeros(3, dtype=torch.long), None, torch.ones(3), n_steps=2, substeps=2, dt_fs=0.5, save_every=5)
    assert vr["positions"] is None and vr["steps_done"] == 2


def test_surrogate_query_preloaded_and_studio_bad_json_and_threads(monkeypatch):
    from lab import surrogate_query as sq

    ckpt_path = Path(__file__).resolve().parents[1] / "checkpoints" / "argon_flow_v0.pt"
    if ckpt_path.exists():
        model, ck = sq.load_release(ckpt_path, "cpu")
        orig_plan = sq.plan_query

        def fake_plan(q):
            p = orig_plan(q)
            p["checkpoint_present"] = False
            return p

        monkeypatch.setattr(sq, "plan_query", fake_plan)
        rep = sq.run_query({"system": "argon", "n_steps": 1, "sampling_steps": 1, "seed": 0}, model_and_ckpt=(model, ck))
        assert rep["status"] == "ok"

    # Studio server returns HTTP 400 on invalid JSON instead of 500 or socket crash
    srv = create_studio_server("127.0.0.1", 0)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/rollout",
            data=b"{not-valid-json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, timeout=5)
        assert exc_info.value.code == 400
    finally:
        srv.shutdown()
        srv.server_close()


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
    # Bonded H-O (0.96 A) and 1-3 H-H (~1.51 A) are all excluded, so no nonbonded pairs exist in 1 water molecule -> rmin == 0
    assert torch.all(ck["model"]["prior.rmin"] == 0.0)
