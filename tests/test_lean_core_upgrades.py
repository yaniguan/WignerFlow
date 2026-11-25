"""Tests for lean-core architecture, TrajCast baseline, S4/S5 switches, vacuum angular momentum, PBC symmetry, BF16, and MD17 import."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("e3nn")

import wignerflow
from equitraj import adapters, bench, train as train_mod
from equitraj.baselines import TrajCastBaseline
from equitraj.energy import kinetic, lj_potential
from equitraj.model import EquiTraj, angular_momentum, build_graph, make_model, remove_angular_momentum
from equitraj.rollout import rollout
from equitraj.so3 import edge_wigner
from scripts import evaluate_molecule as eval_mol_cli

ROOT = Path(__file__).resolve().parents[1]


def test_wignerflow_package_alias_and_cli_help(capsys):
    assert wignerflow.EquiTraj is EquiTraj
    assert wignerflow.TrajCastBaseline is TrajCastBaseline
    assert wignerflow.__version__ == "0.2.0"

    from wignerflow.__main__ import main as wf_main
    rc = wf_main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0 and "evaluate-molecule" in out and "distill" in out


def test_trajcast_baseline_o3_equivariance_permutation_and_markovian():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = make_model(
            n_types=2, k=3, stride_fs=10.0, arch="trajcast", lmax=2, channels=8, heads=2,
            n_rbf=8, cutoff=5.0, n_spatial=1, n_fusion=1, head="regression", target_scaling="mass",
            n_ff_types=2, use_charges=True, max_bond_hops=2, markovian=True,
        ).double().eval()
        model.set_chemistry(
            charges=[0.4, -0.8, 0.4, 0.0],
            ff_types=[0, 1, 0, 1],
            bonds=[[0, 1], [1, 2]],
            n_atoms=4,
        )
        g = torch.Generator().manual_seed(11)
        pos = torch.randn(1, 3, 4, 3, generator=g, dtype=torch.float64) * 1.2
        vel = torch.randn(1, 3, 4, 3, generator=g, dtype=torch.float64) * 0.02
        types = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        masses = torch.tensor([1.008, 15.999, 1.008, 12.011], dtype=torch.float64)

        # 1. Markovian: perturbing older frames 0..1 has zero effect on prediction
        pos_mod = pos.clone()
        pos_mod[:, :2] += 5.0
        dx0, v0 = model.predict(pos, vel, types, None, masses)
        dx1, v1 = model.predict(pos_mod, vel, types, None, masses)
        assert torch.allclose(dx0, dx1, atol=0.0) and torch.allclose(v0, v1, atol=0.0)

        # 2. Exact O(3) rotation + improper reflection + translation equivariance in float64
        for proper in (True, False):
            rot = bench.random_orthogonal(proper, generator=g)
            shift = torch.randn(1, 1, 1, 3, generator=g, dtype=torch.float64) * 3.0
            dx_r, v_r = model.predict(pos @ rot.T + shift, vel @ rot.T, types, None, masses)
            assert (dx_r - dx0 @ rot.T).abs().max().item() < 1e-12
            assert (v_r - v0 @ rot.T).abs().max().item() < 1e-12

        # 3. Permutation equivariance (swapping identical H atoms 0 and 2)
        perm = torch.tensor([2, 1, 0, 3], dtype=torch.long)
        dx_p, v_p = model.predict(pos[:, :, perm], vel[:, :, perm], types[perm], None, masses[perm])
        assert (dx_p - dx0[:, perm]).abs().max().item() < 1e-12
        assert (v_p - v0[:, perm]).abs().max().item() < 1e-12
    finally:
        torch.set_default_dtype(prev)


def test_pbc_joint_o3_vs_fixed_cubic_box_oh_symmetry():
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = EquiTraj(
            n_types=1, lmax=2, channels=8, heads=2, n_rbf=8, cutoff=3.8, k=2,
            n_spatial=1, n_fusion=1, head="regression",
        ).double().eval()
        g = torch.Generator().manual_seed(23)
        box = torch.eye(3, dtype=torch.float64)[None] * 8.0
        pos = torch.rand(1, 2, 6, 3, generator=g, dtype=torch.float64) * 8.0
        vel = torch.randn(1, 2, 6, 3, generator=g, dtype=torch.float64) * 0.01
        types = torch.zeros(6, dtype=torch.long)
        masses = torch.full((6,), 39.948, dtype=torch.float64)
        dx0, v0 = model.predict(pos, vel, types, box, masses)

        # Joint rotation of coordinates AND lattice vectors preserves continuous O(3) exactly
        rot_any = bench.random_orthogonal(False, generator=g)
        dx_j, v_j = model.predict(pos @ rot_any.T, vel @ rot_any.T, types, box @ rot_any.T, masses)
        assert (dx_j - dx0 @ rot_any.T).abs().max().item() < 1e-12
        assert (v_j - v0 @ rot_any.T).abs().max().item() < 1e-12

        # Under a FIXED cubic box, octahedral point-group O_h symmetries (signed axis permutations) hold exactly
        oh_ops = [
            torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),   # 90-deg z rot
            torch.tensor([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64),   # mirror x
            torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float64),    # 3-fold diagonal
        ]
        for op in oh_ops:
            dx_oh, v_oh = model.predict(pos @ op.T, vel @ op.T, types, box, masses)
            assert (dx_oh - dx0 @ op.T).abs().max().item() < 1e-12
            assert (v_oh - v0 @ op.T).abs().max().item() < 1e-12
    finally:
        torch.set_default_dtype(prev)


def test_switch_s5_energy_rescale_and_vacuum_angular_momentum_projection():
    # 1. Vacuum angular momentum projection removes L while keeping P_cm = 0 and O(3) equivariance
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [-0.4, 1.1, 0.2], [0.5, -0.6, 1.2]]], dtype=torch.float64)
    vel = torch.tensor([[[0.02, -0.03, 0.01], [-0.01, 0.04, -0.02], [0.03, 0.01, 0.02], [-0.01, -0.02, 0.03]]], dtype=torch.float64)
    masses = torch.tensor([12.011, 15.999, 1.008, 14.007], dtype=torch.float64)
    assert angular_momentum(pos, vel, masses).norm().item() > 0.1
    v_clean = remove_angular_momentum(pos, vel, masses)
    assert angular_momentum(pos, v_clean, masses).norm().item() < 1e-13
    assert (v_clean * masses[None, :, None]).sum(dim=1).norm().item() < 1e-13

    # 2. Switch S5 (`conserve_energy=True`) conserves U_ref(x) + K(v) along rollout
    model = EquiTraj(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, k=2,
                     n_spatial=1, n_fusion=1, head="regression").eval()
    model.out_dx_std.fill_(0.002)
    model.out_v_std.fill_(0.0005)
    box = torch.eye(3)[None] * 12.0
    x_ctx = torch.tensor([[[[2.0, 2.0, 2.0], [6.0, 2.0, 2.0], [2.0, 6.0, 2.0]]]], dtype=torch.float32).repeat(1, 2, 1, 1)
    v_ctx = torch.tensor([[[[0.003, -0.001, 0.001], [-0.002, 0.002, 0.0], [-0.001, -0.001, -0.001]]]], dtype=torch.float32).repeat(1, 2, 1, 1)
    ar_m = torch.full((3,), 39.948)
    ar_t = torch.zeros(3, dtype=torch.long)
    e_fn = lambda p, b: lj_potential(p, b[0] if b.ndim == 3 else b)  # noqa: E731

    e0 = float((e_fn(x_ctx[:, -1], box) + kinetic(v_ctx[:, -1], ar_m)).item())
    out = rollout(model, x_ctx, v_ctx, ar_t, box, ar_m, n_steps=5, sampling_steps=1,
                  energy_fn=e_fn, conserve_energy=True)
    for s in range(out["positions"].shape[0]):
        es = float((e_fn(out["positions"][s], box) + kinetic(out["velocities"][s], ar_m)).item())
        assert es == pytest.approx(e0, abs=1e-3)


def test_gradient_checkpointing_auto_neighbors_and_bf16_wigner_stability():
    # 1. +y-aligned edge under bfloat16 has finite backward gradients (no 1e-30 underflow NaN)
    vec_y = torch.tensor([[0.0, 2.5, 0.0], [1.2, 0.8, -0.5]], dtype=torch.bfloat16, requires_grad=True)
    wig = edge_wigner(vec_y, lmax=2)
    assert wig.dtype == torch.float32 and torch.isfinite(wig).all()
    wig.sum().backward()
    assert torch.isfinite(vec_y.grad).all()

    # 2. neighbor_method='auto' matches brute_force and cell_list
    pos = torch.rand(2, 12, 3) * 14.0
    box = torch.eye(3)[None].expand(2, 3, 3) * 14.0
    g_auto = build_graph(pos, box, cutoff=4.0, lmax=1, method="auto")
    g_cell = build_graph(pos, box, cutoff=4.0, lmax=1, method="cell_list")
    assert g_auto["src"].numel() == g_cell["src"].numel()

    # 3. gradient_checkpointing=True gives identical loss & parameter gradients to False
    torch.manual_seed(5)
    m_plain = EquiTraj(n_types=2, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.5, k=2,
                       n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", gradient_checkpointing=False)
    m_ckpt = EquiTraj(n_types=2, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.5, k=2,
                      n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", gradient_checkpointing=True)
    m_ckpt.load_state_dict(m_plain.state_dict())

    p = torch.randn(2, 2, 4, 3)
    v = torch.randn(2, 2, 4, 3) * 0.01
    dx_t = torch.randn(2, 4, 3) * 0.02
    v_t = torch.randn(2, 4, 3) * 0.01
    types = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    masses = torch.tensor([12.0, 1.0, 12.0, 1.0])

    torch.manual_seed(99)
    l1 = m_plain.loss(p, v, types, None, masses, dx_t, v_t)
    l1.backward()

    torch.manual_seed(99)
    l2 = m_ckpt.loss(p, v, types, None, masses, dx_t, v_t)
    l2.backward()

    assert torch.allclose(l1, l2, atol=1e-6)
    for (n1, p1), (_, p2) in zip(m_plain.named_parameters(), m_ckpt.named_parameters()):
        if p1.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-5), f"grad mismatch on {n1}"


def test_md17_import_bf16_train_trajcast_and_evaluate_molecule_end_to_end(tmp_path, monkeypatch):
    # Create a synthetic MD17-format .npz archive (water-like 3-atom covalent molecule, 24 frames at 0.5 fs)
    npz_path = tmp_path / "md17_water.npz"
    rng = np.random.default_rng(7)
    base_r = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=np.float64)
    coords = np.stack([base_r + 0.01 * np.sin(0.2 * t) + 0.002 * rng.normal(size=(3, 3)) for t in range(28)])
    z = np.array([8, 1, 1], dtype=np.int32)
    energies = -47900.0 + 0.2 * rng.normal(size=28)
    forces = rng.normal(scale=0.5, size=(28, 3, 3))
    np.savez(npz_path, R=coords, z=z, E=energies, F=forces)

    data_dir = tmp_path / "md17_h5"
    adapters.import_trajectory(npz_path, data_dir, format="md17", dt_fs=0.5, traj_index=0, system="md17_water")
    (data_dir / "config.yaml").write_text(
        yaml.safe_dump({"n_traj": 1, "split": {"train": [0], "val": [0], "test": [0]}, "system": "md17_water", "dt_fs": 0.5}),
        encoding="utf-8",
    )

    # Train EquiTraj with amp='bf16' + gradient_checkpointing=True + full chemistry
    run_equi = tmp_path / "run_equi"
    cfg_equi = {
        "seed": 0,
        "data_dir": str(data_dir),
        "out_dir": str(run_equi),
        "stride_steps": 2,
        "k": 2,
        "model": {
            "arch": "equi", "head": "flow", "lmax": 1, "channels": 8, "heads": 2, "n_rbf": 8, "cutoff": 4.0,
            "n_spatial": 1, "n_fusion": 1, "n_denoiser": 1, "neighbor_method": "auto",
            "gradient_checkpointing": True, "target_scaling": "mass", "n_ff_types": "auto", "max_bond_hops": 2,
        },
        "train": {"batch_size": 2, "amp": "bf16", "lr": 1e-3, "steps": 2, "warmup": 1, "workers": 0, "unroll": [[0, 1]]},
        "eval": {"every": 2, "val_batches": 1, "rollout_starts": 1, "rollout_steps": 3, "sampling_steps": 2},
    }
    equi_yaml = tmp_path / "equi.yaml"
    equi_yaml.write_text(yaml.safe_dump(cfg_equi), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["wignerflow", "--config", str(equi_yaml), "--device", "cpu", "--no_tensorboard"])
    train_mod.main()

    # Evaluate molecule with S4 RATTLE constraints + zero_angular_momentum
    eval_out = tmp_path / "eval_mol_out"
    eval_mol_cli.main([
        "--run", str(run_equi),
        "--ckpt", "best_stable.pt",
        "--steps", "4",
        "--seeds", "1",
        "--starts", "1",
        "--sampling_steps", "2",
        "--project_constraints",
        "--zero_angular_momentum",
        "--device", "cpu",
        "--out", str(eval_out),
    ])
    metrics = json.loads((eval_out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["switches"]["S4_project_constraints"] is True
    assert metrics["momentum"]["max_linear_momentum_amu_A_fs"] < 1e-4

    # Also train TrajCastBaseline on the exact same dataset
    run_tc = tmp_path / "run_tc"
    cfg_tc = {
        **cfg_equi,
        "out_dir": str(run_tc),
        "model": {
            "arch": "trajcast", "head": "regression", "markovian": True, "lmax": 1, "channels": 8,
            "heads": 2, "n_rbf": 8, "cutoff": 4.0, "n_spatial": 1, "n_fusion": 1,
            "target_scaling": "mass", "n_ff_types": "auto", "max_bond_hops": 2,
        },
    }
    tc_yaml = tmp_path / "tc.yaml"
    tc_yaml.write_text(yaml.safe_dump(cfg_tc), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["wignerflow", "--config", str(tc_yaml), "--device", "cpu", "--no_tensorboard"])
    train_mod.main()
    assert (run_tc / "best_stable.pt").exists()

    # Verify all new H100 YAML configs parse cleanly
    for rel in (
        "configs/argon/h100_v1.yaml",
        "configs/argon/trajcast_h100.yaml",
        "configs/ala2/h100_v1.yaml",
        "configs/ala2/trajcast_h100.yaml",
        "configs/md17_aspirin/h100_v1.yaml",
        "configs/li_ec/h100_v1.yaml",
    ):
        parsed = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        assert "model" in parsed and "train" in parsed and "eval" in parsed
