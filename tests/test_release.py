"""Release checkpoints: smoke test, SHA-256 integrity, attention_version warnings and export."""

import hashlib
import os
from pathlib import Path
import sys

import pytest
import torch

CKPT = "checkpoints/argon_flow_v0.pt"
CKPT_V1 = "checkpoints/argon_flow_v1_cpu_reduced.pt"
CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def verify_sha256_files(directory: Path) -> list[str]:
    """Verify every *.sha256 file in `directory` against its target binary file."""
    sha_files = sorted(directory.glob("*.sha256"))
    assert sha_files, f"No .sha256 files found in {directory}"
    checked = []
    for sha_path in sha_files:
        line = sha_path.read_text(encoding="utf-8").strip()
        parts = line.split()
        assert len(parts) == 2, f"Malformed sha256 line in {sha_path}: {line!r}"
        expected_hex, filename = parts[0], parts[1].lstrip("*")
        target = directory / filename
        assert target.is_file(), f"{sha_path.name} references missing file {target}"
        actual_hex = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual_hex == expected_hex, (
            f"SHA-256 mismatch for {filename}: expected {expected_hex}, got {actual_hex}"
        )
        checked.append(filename)
    return checked


def test_checkpoint_sha256_files_match(tmp_path):
    checked = verify_sha256_files(CKPT_DIR)
    assert "argon_flow_v0.pt" in checked
    assert "argon_flow_v1_cpu_reduced.pt" in checked

    # Verify tamper detection fails immediately
    fake_pt = tmp_path / "dummy.pt"
    fake_pt.write_bytes(b"original-weights")
    good_hex = hashlib.sha256(b"original-weights").hexdigest()
    (tmp_path / "dummy.pt.sha256").write_text(f"{good_hex}  dummy.pt\n", encoding="utf-8")
    assert verify_sha256_files(tmp_path) == ["dummy.pt"]

    fake_pt.write_bytes(b"corrupted-weights")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        verify_sha256_files(tmp_path)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="release checkpoint not present")
def test_release_checkpoint_loads_and_steps():
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")  # demo.py imports equitraj.plots
    sys.path.insert(0, "scripts")
    from demo import load_release  # noqa: PLC0415
    from equitraj.rollout import rollout  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)  # other tests switch to float64 and may leave it on
    model, ck = load_release(CKPT, "cpu")
    d = ck["demo"]
    out = rollout(model, d["pos"][None], d["vel"][None], ck["types"], d["box"][None], ck["masses"], 2, sampling_steps=2)
    assert out["positions"].shape[0] == 2
    assert torch.isfinite(out["positions"]).all()


@pytest.mark.skipif(not os.path.exists(CKPT), reason="release checkpoint not present")
def test_v0_checkpoint_loads_in_v0_mode_without_warning(capsys):
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, "scripts")
    from demo import load_release as demo_load_release  # noqa: PLC0415
    from equitraj.api import load_release as api_load_release  # noqa: PLC0415
    from equitraj.layers import _WARNED_LEGACY_ATTENTION  # noqa: PLC0415
    from lab.surrogate_query import load_release as sq_load_release  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)
    _WARNED_LEGACY_ATTENTION.clear()
    capsys.readouterr()

    for loader in (demo_load_release, api_load_release, sq_load_release):
        model, _ = loader(CKPT, "cpu")
        assert model.attention_version == "v0"
        assert all(blk.attn.attention_version == "v0" for blk in model.spatial)
        assert all(blk.attn.attention_version == "v0" for blk in model.fusion)
        assert all(blk.attn.attention_version == "v0" for blk in model.denoiser.blocks)
    assert capsys.readouterr().err == ""


def test_unknown_attention_version_warns_once_and_falls_back(capsys):
    from equitraj.layers import ATTENTION_VERSION, _WARNED_LEGACY_ATTENTION, resolve_attention_version  # noqa: PLC0415

    _WARNED_LEGACY_ATTENTION.clear()
    capsys.readouterr()

    ver1 = resolve_attention_version({"attention_version": "v99"}, source="mystery_ckpt.pt")
    err1 = capsys.readouterr().err
    assert ver1 == ATTENTION_VERSION
    assert err1.count("WARNING:") == 1
    assert "v99" in err1

    ver2 = resolve_attention_version({"attention_version": "v99"}, source="mystery_ckpt.pt")
    assert ver2 == ATTENTION_VERSION
    assert capsys.readouterr().err == ""


@pytest.mark.skipif(
    not (os.path.exists(CKPT) and os.path.exists(CKPT_V1)),
    reason="release and reduced v1 checkpoints not present",
)
def test_v0_mode_reproduces_tag_v0_1_0_golden_and_v1_unchanged():
    import json

    pytest.importorskip("e3nn")
    from equitraj.api import FrameSplitPredictor, load_release  # noqa: PLC0415

    fixture_path = Path(__file__).resolve().parent / "fixtures" / "argon_flow_v0_onestep_v0_1_0.json"
    fix = json.loads(fixture_path.read_text(encoding="utf-8"))
    atom_idx = fix["atom_indices"]
    seed = int(fix["seed"])
    n_steps = int(fix["sampling_steps"])

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        # 1. argon_flow_v0.pt on main in v0 mode matches tag v0.1.0 within < 1e-5 absolute
        m_v0, ck0 = load_release(CKPT, "cpu")
        assert m_v0.attention_version == "v0"
        d0 = ck0["demo"]
        pos0, vel0, box0 = d0["pos"][None].float(), d0["vel"][None].float(), d0["box"][None].float()
        types0, masses0 = ck0["types"].long(), ck0["masses"].float()
        noise0 = m_v0.com_free_noise(1, types0, masses0, generator=torch.Generator().manual_seed(seed), like=vel0)
        dx0, vn0 = m_v0.predict(pos0, vel0, types0, box0, masses0, n_steps=n_steps, noise=noise0)

        ref0 = fix["v0_tag_reference"]
        dx0_ref = torch.tensor(ref0["dx_subset_A"], dtype=torch.float32)
        vn0_ref = torch.tensor(ref0["v_new_subset_A_per_fs"], dtype=torch.float32)
        assert (dx0[0, atom_idx] - dx0_ref).abs().max().item() < 1e-5
        assert (vn0[0, atom_idx] - vn0_ref).abs().max().item() < 1e-5
        assert abs(dx0.pow(2).mean().sqrt().item() - ref0["dx_rms_all_A"]) < 1e-5
        assert abs(vn0.pow(2).mean().sqrt().item() - ref0["v_new_rms_all_A_per_fs"]) < 1e-5

        dx0_fast, vn0_fast = FrameSplitPredictor(m_v0).predict(
            pos0, vel0, types0, box0, masses0, noise=noise0, n_steps=n_steps
        )
        assert (dx0_fast - dx0).abs().max().item() < 1e-5
        assert (vn0_fast - vn0).abs().max().item() < 1e-5

        # 2. argon_flow_v1_cpu_reduced.pt on main stays in v1 mode with unchanged outputs
        m_v1, ck1 = load_release(CKPT_V1, "cpu")
        assert m_v1.attention_version == "v1"
        d1 = ck1["demo"]
        pos1, vel1, box1 = d1["pos"][None].float(), d1["vel"][None].float(), d1["box"][None].float()
        types1, masses1 = ck1["types"].long(), ck1["masses"].float()
        noise1 = m_v1.com_free_noise(1, types1, masses1, generator=torch.Generator().manual_seed(seed), like=vel1)
        dx1, vn1 = m_v1.predict(pos1, vel1, types1, box1, masses1, n_steps=n_steps, noise=noise1)

        ref1 = fix["v1_reduced_reference"]
        dx1_ref = torch.tensor(ref1["dx_subset_A"], dtype=torch.float32)
        vn1_ref = torch.tensor(ref1["v_new_subset_A_per_fs"], dtype=torch.float32)
        assert (dx1[0, atom_idx] - dx1_ref).abs().max().item() < 1e-6
        assert (vn1[0, atom_idx] - vn1_ref).abs().max().item() < 1e-6
    finally:
        torch.set_default_dtype(prev)


@pytest.mark.skipif(not os.path.exists(CKPT_V1), reason="reduced v1 checkpoint not present")
def test_reduced_v1_checkpoint_does_not_warn(capsys):
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, "scripts")
    from demo import load_release  # noqa: PLC0415
    from equitraj.layers import _WARNED_LEGACY_ATTENTION  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)
    _WARNED_LEGACY_ATTENTION.clear()
    capsys.readouterr()
    load_release(CKPT_V1, "cpu")
    assert capsys.readouterr().err == ""


def _tiny_training_run(tmp_path, attention_version=None):
    """Tiny synthetic test dataset + training checkpoint for export_ckpt.main()."""
    import h5py
    import numpy as np
    import yaml

    from equitraj.model import EquiTraj  # noqa: PLC0415

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "config.yaml").write_text(
        yaml.safe_dump({"split": {"train": [0], "val": [0], "test": [0]}, "dt_fs": 25.0})
    )
    rng = np.random.default_rng(0)
    with h5py.File(data_dir / "traj_000.h5", "w") as h:
        h.attrs["dt_fs"] = 25.0
        h.attrs["save_every"] = 1
        h.attrs["periodic"] = True
        h.attrs["complete"] = True
        h.create_dataset("elements", data=np.array([b"Ar"] * 4))
        h.create_dataset("masses", data=np.full(4, 39.948, dtype=np.float32))
        h.create_dataset("time", data=(np.arange(120, dtype=np.float64) * 25.0 / 1000.0))
        h.create_dataset("positions", data=rng.uniform(0.0, 1.0, size=(120, 4, 3)).astype(np.float32))
        h.create_dataset("velocities", data=rng.normal(0.0, 0.1, size=(120, 4, 3)).astype(np.float32))
        h.create_dataset("box", data=np.tile(np.eye(3, dtype=np.float32) * 1.2, (120, 1, 1)))

    model_cfg = {
        "head": "regression",
        "lmax": 1,
        "channels": 8,
        "heads": 2,
        "n_rbf": 4,
        "cutoff": 4.0,
        "n_spatial": 1,
        "n_fusion": 1,
        "n_denoiser": 1,
    }
    tiny = EquiTraj(n_types=1, k=2, stride_fs=50.0, **model_cfg)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    ck = {
        "step": 50,
        "model": tiny.state_dict(),
        "elements": ["Ar"],
        "cfg": {"name": "tiny", "data_dir": str(data_dir), "stride_steps": 2, "k": 2, "model": model_cfg},
    }
    if attention_version is not None:
        ck["attention_version"] = attention_version
    torch.save(ck, run_dir / "best.pt")
    return run_dir


def _export(monkeypatch, run_dir, out_pt, *extra):
    sys.path.insert(0, "scripts")
    import export_ckpt  # noqa: PLC0415

    argv = ["export_ckpt.py", "--run", str(run_dir), "--ckpt", "best.pt", "--out", str(out_pt), *extra]
    monkeypatch.setattr(sys, "argv", argv)
    export_ckpt.main()


def test_export_without_field_or_flag_errors(tmp_path, monkeypatch):
    pytest.importorskip("e3nn")
    pytest.importorskip("h5py")
    torch.set_default_dtype(torch.float32)
    run_dir = _tiny_training_run(tmp_path)
    out_pt = tmp_path / "exported.pt"
    with pytest.raises(SystemExit) as exc:
        _export(monkeypatch, run_dir, out_pt)
    assert "--attention-version" in str(exc.value.code)
    assert not out_pt.exists()


def test_export_with_flag_v0_writes_v0(tmp_path, monkeypatch, capsys):
    pytest.importorskip("e3nn")
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, "scripts")
    from demo import load_release  # noqa: PLC0415
    from equitraj.layers import _WARNED_LEGACY_ATTENTION  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)
    run_dir = _tiny_training_run(tmp_path)
    out_pt = tmp_path / "exported_v0.pt"
    _export(monkeypatch, run_dir, out_pt, "--attention-version", "v0")
    assert torch.load(out_pt, map_location="cpu", weights_only=True)["attention_version"] == "v0"

    _WARNED_LEGACY_ATTENTION.clear()
    capsys.readouterr()
    model, _ = load_release(str(out_pt), "cpu")
    assert model.attention_version == "v0"
    assert capsys.readouterr().err == ""


def test_export_uses_training_checkpoint_field(tmp_path, monkeypatch, capsys):
    pytest.importorskip("e3nn")
    pytest.importorskip("h5py")
    pytest.importorskip("matplotlib")
    sys.path.insert(0, "scripts")
    from demo import load_release  # noqa: PLC0415
    from equitraj.layers import ATTENTION_VERSION  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)
    run_dir = _tiny_training_run(tmp_path, attention_version=ATTENTION_VERSION)
    out_pt = tmp_path / "exported_v1.pt"
    _export(monkeypatch, run_dir, out_pt)
    assert torch.load(out_pt, map_location="cpu", weights_only=True)["attention_version"] == ATTENTION_VERSION
    capsys.readouterr()
    load_release(str(out_pt), "cpu")
    assert capsys.readouterr().err == ""


def test_train_checkpoints_store_attention_version(tmp_path, monkeypatch):
    pytest.importorskip("e3nn")
    import yaml

    from equitraj import train  # noqa: PLC0415
    from equitraj.layers import ATTENTION_VERSION  # noqa: PLC0415
    from test_resume import _config, _write_traj  # noqa: PLC0415

    data_dir, out_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    _write_traj(data_dir / "traj_000.h5", 24)
    _write_traj(data_dir / "traj_001.h5", 8)
    (data_dir / "config.yaml").write_text(yaml.safe_dump({"split": {"train": [0], "val": [1], "test": []}}))
    cfg = _config(data_dir, out_dir, steps=1)
    cfg["eval"]["every"] = 1  # also write best.pt, best_stable.pt and a snapshot
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(cfg))

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--device", "cpu", "--no_tensorboard"])
        train.main()
    finally:
        torch.set_default_dtype(previous_dtype)

    for path in [out_dir / "ckpt.pt", out_dir / "best.pt", out_dir / "best_stable.pt",
                 out_dir / "snapshots" / "step000001.pt"]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["attention_version"] == ATTENTION_VERSION, path


def test_invalid_attention_version_rejected():
    pytest.importorskip("e3nn")
    from equitraj.layers import GraphAttention  # noqa: PLC0415
    from equitraj.model import EquiTraj  # noqa: PLC0415

    with pytest.raises(ValueError, match="attention_version"):
        GraphAttention(lmax=1, c=8, heads=2, n_rbf=4, cutoff=4.0, attention_version="v2")
    with pytest.raises(ValueError, match="attention_version"):
        EquiTraj(n_types=1, lmax=1, channels=8, heads=2, n_rbf=4, cutoff=4.0, attention_version="v2")


def test_committed_v0_and_reduced_eval_metrics_have_energy_kinematics_and_crash_reasons():
    import json
    import math

    root = Path(__file__).resolve().parents[1]
    targets = [
        (root / "results/argon/B_S3a_seed1/eval_step10000_cpu_3x1000/metrics.json", "v0", 1000, 3),
        (root / "results/argon/cpu_reduced/v0/metrics.json", "v0", 100, 1),
        (root / "results/argon/cpu_reduced/v1/metrics.json", "v1", 100, 1),
    ]
    for path, exp_ver, exp_steps, exp_roll in targets:
        m = json.loads(path.read_text(encoding="utf-8"))
        assert m["attention_version"] == exp_ver
        assert m["steps"] == exp_steps and m["n_rollouts"] == exp_roll
        reasons = m["stability"]["crash_reasons"]
        assert "frozen" in reasons and "temperature" in reasons
        assert sum(reasons.values()) == m["stability"]["crashed"]
        en = m["energy"]
        for key in ("H_drift_kJ_mol_ns_dof_mean", "U_model_minus_gt_per_atom", "K_model_minus_gt_per_atom"):
            assert math.isfinite(en[key]), (path, key)
        kin = m["kinematics"]
        for arm in ("model", "ground_truth"):
            for key in ("trapezoid_residual", "velocity_temperature_K", "displacement_temperature_K",
                        "temperature_rel_gap", "green_kubo_D_cm2_s", "einstein_D_cm2_s"):
                assert math.isfinite(kin[arm][key]), (path, arm, key)


def test_docs_do_not_instruct_checkout_v0_1_0():
    root = Path(__file__).resolve().parents[1]
    for rel in ("README.md", "checkpoints/README.md"):
        txt = (root / rel).read_text(encoding="utf-8").lower()
        assert "check out tag `v0.1.0`" not in txt
        assert "checkout v0.1.0" not in txt
