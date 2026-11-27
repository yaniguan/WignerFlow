"""Regression tests for standalone checkpoint evaluation, make_model scripts, configs, and packaging (#94)."""

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("e3nn")

from equitraj import evaluation, multiseed, train as train_mod
from equitraj.dataset import TrajWindows
from equitraj.topology import write_metadata
from equitraj.trajfile import TrajWriter
from scripts import demo as demo_cli, evaluate as evaluate_cli, export_ckpt as export_cli

ROOT = Path(__file__).resolve().parents[1]


def _make_mini_dataset(data_dir: Path, *, n_frames: int = 18, periodic: bool = True, constrained: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "config.yaml").write_text(
        yaml.safe_dump({"split": {"train": [0], "val": [0], "test": [0]}, "dt_fs": 5.0, "system": "argon" if periodic else "mol"}),
        encoding="utf-8",
    )
    rng = np.random.default_rng(0)
    meta = {
        "version": 1,
        "units": {"charge": "e", "distance": "nm"},
        "charges": [0.4, -0.8, 0.4],
        "ff_types": ["H_a", "O_a", "H_a"],
        "molecule_ids": [0, 0, 0],
        "original_indices": [0, 1, 2],
        "bonds": [[0, 1], [1, 2]],
        "constraints": [[0, 1, 0.096], [1, 2, 0.096]] if constrained else [],
        "drude_parent_map": [],
    }
    h5_path = data_dir / "traj_000.h5"
    attrs = {
        "dt_fs": 5.0,
        "save_every": 1,
        "periodic": periodic,
        "complete": True,
        "n_constraints": 2 if constrained else 0,
    }
    with TrajWriter(
        h5_path,
        n_atoms=3,
        masses=[1.008, 15.999, 1.008],
        elements=["H", "O", "H"],
        attrs=attrs,
        particle_metadata=meta,
    ) as w:
        base = np.array([[0.30, 0.30, 0.30], [0.40, 0.30, 0.30], [0.30, 0.40, 0.30]], dtype=np.float32)
        for f in range(n_frames):
            pos_nm = base + 0.002 * f + 0.001 * rng.normal(size=(3, 3)).astype(np.float32)
            vel_nm_ps = 0.25 * rng.normal(size=(3, 3)).astype(np.float32)
            w.add(pos_nm, vel_nm_ps, np.zeros((3, 3), dtype=np.float32), 0.0, 1.0, np.eye(3, dtype=np.float32) * 1.6, f * 0.005)
    return data_dir


def test_standalone_evaluate_plain_arch_and_float64_guard(tmp_path, capsys):
    data_dir = _make_mini_dataset(tmp_path / "data", n_frames=18, periodic=True)
    from equitraj.model import make_model

    plain = make_model(n_types=2, k=2, stride_fs=5.0, arch="plain", channels=8, heads=2, n_rbf=8, cutoff=4.5,
                       n_spatial=1, n_fusion=1, head="regression").float().eval()
    plain.out_dx_std.fill_(0.005)
    plain.out_v_std.fill_(0.0002)
    ckpt_file = tmp_path / "standalone_plain.pt"
    torch.save({
        "format": "wignerflow-v0",
        "attention_version": "v1",
        "elements": ["H", "O"],
        "k": 2,
        "stride_steps": 1,
        "stride_fs": 5.0,
        "train_step": 10,
        "model_cfg": {"arch": "plain", "channels": 8, "heads": 2, "n_rbf": 8, "cutoff": 4.5,
                      "n_spatial": 1, "n_fusion": 1, "head": "regression"},
        "model": plain.state_dict(),
    }, ckpt_file)

    out_dir = tmp_path / "eval_out"
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        evaluate_cli.main([
            "--ckpt", str(ckpt_file),
            "--data_dir", str(data_dir),
            "--steps", "2",
            "--seeds", "1",
            "--starts", "1",
            "--sampling_steps", "1",
            "--device", "cpu",
            "--out", str(out_dir),
        ])
        assert torch.get_default_dtype() == torch.float64
    finally:
        torch.set_default_dtype(prev)

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["steps"] == 2 and "v0_verdict" in metrics

    # Training checkpoint with v0 attention triggers check_attention_version warning in evaluation.load_model
    equi = make_model(n_types=2, k=2, stride_fs=5.0, arch="equi", lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.5,
                      n_spatial=1, n_fusion=1, head="regression", n_ff_types=2).float().eval()
    v0_train_pt = tmp_path / "v0_train.pt"
    torch.save({
        "step": 5,
        "attention_version": "v0",
        "elements": ["H", "O"],
        "cfg": {
            "data_dir": str(data_dir),
            "k": 2,
            "stride_steps": 1,
            "model": {"lmax": 1, "channels": 8, "heads": 2, "n_rbf": 8, "cutoff": 4.5,
                      "n_spatial": 1, "n_fusion": 1, "head": "regression", "n_ff_types": "auto"},
        },
        "model": equi.state_dict(),
    }, v0_train_pt)
    capsys.readouterr()
    evaluation.load_model(None, str(v0_train_pt), torch.device("cpu"), split="val")
    err_out = capsys.readouterr().err
    assert "attention_version='v0'" in err_out


def test_train_resolves_auto_ff_types_and_validates_horizon_early(tmp_path, monkeypatch):
    data_dir = _make_mini_dataset(tmp_path / "short_data", n_frames=10, periodic=False, constrained=True)
    out_dir = tmp_path / "run_auto"
    cfg_bad = {
        "seed": 0,
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "stride_steps": 1,
        "k": 2,
        "model": {"lmax": 1, "channels": 8, "heads": 2, "n_rbf": 8, "cutoff": 4.0,
                  "n_spatial": 1, "n_fusion": 1, "head": "regression", "n_ff_types": "auto"},
        "train": {"batch_size": 2, "lr": 1e-3, "steps": 2, "warmup": 1, "workers": 0, "unroll": [[0, 1]]},
        "eval": {"every": 1, "val_batches": 1, "rollout_starts": 1, "rollout_steps": 50, "sampling_steps": 1},
    }
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(yaml.safe_dump(cfg_bad), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["equitraj.train", "--config", str(bad_yaml), "--device", "cpu", "--no_tensorboard"])
    with pytest.raises(ValueError, match="no validation window has 50 steps"):
        train_mod.main()

    # Valid horizon saves integer n_ff_types instead of 'auto', and export_ckpt + demo.load_release work with chemistry
    cfg_ok = dict(cfg_bad, eval={**cfg_bad["eval"], "rollout_steps": 2})
    ok_yaml = tmp_path / "ok.yaml"
    ok_yaml.write_text(yaml.safe_dump(cfg_ok), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["equitraj.train", "--config", str(ok_yaml), "--device", "cpu", "--no_tensorboard"])
    train_mod.main()

    saved_ck = torch.load(out_dir / "ckpt.pt", map_location="cpu", weights_only=False)
    assert isinstance(saved_ck["cfg"]["model"]["n_ff_types"], int)
    assert saved_ck["cfg"]["model"]["n_ff_types"] == 2

    exported_pt = tmp_path / "exported.pt"
    monkeypatch.setattr(sys, "argv", ["export_ckpt.py", "--run", str(out_dir), "--ckpt", "ckpt.pt", "--out", str(exported_pt)])
    export_cli.main()
    loaded_model, loaded_ck = demo_cli.load_release(str(exported_pt), torch.device("cpu"))
    assert "chemistry" in loaded_ck and loaded_model.n_ff_types == 2

    # multiseed.reference uses ds.n_constraints (3*3 - 2 - 3 = 4 DOF instead of 6 DOF -> 1.5x higher T)
    ds_con = TrajWindows(str(data_dir), "test", 1, 2, m=1)
    assert ds_con.n_constraints == 2
    ref_con = multiseed.reference(ds_con, steps=4)
    ds_con.n_constraints = 0
    ref_uncon = multiseed.reference(ds_con, steps=4)
    assert ref_con["T"] == pytest.approx(ref_uncon["T"] * 1.5, rel=1e-5)


def test_configs_horizons_lazy_h5py_and_pyproject_packages(monkeypatch):
    # 1. nacl/proof_cpu.yaml fits inside default 301-frame make_nacl_proof.py output (f = 10 // 5 = 2)
    nacl_cfg = yaml.safe_load((ROOT / "configs" / "nacl" / "proof_cpu.yaml").read_text(encoding="utf-8"))
    k = nacl_cfg["k"]
    for key in ("rollout_steps", "rollout_steps_select"):
        need_frames = (k - 1 + nacl_cfg["eval"][key]) * 2 + 1
        assert need_frames <= 301, f"{key} needs {need_frames} > 301 frames"

    # 2. data/argon.yaml has >= 10,021 frames for 1000-step stride-10 validation, and ala2/aspirin data configs exist
    ar_data = yaml.safe_load((ROOT / "configs" / "data" / "argon.yaml").read_text(encoding="utf-8"))
    ar_frames = int(round(ar_data["production_ps"] * 1000.0 / (ar_data["dt_fs"] * ar_data["save_every"])))
    assert ar_frames >= 10021

    for mol_name, min_ps in (("ala2", 2000.0), ("aspirin", 1000.0)):
        mol_cfg = yaml.safe_load((ROOT / "configs" / "data" / f"{mol_name}.yaml").read_text(encoding="utf-8"))
        assert mol_cfg["system"] == mol_name
        assert mol_cfg["production_ps"] >= min_ps
        assert mol_cfg["dt_fs"] * mol_cfg["save_every"] == pytest.approx(2.0)

    # 3. equitraj.adapters imports cleanly when h5py is absent
    saved_h5 = sys.modules.get("h5py")
    saved_ad = sys.modules.pop("equitraj.adapters", None)
    monkeypatch.setitem(sys.modules, "h5py", None)
    try:
        ad_mod = importlib.import_module("equitraj.adapters")
        unw = ad_mod.unwrap_positions(np.array([[[0.1, 0.0, 0.0]], [[0.9, 0.0, 0.0]]]), np.eye(3))
        assert np.allclose(unw[1, 0, 0], -0.1)
    finally:
        if saved_h5 is not None:
            sys.modules["h5py"] = saved_h5
        else:
            sys.modules.pop("h5py", None)
        if saved_ad is not None:
            sys.modules["equitraj.adapters"] = saved_ad

    # 4. pyproject.toml packages both equitraj* and lab*
    try:
        import tomllib
        pyproj = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    except ModuleNotFoundError:  # Python 3.10
        from setuptools.config.pyprojecttoml import load_file
        pyproj = load_file(ROOT / "pyproject.toml")
    includes = pyproj["tool"]["setuptools"]["packages"]["find"]["include"]
    assert "lab" in includes and "lab.*" in includes
