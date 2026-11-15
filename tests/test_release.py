"""Smoke test of the v0 release checkpoint (skipped when the file is not present)."""

import os
import sys

import pytest
import torch

CKPT = "checkpoints/argon_flow_v0.pt"


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
def test_v0_checkpoint_warns_once_and_export_sets_attention_version(tmp_path, capsys, monkeypatch):
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")
    import h5py
    import numpy as np
    import yaml

    sys.path.insert(0, "scripts")
    import export_ckpt  # noqa: PLC0415
    from demo import load_release  # noqa: PLC0415
    from equitraj.layers import ATTENTION_VERSION, _WARNED_LEGACY_ATTENTION  # noqa: PLC0415
    from equitraj.model import EquiTraj  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)
    _WARNED_LEGACY_ATTENTION.clear()
    capsys.readouterr()

    load_release(CKPT, "cpu")
    err1 = capsys.readouterr().err
    assert "WARNING:" in err1
    assert "v0.1.0" in err1

    load_release(CKPT, "cpu")
    err2 = capsys.readouterr().err
    assert err2 == ""

    # Build a tiny synthetic test dataset + training checkpoint and run export_ckpt.main()
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
    torch.save(
        {
            "step": 50,
            "model": tiny.state_dict(),
            "elements": ["Ar"],
            "cfg": {"name": "tiny_v1", "data_dir": str(data_dir), "stride_steps": 2, "k": 2, "model": model_cfg},
        },
        run_dir / "best.pt",
    )
    out_pt = tmp_path / "exported_v1.pt"
    monkeypatch.setattr(sys, "argv", ["export_ckpt.py", "--run", str(run_dir), "--ckpt", "best.pt", "--out", str(out_pt)])
    export_ckpt.main()

    exported = torch.load(out_pt, map_location="cpu", weights_only=True)
    assert exported["attention_version"] == ATTENTION_VERSION
    capsys.readouterr()
    load_release(str(out_pt), "cpu")
    assert capsys.readouterr().err == ""
