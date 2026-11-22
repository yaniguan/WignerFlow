"""Regression tests for script, notebook, and packaging audit fixes."""

from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


def test_demo_load_release_keeps_float32_under_float64_default():
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")
    import demo  # noqa: PLC0415

    ckpt = ROOT / "checkpoints" / "argon_flow_v0.pt"
    if not ckpt.exists():
        pytest.skip("release checkpoint not present")

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model, ck = demo.load_release(str(ckpt), torch.device("cpu"))
        assert next(model.parameters()).dtype == torch.float32
        assert ck["masses"].device.type == "cpu"
    finally:
        torch.set_default_dtype(prev)

    assert demo.first_zero(np.array([0.0, 10.0, 20.0]), np.array([-0.2, -0.1, 0.0])) is None
    assert demo.first_zero(np.array([0.0, 10.0, 20.0]), np.array([1.0, 0.5, -0.5])) == pytest.approx(15.0)


def test_bench_api_cross_platform_helpers():
    import bench_api  # noqa: PLC0415

    name = bench_api.cpu_name()
    assert isinstance(name, str) and len(name) > 0
    la = bench_api._load_avg()
    assert la is None or len(la) == 3


def test_summarize_seeds_and_make_results_path_parts(tmp_path, monkeypatch):
    import make_results  # noqa: PLC0415
    import summarize_seeds  # noqa: PLC0415

    seed_dir = tmp_path / "results" / "argon" / "B_S3a_seed1" / "eval"
    seed_dir.mkdir(parents=True)
    m = {
        "n_rollouts": 5,
        "stability": {"crashed": 0},
        "v0_verdict": {"all_pass": True},
        "temperature": {"rel_err": 0.02, "T_model_second_half_K": 93.0},
        "T_ground_truth_K": 95.0,
        "rdf": {"peak_pos_err_A": 0.02, "peak_height_rel_err": 0.03},
        "vacf": {"first_zero_rel_err": 0.01},
        "diffusion": {"rel_err": 0.05},
        "head": "flow",
        "stride_fs": 50.0,
        "steps": 100,
        "train_step": 1000,
    }
    (seed_dir / "metrics.json").write_text(json.dumps(m), encoding="utf-8")
    out_prefix = tmp_path / "summary" / "seeds_out"
    assert PureWindowsPath(r"results\argon\B_S3a_seed1\eval\metrics.json").parts[-3] == "B_S3a_seed1"
    glob_pat = str(tmp_path / "results" / "argon" / "B_S3a_seed*" / "eval" / "metrics.json")
    summarize_seeds.main(["--glob", glob_pat, "--out", str(out_prefix)])
    data = json.loads(Path(str(out_prefix) + ".json").read_text(encoding="utf-8"))
    assert data["runs"] == ["B_S3a_seed1"]

    monkeypatch.chdir(tmp_path)
    gt_dir = tmp_path / "results" / "argon" / "ground_truth"
    gt_dir.mkdir(parents=True)
    (gt_dir / "metrics.json").write_text(
        json.dumps({
            "n_traj": 2,
            "T_mean_K": 95.0,
            "energy_drift_kJmol_per_ns_per_dof": {"mean": 1e-4, "max_abs": 2e-4},
            "rdf": {"Ar-Ar": {"first_peak_A": 3.70, "first_peak_height": 2.95}},
            "diffusion": {"D_cm2_per_s": 2.4e-5},
            "per_traj": [{"ns_per_day_with_io": 500.0, "platform": "CPU"}],
        }),
        encoding="utf-8",
    )
    # Include a crashed validation log entry where val_T_end is null
    log_path = tmp_path / "results" / "argon" / "B_S3a_seed1" / "log.jsonl"
    log_path.write_text(
        json.dumps({"step": 200, "val_T_end": 94.0, "val_rdf_l1": 0.12}) + "\n"
        + json.dumps({"step": 400, "val_T_end": None, "val_rdf_l1": None}) + "\n",
        encoding="utf-8",
    )
    make_results.main()
    assert (tmp_path / "RESULTS.md").exists()
    assert "argon/B_S3a_seed1" in (tmp_path / "RESULTS.md").read_text(encoding="utf-8")


def test_stability_timeline_vacuum_trajectory(tmp_path):
    pytest.importorskip("h5py")
    import stability_timeline  # noqa: PLC0415
    from equitraj.trajfile import TrajWriter  # noqa: PLC0415

    h5_path = tmp_path / "vac.h5"
    rng = np.random.default_rng(0)
    with TrajWriter(
        h5_path,
        n_atoms=4,
        masses=np.array([12.0, 12.0, 1.0, 1.0], dtype=np.float32),
        elements=["C", "C", "H", "H"],
        attrs={"periodic": False, "dt_fs": 1.0, "save_every": 1, "complete": True},
    ) as writer:
        base = np.array([[0.0, 0.0, 0.0], [0.15, 0.0, 0.0], [0.0, 0.11, 0.0], [0.15, 0.11, 0.0]], dtype=np.float64)
        for step in range(12):
            pos = base + rng.normal(0.0, 0.002, size=base.shape)
            vel = rng.normal(0.0, 0.2, size=base.shape)
            writer.add(pos, vel, np.zeros_like(pos), -10.0, 5.0, np.zeros((3, 3)), step * 0.001)

    out_json = tmp_path / "reports" / "timeline.json"
    stability_timeline.main([str(h5_path), "--reference", str(h5_path), "--blocks", "3", "--out", str(out_json)])
    rep = json.loads(out_json.read_text(encoding="utf-8"))
    assert len(rep["blocks"]) == 3


def test_analyze_profiling_verdict_none_gpu_util_and_empty_plots(tmp_path):
    pytest.importorskip("matplotlib")
    import analyze_profiling  # noqa: PLC0415

    run = {
        "repo": "WignerFlow",
        "label": "test_run",
        "key_averages": {
            "self_cuda_total_ms": 100.0,
            "self_cpu_total_ms": None,
            "top_cuda": [{"name": "aten::add_", "self_cuda_ms": 60.0}],
        },
    }
    bound, ev = analyze_profiling.verdict(run)
    assert "memory-bandwidth" in bound
    assert len(ev) == 1
    paths = analyze_profiling.plot({"r1": run}, str(tmp_path / "charts"))
    assert isinstance(paths, list)


def test_colab_notebooks_are_valid_json_and_compile():
    for nb_path in sorted((ROOT / "colab").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for idx, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") == "code":
                src = "".join(cell.get("source", []))
                compile(src, f"{nb_path.name}#cell{idx}", "exec")
