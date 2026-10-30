"""Profiling helpers (equitraj/profiling.py, scripts/profile_run.py). CPU, tiny models, a few seconds.

Dtypes are explicit because test_model.py changes the global default dtype.
"""

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest
import torch

from equitraj import model as model_module
from equitraj import profiling
from equitraj.model import EquiTraj
from equitraj.profiling import (GpuSampler, StepTimer, parse_nvidia_smi, profile_training, rollout_speed,
                                time_neighbor_list, torch_profile, write_summary)

F32 = torch.float32
ROOT = Path(__file__).resolve().parent.parent


def tiny_model(head="flow", k=2):
    torch.manual_seed(0)
    model = EquiTraj(n_types=2, lmax=1, channels=8, heads=2, n_rbf=4, cutoff=4.0, k=k, n_spatial=1,
                     n_fusion=1, n_denoiser=1, head=head, stride_fs=10.0).float()
    model.dx_std.fill_(0.05)
    model.v_std.fill_(1e-3)
    model.out_dx_std.fill_(1e-3)
    model.out_v_std.fill_(1e-5)
    return model


def system(b=2, frames=3, seed=0):
    """8 atoms on a 3 Å cubic grid in a periodic 12 Å box; pos, vel (b, frames, 8, 3), box (b, 3, 3)."""
    g = torch.Generator().manual_seed(seed)
    r = torch.arange(2, dtype=F32) * 3.0
    x = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    pos = x[None, None] + 0.01 * torch.randn(b, frames, 8, 3, generator=g, dtype=F32)
    vel = 1e-3 * torch.randn(b, frames, 8, 3, generator=g, dtype=F32)
    box = (12.0 * torch.eye(3, dtype=F32))[None].repeat(b, 1, 1)
    types = torch.tensor([0, 1] * 4)
    masses = torch.tensor([40.0, 20.0] * 4, dtype=F32)
    return pos, vel, box, types, masses


# ---------------------------------------------------------------- StepTimer

def test_step_timer_nested_phases_are_exclusive():
    timer = StepTimer("cpu")
    for _ in range(2):
        with timer.step():
            with timer.phase("data"):
                time.sleep(0.01)
            with timer.phase("forward"):
                time.sleep(0.005)
                with timer("neighbor_list"):
                    time.sleep(0.02)
    s = timer.summary()
    assert s["n_steps"] == 2
    ph = s["phases"]
    assert set(profiling.PHASES) <= set(ph)
    assert ph["neighbor_list"]["calls"] == 2 and ph["backward"]["calls"] == 0
    assert ph["neighbor_list"]["mean_ms"] >= 19 and ph["forward"]["mean_ms"] < ph["neighbor_list"]["mean_ms"]
    total = sum(p["total_s"] for p in ph.values())
    assert total == pytest.approx(sum(timer.step_times), rel=0.2)
    assert sum(p["frac"] for p in ph.values()) == pytest.approx(1.0)
    timer.reset()
    assert timer.summary()["n_steps"] == 0


def test_time_neighbor_list_counts_and_restores():
    model = tiny_model()
    pos, vel, box, types, masses = system()
    original = model_module.build_graph
    timer = StepTimer("cpu")
    with time_neighbor_list(timer):
        with timer.phase("forward"):
            model.loss(pos[:, :2], vel[:, :2], types, box, masses, pos[:, 2] - pos[:, 1], vel[:, 2])
    assert model_module.build_graph is original
    assert timer.counts["neighbor_list"] == 2  # context frames + newest frame
    assert timer.totals["neighbor_list"] > 0


# ---------------------------------------------------------------- training / rollout

@pytest.mark.parametrize("m,backprop", [(1, "truncated"), (2, "full")])
def test_profile_training_cpu(m, backprop):
    model = tiny_model()
    pos, vel, box, types, masses = system(frames=2 + m)
    batches = [(pos, vel, box), (pos.flip(0), vel.flip(0), box)]
    before = [p.detach().clone() for p in model.parameters()]
    original = model_module.build_graph
    out = profile_training(model, batches, types, masses, n_steps=3, warmup=1, m=m, backprop=backprop,
                           unroll_sampling_steps=1, lr=1e-3)
    assert out["n_steps"] == 3 and out["batch_size"] == 2 and out["atoms"] == 8 and out["m"] == m
    for p in profiling.PHASES:
        assert out["phases"][p]["total_s"] >= 0
        assert out["phases"][p]["calls"] > 0
    assert out["loss_finite"] and out["steps_per_s"] > 0
    assert out["peak_mem_bytes"] is None  # CPU
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))
    assert model_module.build_graph is original


def test_rollout_speed_cpu():
    model = tiny_model(head="regression")
    pos, vel, box, types, masses = system(b=1, frames=2)
    out = rollout_speed(model, pos, vel, types, box, masses, n_steps=3, sampling_steps=1, warmup=1, nl_repeats=2)
    assert out["finite"] and out["n_steps"] == 3 and out["batch"] == 1 and out["atoms"] == 8
    assert out["steps_per_ns"] == pytest.approx(1e5)  # stride 10 fs
    assert out["ns_per_day"] == pytest.approx(out["steps_per_s"] * 10.0 * 86400 / 1e6)
    assert out["neighbor_list_edges"] > 0 and out["neighbor_list_ms"] > 0
    assert out["neighbor_list_ms_per_step_est"] == pytest.approx(3 * out["neighbor_list_ms"])
    assert model.training  # mode restored
    with pytest.raises(ValueError):
        rollout_speed(model, pos, vel, types, box, masses, n_steps=0)


# ---------------------------------------------------------------- torch.profiler

def test_torch_profile_writes_trace_and_table(tmp_path):
    a = torch.randn(32, 32, dtype=F32)
    out = torch_profile(lambda: (a @ a).sum(), 2, str(tmp_path / "prof"))
    assert os.path.getsize(out["trace"]) > 0
    json.loads(Path(out["trace"]).read_text())
    table = Path(out["key_averages"]).read_text()
    assert "aten::" in table or "mm" in table
    assert out["activities"][0] == "cpu"


# ---------------------------------------------------------------- GPU sampler

def test_parse_nvidia_smi():
    assert parse_nvidia_smi("37, 1024\n[N/A], 2048\n\n") == [(37.0, 1024.0), (None, 2048.0)]
    assert parse_nvidia_smi("") == []


def test_gpu_sampler_noop_without_nvidia_smi(tmp_path):
    csv = tmp_path / "gpu.csv"
    with GpuSampler(str(csv), 0.01, executable="no-such-nvidia-smi-binary") as s:
        assert s.sample_once() == []
    summ = s.summary()
    assert summ["available"] is False and summ["n_samples"] == 0 and summ["util_mean"] is None
    assert not csv.exists()


@pytest.mark.skipif(os.name != "posix", reason="fake nvidia-smi is a shell script")
def test_gpu_sampler_with_fake_nvidia_smi(tmp_path):
    exe = tmp_path / "nvidia-smi"
    exe.write_text("#!/bin/sh\necho '40, 1000'\necho '[N/A], 3000'\n")
    exe.chmod(0o755)
    csv = tmp_path / "out" / "gpu.csv"
    with GpuSampler(str(csv), 0.02, executable=str(exe)) as s:
        deadline = time.time() + 5
        while len(s.samples) < 4 and time.time() < deadline:
            time.sleep(0.02)
    summ = s.summary()
    assert summ["available"] and summ["n_samples"] >= 4 and summ["errors"] == 0
    assert summ["util_mean"] == pytest.approx(40.0) and summ["util_max"] == pytest.approx(40.0)
    assert summ["mem_used_max_MiB"] == pytest.approx(3000.0)
    assert summ["per_gpu"]["1"]["util_mean"] is None and summ["per_gpu"]["0"]["mem_used_mean_MiB"] == 1000.0
    lines = csv.read_text().strip().splitlines()
    assert lines[0] == "t_s,gpu,util_pct,mem_used_MiB" and len(lines) - 1 == summ["n_samples"]


# ---------------------------------------------------------------- summary.json

def test_write_summary_merges(tmp_path):
    path = write_summary(str(tmp_path), {"run": {"a": 1, "b": {"x": 1}}, "train": {"loss": float("nan")}})
    write_summary(str(tmp_path), {"run": {"b": {"y": torch.tensor(2.5)}}, "rollout": {"v": torch.arange(2)}})
    data = json.loads(Path(path).read_text())
    assert data == {"run": {"a": 1, "b": {"x": 1, "y": 2.5}}, "train": {"loss": None}, "rollout": {"v": [0, 1]}}


def test_openmm_comparison(tmp_path):
    spec = importlib.util.spec_from_file_location("profile_run", ROOT / "scripts" / "profile_run.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    bench = {"atoms": 8, "openmm": {"platform": "CPU", "ns_per_day": 20.0, "steps_per_s": 1.0, "step_fs": 1.0},
             "model": {"ns_per_day": 4.0}}
    p = tmp_path / "omm.json"
    p.write_text(json.dumps(bench))
    out = mod.openmm_comparison(str(p), 5.0)
    assert out["speed_ratio"] == pytest.approx(0.25) and out["platform"] == "CPU"
    assert out["bench_model_ns_per_day"] == 4.0
