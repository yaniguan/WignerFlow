"""Tests for batched FrameSplitPredictor, run_batch_queries, narrow distillation, and YLM context (#92)."""

import json
import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("e3nn")

from equitraj.api import ArgonPredictor, FrameSplitPredictor, unwrap_context_frames
from equitraj.bench import make_model as make_bench_model
from equitraj.distill import copy_normalization_stats, distillation_loss
from equitraj.model import EquiTraj
from lab import distill_edge, surrogate_query as sq, ylm_context_example as ylm_ex
from scripts import distill as distill_cli

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / sq.REGISTRY["argon"]["checkpoint"]


def test_frame_split_predictor_batched_heun_and_pbc_context():
    torch.manual_seed(0)
    model = make_bench_model(channels=16, heads=4)
    b, n = 3, 20
    box = torch.stack([torch.eye(3) * (13.0 + i) for i in range(b)])
    pos_unw = torch.rand(b, 3, n, 3) * 12.0
    pos_unw[:, 2, 0, 0] = 13.4  # crosses boundary of system 0 (L=13.0)
    pos_wrap = torch.stack([pos_unw[i] % float(13.0 + i) for i in range(b)])
    vel = torch.randn(b, 3, n, 3) * 0.003
    types = torch.zeros(n, dtype=torch.long)
    masses = torch.full((n,), 39.948)
    noise = model.com_free_noise(b, types, masses, like=vel, generator=torch.Generator().manual_seed(5))

    fsp = FrameSplitPredictor(model)
    for sampler in ("euler", "heun"):
        dx_ref, v_ref = model.predict(pos_unw, vel, types, box, masses, n_steps=2, noise=noise, sampler=sampler)
        dx_fsp, v_fsp = fsp.predict(pos_unw, vel, types, box, masses, n_steps=2, noise=noise, sampler=sampler)
        torch.testing.assert_close(dx_fsp, dx_ref, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(v_fsp, v_ref, rtol=1e-5, atol=1e-6)

    # Smooth trajectory crossing periodic boundary unwraps back to continuous coordinates
    smooth = torch.zeros(b, 3, n, 3)
    smooth[0, :, 0, 0] = torch.tensor([12.8, 13.1, 13.4])
    wrapped = torch.stack([smooth[i] % float(13.0 + i) for i in range(b)])
    recovered, did_unwrap = unwrap_context_frames(wrapped, box)
    assert did_unwrap is True
    dx_s, v_s = fsp.predict(smooth, vel, types, box, masses, n_steps=2, noise=noise)
    dx_r, v_r = fsp.predict(recovered, vel, types, box, masses, n_steps=2, noise=noise)
    torch.testing.assert_close(dx_r, dx_s, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v_r, v_s, rtol=1e-5, atol=1e-6)


def test_argon_predictor_preserves_threads_when_unset():
    if not CKPT.exists():
        pytest.skip("release checkpoint not present")
    prev = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        _ = ArgonPredictor(CKPT)
        assert torch.get_num_threads() == 2
    finally:
        torch.set_num_threads(prev)


def test_run_batch_queries_and_cuda_error_handling(tmp_path, monkeypatch):
    if not CKPT.exists():
        pytest.skip("release checkpoint not present")
    rel = sq.load_release(CKPT, "cpu")
    queries = [
        {"system": "argon", "n_steps": 1, "sampling_steps": 1, "seed": 2},
        {"smiles": "CCO", "n_steps": 1, "sampling_steps": 1},
        {"system": "argon", "n_steps": 2, "sampling_steps": 1, "seed": 9},
        {"bad_key": 123},
    ]
    batch_reps = sq.run_batch_queries(queries, model_and_ckpt=rel)
    assert len(batch_reps) == 4
    assert batch_reps[1]["status"] == "unsupported"
    assert batch_reps[3]["status"] == "invalid"
    assert batch_reps[0]["status"] == "ok" and batch_reps[0]["batch_size"] == 2
    assert batch_reps[2]["status"] == "ok" and batch_reps[2]["batch_size"] == 2
    assert "gate_verdict" in batch_reps[0] and isinstance(batch_reps[0]["gate_verdict"]["passed"], bool)

    # Batched per-query seed + step slicing matches individual run_query calls
    single_0 = sq.run_query(queries[0], model_and_ckpt=rel)
    single_2 = sq.run_query(queries[2], model_and_ckpt=rel)
    assert batch_reps[0]["frames_used"] == 1 and batch_reps[2]["frames_used"] == 2
    assert batch_reps[0]["T_model_mean_K"] == pytest.approx(single_0["T_model_mean_K"], rel=1e-4)
    assert batch_reps[2]["T_model_mean_K"] == pytest.approx(single_2["T_model_mean_K"], rel=1e-4)

    # Batch JSON file works via CLI
    batch_json = tmp_path / "batch_queries.json"
    batch_json.write_text(json.dumps(queries[:2]), encoding="utf-8")
    assert sq.main(["--query", str(batch_json), "--dry-run"]) == 0

    # Requesting device='cuda' when CUDA is unavailable returns status='error' cleanly
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cuda_rep = sq.run_query({"system": "argon", "n_steps": 1, "sampling_steps": 1, "device": "cuda"})
    assert cuda_rep["status"] == "error" and "CUDA" in cuda_rep["errors"][0]


def test_distillation_default_steps_chemistry_and_narrow_cli(tmp_path):
    kw = dict(
        n_types=2, lmax=1, channels=16, heads=2, n_rbf=8, cutoff=4.5, k=2,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", stride_fs=10.0,
        use_charges=True, max_bond_hops=2,
    )
    teacher = EquiTraj(**kw).eval()
    teacher.out_dx_std.copy_(torch.tensor([0.2, 0.3]))
    teacher.out_v_std.copy_(torch.tensor([0.02, 0.03]))

    student_kw = dict(kw, channels=8, heads=1)
    student = copy_normalization_stats(EquiTraj(**student_kw), teacher)
    assert torch.allclose(student.out_dx_std, teacher.out_dx_std)

    pos = torch.randn(2, 2, 3, 3)
    vel = torch.randn(2, 2, 3, 3) * 0.01
    types = torch.tensor([0, 1, 0], dtype=torch.long)
    masses = torch.tensor([12.0, 16.0, 12.0])
    charges = torch.tensor([0.4, -0.8, 0.4])
    bonds = [[0, 1], [1, 2]]

    # Default teacher_steps=4 produces a 2-step student loss with chemistry forwarded
    loss = distillation_loss(student, teacher, pos, vel, types, None, masses,
                             charges=charges, bonds=bonds, sampler="heun", supervised_weight=0.1)
    assert torch.isfinite(loss)
    loss.backward()

    # CLI scripts/distill.py handles float64 context tensors and narrow student flags
    teacher_pt = tmp_path / "teacher.pt"
    torch.save({
        "format": "wignerflow-v0",
        "elements": ["C", "O"],
        "k": 2,
        "stride_fs": 10.0,
        "model_cfg": {k: v for k, v in kw.items() if k not in ("n_types", "k", "stride_fs")},
        "model": teacher.state_dict(),
    }, teacher_pt)
    ctx_pt = tmp_path / "contexts.pt"
    torch.save({
        "pos": pos.double(),
        "vel": vel,
        "types": types,
        "masses": masses,
        "charges": charges,
        "bonds": bonds,
    }, ctx_pt)
    out_pt = tmp_path / "narrow_student.pt"
    distill_cli.main([
        "--teacher", str(teacher_pt),
        "--data", str(ctx_pt),
        "--out", str(out_pt),
        "--teacher-steps", "2",
        "--student-steps", "1",
        "--student-channels", "8",
        "--student-blocks", "1",
        "--iterations", "2",
    ])
    saved = torch.load(out_pt, map_location="cpu", weights_only=False)
    assert saved["sampling_steps"] == 1
    assert saved["model_cfg"]["channels"] == 8

    # distill_edge config includes S4_narrow_student command
    cfg = distill_edge.load_config()
    s4 = next(s for s in cfg["stages"] if s["id"] == "S4_narrow_student")
    assert s4["command"] and "--student-channels" in s4["command"]


def test_ylm_context_example_pythonpath_and_query_smiles(tmp_path, monkeypatch):
    fake_ylm = tmp_path / "fake_ylm"
    (fake_ylm / "lab").mkdir(parents=True)
    (fake_ylm / "lab" / "protocol_qa.py").write_text(
        "def build_prompt(item, molecule_block='', physics_lines=()):\n"
        "    return 'CUSTOM_YLM\\n' + molecule_block + '\\n'.join(physics_lines)\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(fake_ylm) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    q_file = tmp_path / "cco_query.json"
    q_file.write_text(json.dumps({"smiles": "CCO", "n_steps": 2, "sampling_steps": 1}), encoding="utf-8")
    out = ylm_ex.build_example_prompt(query_path=q_file, dry_run=True)
    assert out.startswith("CUSTOM_YLM\n")
    assert "CCO" in out and "[Ar]" not in out
