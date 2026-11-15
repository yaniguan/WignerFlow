"""Fine-tune / curriculum-safety options of equitraj.train and the v0 comparison verdict."""
import json
import math
import subprocess
import sys
from pathlib import Path

import torch

from equitraj.model import EquiTraj
from equitraj.train import SpikeGuard, last_switch_step, load_init_weights, lr_scale_after_switch, unroll_length

ROOT = Path(__file__).resolve().parents[1]
SCHED = [[0, 2], [3000, 3], [8000, 4]]


def test_last_switch_step_and_unroll_length():
    assert last_switch_step(SCHED, 0) is None
    assert last_switch_step(SCHED, 2999) is None
    assert last_switch_step(SCHED, 3000) == 3000
    assert last_switch_step(SCHED, 7999) == 3000
    assert last_switch_step(SCHED, 9000) == 8000
    assert [unroll_length(SCHED, s) for s in (0, 3000, 8000)] == [2, 3, 4]
    # repeated m does not count as a switch
    assert last_switch_step([[0, 1], [10, 1], [20, 2]], 15) is None


def test_lr_rewarmup_ramp():
    assert lr_scale_after_switch(100, None, 1000) == 1.0
    assert lr_scale_after_switch(3000, 3000, 1000) == 0.1
    assert math.isclose(lr_scale_after_switch(3500, 3000, 1000), 0.55)
    assert lr_scale_after_switch(4000, 3000, 1000) == 1.0
    assert lr_scale_after_switch(3000, 3000, 0) == 1.0


def test_spike_guard_skips_spikes_and_resets_on_m_change():
    g = SpikeGuard(factor=5.0, window=100, min_history=10)
    for _ in range(10):
        assert g.check(0.1, 2)
    assert not g.check(11.8, 2)          # the 20050-step explosion pattern (0.12 -> 11.8) is skipped
    assert g.check(0.3, 2)
    assert not g.check(float("nan"), 2)
    assert g.skipped == 2
    # new unroll length: history reset, no guarding until min_history accepted losses
    assert g.check(11.8, 4)
    off = SpikeGuard(factor=0.0)
    assert off.check(1e6, 1) and not off.check(float("inf"), 1)


def _tiny(prior):
    kw = dict(head="flow", lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, n_spatial=1, n_fusion=1,
              n_denoiser=1, parity=True, use_velocity=True, target="residual")
    if prior:
        kw["repulsion_prior"] = {"enabled": True, "strength": 0.5, "rmin_scale": 0.95}
    return EquiTraj(n_types=1, k=3, stride_fs=50.0, **kw)


def test_load_init_weights_from_snapshot_without_prior(tmp_path):
    torch.manual_seed(0)
    src = _tiny(prior=False)
    with torch.no_grad():
        src.dx_std.fill_(0.07)
    path = tmp_path / "step020000.pt"
    torch.save({"model": src.state_dict(), "step": 20000}, path)
    dst = _tiny(prior=True)
    dst.prior.rmin.fill_(3.1)
    missing, step = load_init_weights(dst, str(path), "cpu")
    assert step == 20000
    assert missing and all(k.startswith("prior.") for k in missing)
    assert torch.allclose(dst.dx_std, src.dx_std)
    assert math.isclose(float(dst.prior.rmin.flatten()[0]), 3.1, rel_tol=1e-6)   # estimated r_min kept
    for (k, a), b in zip(src.state_dict().items(), [dst.state_dict()[k] for k in src.state_dict()]):
        assert torch.equal(a, b), k


def _report(v0, v1, crash=(0.0, 0.0)):
    def ck(vals, c):
        names = ("temperature_rel_err", "rdf_l1", "vacf_zero_rel_err", "D_rel_err")
        met = {n: {"mean": v, "ci95": [v, v], "n_valid": 24} for n, v in zip(names, vals)}
        met["crash_fraction"] = {"mean": c, "ci95": [c, c], "n_valid": 24}
        return {"checkpoint": "x.pt", "train_step": 1, "metrics": met}
    return {"checkpoints": [ck(v0, crash[0]), ck(v1, crash[1])]}


def test_compare_v0_verdict(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    from compare_v0 import compare
    v0 = (0.156, 0.533, 0.074, 0.022)
    assert compare(_report(v0, (0.07, 0.2, 0.05, 0.02)))["v1_beats_v0_on_all_four"]
    assert not compare(_report(v0, (0.07, 0.2, 0.09, 0.02)))["v1_beats_v0_on_all_four"]   # VACF worse
    assert not compare(_report(v0, (0.07, 0.2, 0.05, 0.02), crash=(0.0, 0.25)))["v1_beats_v0_on_all_four"]
    assert not compare(_report(v0, (None, 0.2, 0.05, 0.02)))["v1_beats_v0_on_all_four"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(_report(v0, (0.07, 0.2, 0.05, 0.02))))
    out = tmp_path / "v.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts/compare_v0.py"), "--metrics", str(p), "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["v1_beats_v0_on_all_four"] is True


def test_micro_batch_gradient_equals_full_batch():
    from equitraj.train import accumulate_gradients, micro_batch_size_for, training_loss

    # deterministic setup: regression head and no input noise, so no random draws inside the loss
    kw = dict(head="regression", lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, n_spatial=1, n_fusion=1,
              n_denoiser=1, parity=True, use_velocity=True, target="residual",
              repulsion_prior={"enabled": True, "strength": 0.5, "rmin_scale": 0.95})
    torch.manual_seed(0)
    model = EquiTraj(n_types=1, k=3, stride_fs=50.0, **kw).double()
    model.prior.rmin.fill_(2.0)
    b, n, m = 4, 8, 4
    pos = torch.rand(b, 1, n, 3, dtype=torch.float64) * 12.0
    pos = pos + 0.05 * torch.arange(3 + m, dtype=torch.float64).view(1, -1, 1, 1) * torch.randn(b, 1, n, 3, dtype=torch.float64)
    vel = 0.05 * torch.randn(b, 3 + m, n, 3, dtype=torch.float64)
    box = (torch.eye(3, dtype=torch.float64) * 12.0).repeat(b, 1, 1)
    types, masses = torch.zeros(n, dtype=torch.long), torch.full((n,), 40.0, dtype=torch.float64)
    opts = dict(unroll_weights=[1.0, 1.0, 1.0, 1.0], overlap_weight=0.1,
                overlap_r0=torch.full((1, 1), 3.0, dtype=torch.float64))

    full = training_loss(model, pos, vel, box, types, masses, m, 0.0, "truncated", 2, **opts)
    model.zero_grad()
    full.backward()
    full_grads = [p.grad.clone() for p in model.parameters()]

    for size in (1, 3):  # 3 gives uneven micro-batches (3 + 1)
        model.zero_grad()
        value = accumulate_gradients(model, pos, vel, box, types, masses, m, 0.0, "truncated", 2, size, **opts)
        assert math.isclose(value, full.item(), rel_tol=1e-9)
        for g_full, p in zip(full_grads, model.parameters()):
            assert torch.allclose(p.grad, g_full, rtol=1e-8, atol=1e-12)

    tc = {"batch_size": 8}
    assert micro_batch_size_for(tc, 3) == 8           # below m=4: full batch
    assert micro_batch_size_for(tc, 4) == 4           # default: half the batch
    assert micro_batch_size_for({"batch_size": 8, "micro_batch_size": 2}, 4) == 2
    assert micro_batch_size_for({"batch_size": 8, "micro_batch_min_m": 1}, 2) == 4
    assert micro_batch_size_for({"batch_size": 8, "micro_batch_size": 99}, 4) == 8


def test_colab_init_cell_warns_when_init_snapshot_ignored_on_resume(tmp_path, capsys):
    import yaml

    nb = json.loads((ROOT / "colab" / "argon_stable_ft.ipynb").read_text(encoding="utf-8"))
    init_cells = [
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code" and "B_S3a_stable_ft_runtime.yaml" in "".join(c["source"])
    ]
    assert len(init_cells) == 1
    cell_code = init_cells[0]

    run_dir = tmp_path / "argon_full_oct29"
    snaps = run_dir / "train" / "snapshots"
    snaps.mkdir(parents=True)
    s8k = snaps / "step008000.pt"
    s16k = snaps / "step016000.pt"
    s8k.write_bytes(b"s8k")
    s16k.write_bytes(b"s16k")
    ft = run_dir / "train_stable_ft"
    ft.mkdir()
    (ft / "init_rank.json").write_text(json.dumps({"chosen": str(s8k), "ranking": []}))

    def exec_cell(init_snapshot):
        capsys.readouterr()
        ns = {
            "RUN": run_dir,
            "FT": ft,
            "FT_NAME": "train_stable_ft",
            "REPO": ROOT,
            "DATA_LOCAL": tmp_path / "data",
            "INIT_SNAPSHOT": init_snapshot,
            "RANK_TOP": 3,
            "RANK_STEPS": 2000,
            "Path": Path,
            "yaml": yaml,
            "json": json,
            "sys": sys,
            "run": lambda *a, **kw: (_ for _ in ()).throw(AssertionError("rank_snapshots should not run")),
        }
        exec(cell_code, ns)  # noqa: S102
        return capsys.readouterr().out, ns["INIT"]

    # 1. Fresh run with default 'auto' picks step008000.pt and prints full path without warning
    out1, chosen1 = exec_cell("auto")
    assert chosen1 == str(s8k.resolve())
    assert f"init_from: {s8k.resolve()}" in out1
    assert "WARNING:" not in out1

    # 2. Resume with default 'auto' keeps saved init_from without warning
    out2, chosen2 = exec_cell("auto")
    assert chosen2 == str(s8k.resolve())
    assert "WARNING:" not in out2

    # 3. Resume with matching INIT_SNAPSHOT keeps saved init_from without warning
    out3, chosen3 = exec_cell("step008000.pt")
    assert chosen3 == str(s8k.resolve())
    assert "WARNING:" not in out3

    # 4. Resume with different INIT_SNAPSHOT keeps saved init_from AND prints clear warning naming FT_NAME
    out4, chosen4 = exec_cell("step016000.pt")
    assert chosen4 == str(s8k.resolve())
    assert "WARNING:" in out4
    assert "Change FT_NAME" in out4
    assert f"init_from: {s8k.resolve()}" in out4
