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
