"""Rollout-stability tools: repulsion prior, stability monitor, selection score, new training options.

All CPU, tiny models. Dtypes are explicit because test_model.py changes the global default dtype.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from equitraj.model import EquiTraj
from equitraj.monitor import StabilityMonitor
from equitraj.prior import RepulsionPrior, estimate_rmin, overlap_penalty
from equitraj.rollout import rollout, temperature
from equitraj.train import noise_sigma_at, selection_score, training_loss
from equitraj.units import KB

F64 = torch.float64
CONFIGS = Path(__file__).resolve().parent.parent / "configs" / "argon"


def random_orthogonal(seed, reflect=True):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=F64))
    q = q * torch.sign(torch.diagonal(r))[None, :]
    return -q if reflect else q


def grid(n_side, spacing, dtype=F64):
    r = torch.arange(n_side, dtype=dtype) * spacing
    return torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)


def com(x, masses):
    return torch.einsum("n,bnd->bd", masses, x)


# ---------------------------------------------------------------- repulsion prior

def clustered_inputs(seed=0, b=2, n=10):
    g = torch.Generator().manual_seed(seed)
    pos = torch.randn(b, n, 3, generator=g, dtype=F64) * 1.5  # many pairs below 2 Å
    dx = torch.randn(b, n, 3, generator=g, dtype=F64) * 0.05
    vel = torch.randn(b, n, 3, generator=g, dtype=F64) * 0.01
    types = torch.tensor([0, 1] * (n // 2))
    masses = torch.tensor([1.0, 40.0], dtype=F64)[types]
    return pos, dx, vel, types, masses


def test_prior_rotation_translation_equivariant():
    pos, dx, vel, types, masses = clustered_inputs()
    prior = RepulsionPrior(2, strength=0.7, n_iter=2, damp_velocity=0.5, rmin=2.0).double()
    dx1, v1 = prior(pos, dx, vel, types, None, masses)
    assert (dx1 - dx).abs().max() > 1e-3 and (v1 - vel).abs().max() > 1e-4  # the prior is active
    r = random_orthogonal(1)
    shift = torch.tensor([3.0, -7.0, 11.0], dtype=F64)
    dx2, v2 = prior(pos @ r.T + shift, dx @ r.T, vel @ r.T, types, None, masses)
    assert torch.allclose(dx2, dx1 @ r.T, atol=1e-10)
    assert torch.allclose(v2, v1 @ r.T, atol=1e-10)


def test_prior_periodic_translation_invariant():
    g = torch.Generator().manual_seed(3)
    pos = torch.rand(2, 12, 3, generator=g, dtype=F64) * 8.0
    dx = torch.randn(2, 12, 3, generator=g, dtype=F64) * 0.05
    vel = torch.zeros_like(dx)
    types = torch.zeros(12, dtype=torch.long)
    masses = torch.full((12,), 40.0, dtype=F64)
    box = (torch.eye(3, dtype=F64) * 8.0).expand(2, 3, 3)
    prior = RepulsionPrior(1, strength=1.0, rmin=2.5).double()
    dx1, _ = prior(pos, dx, vel, types, box, masses)
    assert (dx1 - dx).abs().max() > 1e-3
    shifted = pos + torch.tensor([5.0, -3.0, 9.0], dtype=F64)
    shifted[:, 0] += torch.tensor([8.0, 0.0, -8.0], dtype=F64)  # a periodic image of atom 0
    dx2, _ = prior(shifted, dx, vel, types, box, masses)
    assert torch.allclose(dx2, dx1, atol=1e-10)


def test_prior_separates_overlapping_pair_and_keeps_com():
    pos = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]], dtype=F64)
    dx = torch.zeros_like(pos)
    types = torch.tensor([0, 0])
    masses = torch.tensor([1.0, 39.0], dtype=F64)
    hard = RepulsionPrior(1, strength=1.0, width=0.0, rmin=3.0).double()
    dx_h, _ = hard(pos, dx, dx, types, None, masses)
    new = pos + dx_h
    assert torch.allclose(torch.linalg.norm(new[0, 1] - new[0, 0]), torch.tensor(3.0, dtype=F64), atol=1e-10)
    assert com(dx_h, masses).abs().max() < 1e-12
    assert dx_h[0, 0, 0] < 0 < dx_h[0, 1, 0]  # pushed apart along the pair axis
    assert abs(dx_h[0, 0, 0]) > abs(dx_h[0, 1, 0])  # the light atom moves more
    soft = RepulsionPrior(1, rmin=3.0).double()  # default soft wall: partial push
    dx_s, _ = soft(pos, dx, dx, types, None, masses)
    d = torch.linalg.norm((pos + dx_s)[0, 1] - (pos + dx_s)[0, 0])
    assert 1.0 < d < 3.0


def test_prior_is_noop_for_separated_pairs():
    pos = grid(3, 4.0)[None]
    g = torch.Generator().manual_seed(0)
    dx = torch.randn(pos.shape, generator=g, dtype=F64) * 0.05
    vel = torch.randn(pos.shape, generator=g, dtype=F64) * 0.01
    types = torch.zeros(27, dtype=torch.long)
    masses = torch.full((27,), 40.0, dtype=F64)
    prior = RepulsionPrior(1, strength=1.0, damp_velocity=1.0, rmin=3.0).double()
    dx1, v1 = prior(pos, dx, vel, types, None, masses)
    assert torch.equal(dx1, dx) and torch.equal(v1, vel)
    assert overlap_penalty(pos + dx, None, types, prior.r0()) == 0


def test_prior_preserves_com_and_momentum():
    pos, dx, vel, types, masses = clustered_inputs(seed=5)
    prior = RepulsionPrior(2, strength=1.0, n_iter=3, damp_velocity=1.0, rmin=2.0).double()
    dx1, v1 = prior(pos, dx, vel, types, None, masses)
    assert torch.allclose(com(dx1, masses), com(dx, masses), atol=1e-10)
    assert torch.allclose(com(v1, masses), com(vel, masses), atol=1e-10)


def test_overlap_penalty_differentiable():
    pos, dx, _, types, _ = clustered_inputs()
    x = (pos + dx).requires_grad_()
    pen = overlap_penalty(x, None, types, torch.full((2, 2), 2.0, dtype=F64))
    pen.backward()
    assert pen > 0 and torch.isfinite(x.grad).all() and x.grad.abs().max() > 0


def test_estimate_rmin_checkerboard():
    x = grid(4, 3.0, torch.float32)  # simple cubic, periodic box 12 Å
    idx = (x / 3.0).round().long()
    types = idx.sum(-1) % 2  # checkerboard: unlike neighbors at 3 Å, like neighbors at 3*sqrt(2) Å
    ds = SimpleNamespace(pos=[np.stack([x.numpy(), x.numpy()])], boxes=[torch.eye(3) * 12.0], periodic=True,
                         types=types, elements=["A", "B"])
    rmin = estimate_rmin(ds, percentile=0.1)
    assert torch.allclose(rmin, rmin.T)
    assert abs(rmin[0, 1] - 3.0) < 1e-4
    assert abs(rmin[0, 0] - 3.0 * 2 ** 0.5) < 1e-4 and abs(rmin[1, 1] - 3.0 * 2 ** 0.5) < 1e-4


# ---------------------------------------------------------------- model integration

def tiny_model(prior=None, n_types=1):
    torch.manual_seed(0)
    return EquiTraj(n_types=n_types, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.0, k=3, n_spatial=1,
                    n_fusion=1, n_denoiser=1, head="flow", repulsion_prior=prior)


def test_model_prior_separates_pair_keeps_com():
    model = tiny_model({"enabled": True, "strength": 1.0, "width": 0.0, "rmin": 3.0}).double().eval()
    model.out_dx_std.fill_(1e-3)
    model.out_v_std.fill_(1e-5)
    x = torch.cat([grid(2, 5.0), torch.tensor([[-1.0, 0.0, 0.0]], dtype=F64)])  # atom 8 is 1 Å from atom 0
    n = x.shape[0]
    pos = x[None, None].expand(1, 3, n, 3).clone()
    vel = torch.zeros_like(pos)
    types = torch.zeros(n, dtype=torch.long)
    masses = torch.full((n,), 40.0, dtype=F64)
    dx, v = model.predict(pos, vel, types, None, masses, n_steps=2)
    new = pos[:, -1] + dx
    assert abs(torch.linalg.norm(new[0, 8] - new[0, 0]) - 3.0) < 1e-8
    assert com(dx, masses).abs().max() < 1e-10 and com(v, masses).abs().max() < 1e-10


def test_prior_buffer_in_state_dict_and_old_checkpoints_load():
    plain = tiny_model()
    with_prior = tiny_model({"enabled": True, "rmin": 3.0})
    disabled = tiny_model({"enabled": False, "strength": 1.0})
    assert plain.prior is None and disabled.prior is None
    assert set(plain.state_dict()) == set(disabled.state_dict())
    assert "prior.rmin" in with_prior.state_dict()
    with_prior.load_state_dict(plain.state_dict())  # old checkpoint: r_min keeps its value
    assert torch.allclose(with_prior.prior.rmin, torch.full((1, 1), 3.0))
    fresh = tiny_model({"enabled": True})
    fresh.load_state_dict(with_prior.state_dict())
    assert torch.allclose(fresh.prior.rmin, torch.full((1, 1), 3.0))


def test_old_config_still_loads():
    cfg = yaml.safe_load(open(CONFIGS / "B_S3a.yaml"))
    model = EquiTraj(n_types=1, k=cfg["k"], stride_fs=50.0, **cfg["model"])
    assert model.prior is None and not any(k.startswith("prior.") for k in model.state_dict())
    tc = cfg["train"]
    assert noise_sigma_at(tc, 0) == tc["noise_sigma"] and noise_sigma_at(tc, 10 ** 6) == tc["noise_sigma"]
    new = yaml.safe_load(open(CONFIGS / "B_S3a_stable.yaml"))
    model2 = EquiTraj(n_types=1, k=new["k"], stride_fs=50.0, **new["model"])
    assert model2.prior is not None
    model2.load_state_dict(model.state_dict())


def test_rollout_returns_diagnostics():
    model = tiny_model().float().eval()
    model.out_dx_std.fill_(1e-3)
    model.out_v_std.fill_(1e-5)
    x = grid(2, 3.0, torch.float32)
    pos = x[None, None].expand(1, 3, 8, 3).clone()
    vel = torch.full_like(pos, 1e-3)
    out = rollout(model, pos, vel, torch.zeros(8, dtype=torch.long), None, torch.full((8,), 40.0, dtype=torch.float32), 3,
                  sampling_steps=1, check_every=1)
    d = out["diagnostics"]
    assert out["crash_step"].tolist() == [-1] and d["crash_step"] == [-1] and d["crashes"] == [None]
    assert len(d["T_drift_K_per_ns"]) == 1 and d["series"]["step"] == [1, 2, 3]
    assert d["min_pair_distance"][0] > 2.5


# ---------------------------------------------------------------- stability monitor

def monitor_inputs():
    g = torch.Generator().manual_seed(0)
    x = grid(2, 3.0)[None].repeat(2, 1, 1)
    v = torch.randn(x.shape, generator=g, dtype=F64) * 0.002
    types = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    masses = torch.full((8,), 40.0, dtype=F64)
    return x, v, types, masses


def test_monitor_detects_nan():
    x, v, types, masses = monitor_inputs()
    mon = StabilityMonitor(types, masses, t_max=1e4)
    x[1, 3, 0] = float("nan")
    bad = mon.check(10, x, v)
    assert bad.tolist() == [False, True]
    d = mon.diagnostics()
    assert d["crash_step"] == [-1, 10] and d["crash_reason"] == [None, "nan"]
    c = d["crashes"][1]
    assert c["atoms"] == [3] and c["atom_types"] == [0] and c["history"][-1]["step"] == 10


def test_monitor_detects_overlap_with_pair():
    x, v, types, masses = monitor_inputs()
    mon = StabilityMonitor(types, masses, t_max=1e4)
    x[0, 5] = x[0, 2] + torch.tensor([0.2, 0.0, 0.0], dtype=F64)
    mon.check(10, x, v)
    c = mon.diagnostics()["crashes"][0]
    assert c["reason"] == "overlap" and c["atoms"] == [2, 5] and c["atom_types"] == [0, 1]
    assert c["pairs"][0][:2] == [2, 5] and abs(c["value"] - 0.2) < 1e-10
    assert mon.diagnostics()["crashes"][1] is None


def test_monitor_detects_hot_temperature():
    x, v, types, masses = monitor_inputs()
    mon = StabilityMonitor(types, masses, t_max=1000.0)
    assert not mon.check(10, x, v).any()
    v[1, 6] = torch.tensor([0.5, 0.0, 0.0], dtype=F64)
    mon.check(20, x, v)
    c = mon.diagnostics()["crashes"][1]
    assert c["reason"] == "temperature" and c["atoms"][0] == 6 and c["atom_types"][0] == 1
    assert c["value"] > 1000.0 and [h["step"] for h in c["history"]] == [10, 20]


def test_monitor_reason_priority_bonds_and_first_crash_kept():
    x, v, types, masses = monitor_inputs()
    bonds, r0 = torch.tensor([[0, 1]]), torch.tensor([3.0], dtype=F64)
    mon = StabilityMonitor(types, masses, t_max=1e4, bonds=bonds, bond_r0=r0, history=2)
    x[0, 5] = x[0, 2] + 0.1  # overlap ...
    v[0, 4, 1] = float("nan")  # ... and NaN: NaN wins
    x[1, 1, 2] += 3.0  # bond 0-1 stretched by 100 %
    mon.check(10, x, v)
    mon.check(20, x, v)
    mon.check(30, x, v)
    d = mon.diagnostics()
    assert d["crash_step"] == [10, 10]
    a, b = d["crashes"]
    assert a["reason"] == "nan" and a["reasons"] == ["nan", "overlap"] and a["atoms"] == [4]
    assert b["reason"] == "bond" and b["atoms"] == [0, 1] and abs(b["value"] - 1.0) < 1e-10
    assert len(a["history"]) == 1


def test_monitor_temperature_and_ke_drift():
    x, v0, types, masses = monitor_inputs()
    mon = StabilityMonitor(types, masses, stride_fs=10.0)
    t0 = temperature(v0, masses)
    rate = 1e5  # K/ns
    for step in range(10, 110, 10):
        t_ns = step * 10.0 * 1e-6
        mon.check(step, x, v0 * torch.sqrt((t0 + rate * t_ns) / t0)[:, None, None])
    d = mon.diagnostics()
    assert np.allclose(d["T_drift_K_per_ns"], rate, rtol=1e-6)
    dof = 3 * 8 - 3
    assert np.allclose(d["KE_drift_kJ_mol_per_ns"], rate * dof * KB / 2, rtol=1e-6)
    assert all(r > 0 for r in d["KE_rel_drift"]) and d["crash_step"] == [-1, -1]


# ---------------------------------------------------------------- selection and training

def test_selection_score_ordering():
    a = {"val_crash_frac": 0.0, "val_steps_before_crash": 1000, "val_rdf_l1": 0.5, "val_T_rel_err": 0.1}
    b = {"val_crash_frac": 0.0, "val_steps_before_crash": 1000, "val_rdf_l1": 0.2, "val_T_rel_err": 0.05}
    c = {"val_crash_frac": 0.25, "val_steps_before_crash": 900, "val_rdf_l1": 0.01, "val_T_rel_err": 0.0}
    d = {"val_crash_frac": 0.25, "val_steps_before_crash": 950}
    e = {"val_crash_frac": 1.0, "val_steps_before_crash": 865}
    ranked = sorted([e, c, a, d, b], key=selection_score)
    assert ranked == [b, a, d, c, e]
    # longer selection rollouts take precedence over the short ones
    s1 = dict(a, sel_val_crash_frac=0.5, sel_val_steps_before_crash=3000)
    s2 = dict(c, sel_val_crash_frac=0.0, sel_val_steps_before_crash=4000)
    assert selection_score(s2) < selection_score(s1)


def test_noise_schedule():
    tc = {"noise_sigma": 0.01, "noise_schedule": [[0, 0.03], [100, 0.01], [200, 0.005]]}
    assert noise_sigma_at(tc, 0) == pytest.approx(0.03)
    assert noise_sigma_at(tc, 50) == pytest.approx(0.02)
    assert noise_sigma_at(tc, 150) == pytest.approx(0.0075)
    assert noise_sigma_at(tc, 10 ** 6) == pytest.approx(0.005)
    assert noise_sigma_at({"noise_sigma": 0.01}, 50) == 0.01 and noise_sigma_at({}, 50) == 0.0


def train_batch(m=2):
    g = torch.Generator().manual_seed(1)
    x0 = grid(2, 3.0)
    vel = torch.randn(2, 3 + m, 8, 3, generator=g, dtype=F64) * 0.01
    pos = x0[None, None] + torch.cumsum(vel, dim=1)  # stride_fs = 1
    types = torch.zeros(8, dtype=torch.long)
    masses = torch.full((8,), 40.0, dtype=F64)
    return pos, vel, types, masses


@pytest.mark.parametrize("backprop", ["truncated", "full"])
def test_training_loss_with_new_options(backprop):
    pos, vel, types, masses = train_batch()
    model = tiny_model({"enabled": True, "rmin": 2.8}).double().train()
    loss = training_loss(model, pos, vel, None, types, masses, 2, 0.02, backprop, 1,
                         unroll_weights=[1.0, 2.0], noise_v_ratio=0.5, noise_type_scale=torch.tensor([0.5]),
                         overlap_weight=0.1, overlap_r0=torch.full((1, 1), 3.5, dtype=F64))
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(loss) and sum(g is not None for g in grads) > 0.9 * len(grads)
    assert all(torch.isfinite(g).all() for g in grads if g is not None)


def test_unit_unroll_weights_match_default_full():
    pos, vel, types, masses = train_batch()
    model = tiny_model().double().train()
    torch.manual_seed(7)
    ref = training_loss(model, pos, vel, None, types, masses, 2, 0.01, "full", 1)
    torch.manual_seed(7)
    new = training_loss(model, pos, vel, None, types, masses, 2, 0.01, "full", 1, unroll_weights=[1.0])
    assert torch.allclose(ref, new, rtol=1e-12)
