"""Fixes from the external review: smooth cutoff attention, true position-only model."""

import torch

from equitraj.model import make_model

torch.set_default_dtype(torch.float64)


def small(arch="equi", **kw):
    cfg = dict(n_types=1, k=3, stride_fs=50.0, channels=16, heads=2, n_rbf=8, cutoff=4.0, head="regression",
               n_spatial=1, n_fusion=1, n_denoiser=1, **kw)
    if arch == "equi":
        cfg.update(lmax=2)
    else:
        cfg.update(arch="plain")
    torch.manual_seed(0)
    return make_model(**cfg).eval()


def context(extra_distance):
    """Four atoms, plus one far atom placed `extra_distance` from atom 0 along x (frames identical)."""
    base = torch.tensor([[0.0, 0, 0], [1.5, 0.3, 0], [0.2, 1.4, 0.1], [-1.0, -0.8, 0.9]])
    far = torch.tensor([[extra_distance, 0.0, 0.0]])
    pos = torch.cat([base, far])[None, None].repeat(1, 3, 1, 1)
    vel = torch.zeros_like(pos)
    return pos, vel, torch.zeros(5, dtype=torch.long), torch.ones(5)


def jump_across_cutoff(model):
    out = []
    for dist in (4.0 - 1e-6, 4.0 + 1e-6):  # the far atom is just inside / just outside the 4 Å cutoff of atom 0
        pos, vel, types, masses = context(dist)
        dx, v = model.predict(pos, vel, types, None, masses, n_steps=1)
        out.append(torch.cat([dx, v], -1))
    return (out[0] - out[1])[:, :4].abs().max().item()  # change on the four atoms that were always there


def test_smooth_attention_is_continuous_at_the_cutoff():
    # the v0 attention (softmax, then envelope) renormalizes the other edges when an edge crosses the cutoff
    for arch in ("equi", "plain"):
        plain_jump = jump_across_cutoff(small(arch))
        smooth_jump = jump_across_cutoff(small(arch, smooth_attention=True))
        assert smooth_jump < 1e-4, (arch, smooth_jump)
        assert plain_jump > 10 * smooth_jump, (arch, plain_jump, smooth_jump)


def test_position_only_model_ignores_velocities():
    m = small(use_velocity=False)
    pos, vel, types, masses = context(2.0)
    a = m.predict(pos, vel, types, None, masses, n_steps=1)
    b = m.predict(pos, vel + 0.05 * torch.randn_like(vel), types, None, masses, n_steps=1)
    assert torch.allclose(a[0], b[0]) and torch.allclose(a[1], b[1])  # neither dx nor v_new depends on the true velocity
