"""EquiFF: forces are the gradient of the energy, equivariant, translation invariant, and Verlet conserves the model's own energy."""

import torch
from e3nn import o3

from equitraj.forcefield import EquiFF, verlet_rollout

torch.set_default_dtype(torch.float64)


def build():
    return EquiFF(n_types=1, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.0, n_spatial=1).eval()


def conf(n=8):
    torch.manual_seed(1)
    # a loose cluster so every atom has neighbours; no box
    return (torch.randn(1, n, 3) * 1.2), torch.zeros(n, dtype=torch.long)


def test_forces_match_finite_difference():
    m = build()
    x, t = conf()
    f = m.forces(x, t, None)
    eps = 1e-5
    for i, d in [(0, 0), (3, 2)]:
        xp, xm = x.clone(), x.clone()
        xp[0, i, d] += eps
        xm[0, i, d] -= eps
        num = -(m.atomic_energy(xp, t, None) - m.atomic_energy(xm, t, None)) / (2 * eps) * m.force_scale
        assert torch.allclose(f[0, i, d], num[0], atol=1e-6)


def test_forces_equivariant_and_translation_invariant():
    m = build()
    x, t = conf()
    f = m.forces(x, t, None)
    r = o3.rand_matrix()
    assert torch.allclose(m.forces(x @ r.T, t, None), f @ r.T, atol=1e-8)
    assert torch.allclose(m.forces(x + torch.tensor([1.0, -2.0, 0.5]), t, None), f, atol=1e-8)
    assert torch.allclose(f.sum(1), torch.zeros(1, 3), atol=1e-8)  # no net force: a consequence of translation invariance


def test_verlet_conserves_model_energy():
    m = build()
    x, t = conf()
    masses = torch.full((8,), 40.0)
    v = torch.randn(1, 8, 3) * 0.002
    e0 = m.atomic_energy(x, t, None).item() * m.force_scale.item() + 0.5 * (masses[None, :, None] * v**2).sum().item() * 1e4
    out = verlet_rollout(m, x, v, t, None, masses, n_steps=5, substeps=4, dt_fs=1.0, min_dist=0.0)
    x1, v1 = out["positions"][-1], out["velocities"][-1]
    e1 = m.atomic_energy(x1.double(), t, None).item() * m.force_scale.item() + 0.5 * (masses[None, :, None] * v1.double() ** 2).sum().item() * 1e4
    assert abs(e1 - e0) < 1e-3 * max(abs(e0), 1.0) + 0.5
