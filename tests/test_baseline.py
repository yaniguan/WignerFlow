"""The non-equivariant baseline: valid interface, translation invariant, permutation equivariant, and NOT rotation equivariant."""

import torch
from e3nn import o3

from equitraj.model import make_model

torch.set_default_dtype(torch.float64)


def build(head):
    m = make_model(n_types=2, k=3, stride_fs=50.0, arch="plain", channels=16, heads=2, n_rbf=8, cutoff=4.0, head=head,
                   n_spatial=1, n_fusion=1, n_denoiser=1)
    return m.eval()


def data(b=2, n=6):
    torch.manual_seed(0)
    pos = torch.randn(b, 3, n, 3) * 1.5
    vel = torch.randn(b, 3, n, 3) * 0.01
    types = torch.tensor([0, 1, 0, 1, 0, 1])
    return pos, vel, types, torch.ones(n)


def test_regression_and_flow_run_and_train():
    for head in ["regression", "flow"]:
        m = build(head)
        pos, vel, types, masses = data()
        dx, v = m.predict(pos, vel, types, None, masses, n_steps=2)
        assert dx.shape == (2, 6, 3) and torch.isfinite(dx).all() and torch.isfinite(v).all()
        loss = m.loss(pos, vel, types, None, masses, torch.randn(2, 6, 3) * 0.05, torch.randn(2, 6, 3) * 0.01)
        loss.backward()
        assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_translation_invariant_and_permutation_equivariant():
    m = build("regression")
    pos, vel, types, masses = data()
    dx, v = m.predict(pos, vel, types, None, masses)
    dx2, v2 = m.predict(pos + torch.tensor([3.0, -1.0, 2.0]), vel, types, None, masses)
    assert torch.allclose(dx, dx2, atol=1e-9) and torch.allclose(v, v2, atol=1e-9)
    perm = torch.tensor([3, 1, 5, 0, 2, 4])
    dx3, v3 = m.predict(pos[:, :, perm], vel[:, :, perm], types[perm], None, masses[perm])
    assert torch.allclose(dx[:, perm], dx3, atol=1e-9)


def test_not_rotation_equivariant():
    """The point of the baseline: rotating the input does not rotate the output (so a result gap is due to symmetry)."""
    m = build("regression")
    pos, vel, types, masses = data()
    r = o3.rand_matrix()
    dx, _ = m.predict(pos, vel, types, None, masses)
    dx_rot, _ = m.predict(pos @ r.T, vel @ r.T, types, None, masses)
    assert (dx_rot - dx @ r.T).abs().max() > 1e-4
