"""Symmetry tests for the model and its parts (float64, CPU)."""

import pytest
import torch
from e3nn import o3

from equitraj import so3
from equitraj.layers import SO2Linear
from equitraj.model import EquiTraj
from equitraj.neighbors import brute_force, cell_list

torch.set_default_dtype(torch.float64)


def random_rotation(reflect):
    r = o3.rand_matrix()
    if reflect:
        r = -r  # an improper rotation (rotation times inversion)
    return r


# ---------------------------------------------------------------- SO(3) helpers

def test_edge_wigner_aligns_edge_with_y():
    vec = torch.randn(20, 3)
    d = so3.edge_wigner(vec, 2)
    sh = o3.spherical_harmonics(list(range(3)), vec, normalize=True)[..., None]  # (E, 9, 1)
    aligned = so3.to_edge_frame(d, sh)[..., 0]
    m0 = [so3.index(l, 0) for l in range(3)]
    other = [i for i in range(9) if i not in m0]
    assert aligned[:, other].abs().max() < 1e-10
    # D is orthogonal
    eye = torch.einsum("eij,eik->ejk", d, d)
    assert torch.allclose(eye, torch.eye(9).expand_as(eye), atol=1e-10)


def test_edge_wigner_matches_e3nn():
    vec = torch.randn(30, 3)
    alpha, beta = so3.edge_angles(vec)
    d = so3.edge_wigner(vec, 3)
    for l in range(4):
        ref = o3.wigner_D(l, alpha, beta, torch.zeros_like(alpha))
        assert torch.allclose(d[:, l * l:(l + 1) ** 2, l * l:(l + 1) ** 2], ref, atol=1e-10)


def test_mirror_sign_pattern():
    """A mirror through a plane containing y flips exactly the m<0 components, for every l.
    SO2Linear(parity=True) relies on this."""
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    x = torch.eye(16)[..., None]  # basis vectors, lmax = 3
    mx = so3.rotate(x, mirror, 3)[..., 0]
    expected = torch.ones(16)
    for l in range(4):
        for m in range(-l, 0):
            expected[so3.index(l, m)] = -1.0
    assert torch.allclose(torch.diag(mx), expected, atol=1e-10)


@pytest.mark.parametrize("parity", [True, False])
def test_so2_linear_commutes_with_rotation_about_y(parity):
    lin = SO2Linear(2, 4, 5, parity=parity)
    x = torch.randn(7, 9, 4)
    g = torch.tensor(0.83)
    ry = o3.angles_to_matrix(g, torch.tensor(0.0), torch.tensor(0.0))  # rotation about y
    lhs = lin(so3.rotate(x, ry, 2))
    rhs = so3.rotate(lin(x), ry, 2)
    assert torch.allclose(lhs, rhs, atol=1e-10)
    # with parity=True it also commutes with the mirror x -> -x (a plane containing y)
    mirror = torch.diag(torch.tensor([-1.0, 1.0, 1.0]))
    err = (lin(so3.rotate(x, mirror, 2)) - so3.rotate(lin(x), mirror, 2)).abs().max()
    assert (err < 1e-10) == parity


# ---------------------------------------------------------------- neighbor lists

def edge_set(src, dst):
    return set(zip(src.tolist(), dst.tolist()))


@pytest.mark.parametrize("n_atoms,length", [(300, 15.0), (50, 10.1)])
def test_cell_list_matches_brute_force(n_atoms, length):
    pos = torch.rand(n_atoms, 3) * length - 3.0  # some atoms outside the box on purpose
    box = torch.eye(3) * length
    a = brute_force(pos, box, 3.3)
    b = cell_list(pos, box, 3.3)
    assert edge_set(a[0], a[1]) == edge_set(b[0], b[1])


def test_brute_force_triclinic_matches_image_search():
    box = torch.tensor([[10.0, 0.0, 0.0], [2.0, 9.0, 0.0], [1.0, 1.5, 11.0]])
    pos = torch.rand(40, 3) @ box
    src, dst, vec = brute_force(pos, box, 4.0)
    # reference: check all 27 periodic images explicitly
    shifts = torch.stack(torch.meshgrid(*[torch.arange(-1, 2)] * 3, indexing="ij"), -1).reshape(-1, 3).double() @ box
    dr = pos[:, None, None, :] - pos[None, :, None, :] + shifts[None, None]
    dmin = dr.norm(dim=-1).min(dim=-1).values
    ref = (dmin < 4.0) & ~torch.eye(40, dtype=torch.bool)
    assert edge_set(src, dst) == edge_set(*torch.nonzero(ref, as_tuple=True))
    assert torch.allclose(vec.norm(dim=-1), dmin[src, dst])


# ---------------------------------------------------------------- full model

def make_inputs(b=2, k=3, n=12, periodic=False, seed=0):
    g = torch.Generator().manual_seed(seed)
    types = torch.randint(0, 3, (n,), generator=g)
    masses = torch.tensor([1.0, 12.0, 16.0])[types]
    if periodic:
        box = torch.eye(3).expand(b, 3, 3) * 12.0
        pos0 = torch.rand(b, 1, n, 3, generator=g) * 12.0
    else:
        box = None
        pos0 = torch.randn(b, 1, n, 3, generator=g) * 2.5
    vel = torch.randn(b, k, n, 3, generator=g) * 0.01
    pos = pos0 + torch.cumsum(vel, dim=1) * 10.0
    return pos, vel, types, masses, box


def make_model(head, periodic=False):
    torch.manual_seed(0)
    m = EquiTraj(n_types=3, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.0, k=3, n_spatial=1,
                 n_fusion=1, n_denoiser=1, head=head)
    return m.double().eval()


def run(model, pos, vel, types, masses, box, noise):
    return model.predict(pos, vel, types, box, masses, n_steps=2, noise=noise)


@pytest.mark.parametrize("head", ["regression", "flow"])
@pytest.mark.parametrize("reflect", [False, True])
def test_model_is_equivariant_vacuum(head, reflect):
    pos, vel, types, masses, box = make_inputs()
    model = make_model(head)
    noise = model.com_free_noise(2, types, masses, like=pos)
    r = random_rotation(reflect)
    shift = torch.randn(3) * 5
    dx, v = run(model, pos, vel, types, masses, box, noise)
    # rotate and translate the inputs; the noise is rotated too (it is a vector-valued input)
    noise_r = (noise @ r.T)
    dx_r, v_r = run(model, pos @ r.T + shift, vel @ r.T, types, masses, box, noise_r)
    assert torch.allclose(dx_r, dx @ r.T, atol=1e-10)
    assert torch.allclose(v_r, v @ r.T, atol=1e-10)


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_model_is_equivariant_float32(head):
    """float32 error over 10 random inputs, relative to the largest output.

    Measured (untrained model): median ~4e-6, worst ~2e-5 over 40 inputs. The error is ordinary
    float32 rounding accumulated through the layers; we require median < 1e-5 and max < 1e-4.
    """
    errs = []
    for seed in range(10):
        torch.manual_seed(seed)
        pos, vel, types, masses, box = make_inputs(seed=seed)
        model = make_model(head).float()
        pos, vel, masses = pos.float(), vel.float(), masses.float()
        noise = model.com_free_noise(2, types, masses, like=pos)
        r = random_rotation(True).float()
        dx, v = run(model, pos, vel, types, masses, box, noise)
        dx_r, v_r = run(model, pos @ r.T, vel @ r.T, types, masses, box, noise @ r.T)
        errs.append(max(((dx_r - dx @ r.T).abs().max() / dx.abs().max()).item(),
                        ((v_r - v @ r.T).abs().max() / v.abs().max()).item()))
    errs = torch.tensor(errs)
    assert errs.median() < 1e-5 and errs.max() < 1e-4, errs


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_model_is_permutation_equivariant(head):
    pos, vel, types, masses, box = make_inputs()
    model = make_model(head)
    noise = model.com_free_noise(2, types, masses, like=pos)
    perm = torch.randperm(types.shape[0])
    dx, v = run(model, pos, vel, types, masses, box, noise)
    noise_p = noise.view(2, -1, 2, 3)[:, perm].reshape(-1, 2, 3)
    dx_p, v_p = run(model, pos[:, :, perm], vel[:, :, perm], types[perm], masses[perm], box, noise_p)
    assert torch.allclose(dx_p, dx[:, perm], atol=1e-10)
    assert torch.allclose(v_p, v[:, perm], atol=1e-10)


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_model_periodic_invariances(head):
    pos, vel, types, masses, box = make_inputs(periodic=True)
    model = make_model(head)
    noise = model.com_free_noise(2, types, masses, like=pos)
    dx, v = run(model, pos, vel, types, masses, box, noise)
    # 1. shift one atom by a box vector in every frame: nothing changes
    pos2 = pos.clone()
    pos2[:, :, 3] += box[0, 1]
    dx2, v2 = run(model, pos2, vel, types, masses, box, noise)
    assert torch.allclose(dx2, dx, atol=1e-10) and torch.allclose(v2, v, atol=1e-10)
    # 2. rotate atoms and box together: outputs rotate
    r = random_rotation(True)
    dx3, v3 = run(model, pos @ r.T, vel @ r.T, types, masses, box @ r.T, noise @ r.T)
    assert torch.allclose(dx3, dx @ r.T, atol=1e-10) and torch.allclose(v3, v @ r.T, atol=1e-10)


def test_predictions_have_no_com_motion():
    pos, vel, types, masses, box = make_inputs()
    for head in ["regression", "flow"]:
        dx, v = run(make_model(head), pos, vel, types, masses, box, None)
        assert torch.einsum("n,bnd->bd", masses, v).abs().max() < 1e-10
        assert torch.einsum("n,bnd->bd", masses, dx).abs().max() < 1e-10


def test_losses_backpropagate():
    pos, vel, types, masses, box = make_inputs()
    for head in ["regression", "flow"]:
        model = make_model(head).train()
        loss = model.loss(pos, vel, types, box, masses, torch.randn_like(vel[:, 0]), torch.randn_like(vel[:, 0]))
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert torch.isfinite(loss) and sum(g is not None for g in grads) > 0.9 * len(grads)
