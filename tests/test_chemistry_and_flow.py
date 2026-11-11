"""Tests for chemistry/topology conditioning, geometry-aware denoiser, Heun sampler, mass scaling, and stride conditioning."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

o3 = pytest.importorskip("e3nn.o3")

from equitraj.dataset import TrajWindows, compute_stats  # noqa: E402
from equitraj.fast import FastRollout  # noqa: E402
from equitraj.model import EquiTraj  # noqa: E402
from equitraj.pair_masks import bond_hop_matrix, exclusions_from_bonds  # noqa: E402
from equitraj.prior import RepulsionPrior, overlap_penalty  # noqa: E402
from equitraj.rollout import rollout  # noqa: E402
from equitraj.topology import write_metadata  # noqa: E402
from equitraj.trajfile import TrajWriter  # noqa: E402

F64 = torch.float64


def random_rotation(seed, reflect=True):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=F64))
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return -q if reflect else q


def molecular_inputs(b=2, k=3, periodic=False, seed=0):
    """6-atom system (two 3-atom molecules: 0-1-2 and 3-4-5) with mixed masses, charges, and FF types."""
    g = torch.Generator().manual_seed(seed)
    n = 6
    types = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.long)
    ff_types = torch.tensor([0, 1, 2, 0, 3, 2], dtype=torch.long)
    masses = torch.tensor([1.008, 12.011, 15.999, 6.94, 12.011, 18.998], dtype=F64)
    charges = torch.tensor([0.25, 0.55, -0.80, 0.80, 0.10, -0.90], dtype=F64)
    bonds = torch.tensor([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=torch.long)
    if periodic:
        box = (torch.eye(3, dtype=F64) * 12.0).expand(b, 3, 3).clone()
        pos0 = torch.rand(b, 1, n, 3, generator=g, dtype=F64) * 10.0 + 1.0
    else:
        box = None
        pos0 = torch.randn(b, 1, n, 3, generator=g, dtype=F64) * 1.8
    vel = torch.randn(b, k, n, 3, generator=g, dtype=F64) * 0.01
    pos = pos0 + torch.cumsum(vel, dim=1) * 10.0
    return pos, vel, types, masses, box, charges, ff_types, bonds


def make_upgraded_model(head="flow", **overrides):
    torch.manual_seed(0)
    kw = dict(
        n_types=3, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.5, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head=head, stride_fs=20.0,
        n_ff_types=4, use_charges=True, max_bond_hops=3,
        geometry_aware=(head == "flow"), target_scaling="mass", stride_conditioning=True,
    )
    kw.update(overrides)
    m = EquiTraj(**kw).to(F64).eval()
    if m.target_scaling == "mass":
        m.mass_dx_std.fill_(0.4)
        m.mass_v_std.fill_(0.03)
    return m


# ---------------------------------------------------------------- bond hops and exclusions

def test_bond_hop_matrix_and_exclusions():
    # Chain 0-1-2-3 plus isolated atom 4
    bonds = [[0, 1], [1, 2], [2, 3]]
    hops = bond_hop_matrix(bonds, 5, max_hops=3)
    expected = torch.tensor([
        [0, 1, 2, 3, 0],
        [1, 0, 1, 2, 0],
        [2, 1, 0, 1, 0],
        [3, 2, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=torch.long)
    assert torch.equal(hops, expected)
    excl_13 = exclusions_from_bonds(bonds, 5, max_hops=2)
    assert excl_13[0, 1] and excl_13[0, 2] and not excl_13[0, 3] and not excl_13[0, 0]
    assert torch.equal(excl_13, excl_13.T)

    # Integrates cleanly with RepulsionPrior and overlap_penalty
    pos = torch.tensor([[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [1.8, 0.0, 0.0],
                         [5.0, 0.0, 0.0], [5.5, 0.0, 0.0]]], dtype=F64)
    types = torch.zeros(5, dtype=torch.long)
    masses = torch.ones(5, dtype=F64)
    prior = RepulsionPrior(1, strength=1.0, rmin=2.0, excluded_pairs=excl_13).to(F64)
    corr = prior.correction(pos, None, types, masses)
    # 0-1 and 1-2 and 0-2 are excluded; non-bonded 3-4 (0.5 Å < 2.0 Å) is pushed apart
    assert torch.allclose(corr[0, :3], torch.zeros(3, 3, dtype=F64))
    assert corr[0, 3, 0] < 0 < corr[0, 4, 0]
    assert overlap_penalty(pos[:, :3], None, types[:3], prior.r0(),
                           excluded_pairs=exclusions_from_bonds([[0, 1], [1, 2]], 3, max_hops=2)) == 0


def test_bond_hop_validation():
    assert torch.equal(bond_hop_matrix(None, 3), torch.zeros((3, 3), dtype=torch.long))
    assert torch.equal(bond_hop_matrix([], 3), torch.zeros((3, 3), dtype=torch.long))
    for bad in ([[0, 0]], [[-1, 1]], [[0, 3]], [[0.5, 1.0]], [[[0, 1]]]):
        with pytest.raises(ValueError):
            bond_hop_matrix(bad, 3)


# ---------------------------------------------------------------- backwards compatibility

def test_default_state_dict_unchanged_and_v0_loads():
    m_default = EquiTraj(n_types=1, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.0, k=3,
                         n_spatial=1, n_fusion=1, n_denoiser=1, head="flow")
    keys = set(m_default.state_dict().keys())
    for forbidden in ("ff_type_embed", "charge_in", "bond_embed", "stride_in", "mass_dx_std", "_chem"):
        assert not any(forbidden in k for k in keys)

    ckpt = Path(__file__).resolve().parents[1] / "checkpoints" / "argon_flow_v0.pt"
    if ckpt.exists():
        ck = torch.load(ckpt, map_location="cpu", weights_only=True)
        model = EquiTraj(n_types=len(ck["elements"]), k=ck["k"], stride_fs=ck["stride_fs"], **ck["model_cfg"]).float().eval()
        model.load_state_dict(ck["model"], strict=True)
        d = ck["demo"]
        for sampler in ("euler", "heun"):
            dx, v = model.predict(d["pos"][None], d["vel"][None], ck["types"], d["box"][None],
                                  ck["masses"], n_steps=2, sampler=sampler)
            assert dx.shape == (1, len(ck["types"]), 3) and torch.isfinite(dx).all() and torch.isfinite(v).all()


# ---------------------------------------------------------------- exact O(3), PBC, and permutation symmetries

@pytest.mark.parametrize("head", ["regression", "flow"])
@pytest.mark.parametrize("sampler", ["euler", "heun"])
@pytest.mark.parametrize("reflect", [False, True])
def test_upgraded_model_o3_and_translation_equivariant(head, sampler, reflect):
    pos, vel, types, masses, box, charges, ff_types, bonds = molecular_inputs()
    model = make_upgraded_model(head=head)
    noise = model.com_free_noise(2, types, masses, like=pos)
    rot = random_rotation(11 if reflect else 7, reflect=reflect)
    shift = torch.tensor([4.2, -3.1, 8.7], dtype=F64)
    strides = torch.tensor([15.0, 35.0], dtype=F64)

    dx, v = model.predict(pos, vel, types, box, masses, n_steps=2, noise=noise, sampler=sampler,
                          charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=strides)
    dx_r, v_r = model.predict(pos @ rot.T + shift, vel @ rot.T, types, box, masses, n_steps=2,
                              noise=noise @ rot.T, sampler=sampler,
                              charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=strides)
    assert torch.allclose(dx_r, dx @ rot.T, atol=1e-10)
    assert torch.allclose(v_r, v @ rot.T, atol=1e-10)
    # COM motion remains identically zero
    assert torch.einsum("n,bnd->bd", masses, dx).abs().max() < 1e-10
    assert torch.einsum("n,bnd->bd", masses, v).abs().max() < 1e-10


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_upgraded_model_periodic_and_permutation_equivariant(head):
    pos, vel, types, masses, box, charges, ff_types, bonds = molecular_inputs(periodic=True)
    model = make_upgraded_model(head=head)
    noise = model.com_free_noise(2, types, masses, like=pos)

    dx, v = model.predict(pos, vel, types, box, masses, n_steps=2, noise=noise, sampler="heun",
                          charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=25.0)

    # 1. Periodic lattice shift of one atom across all frames leaves outputs unchanged
    pos_shift = pos.clone()
    pos_shift[:, :, 2] += box[0, 0] - box[0, 2]
    dx_s, v_s = model.predict(pos_shift, vel, types, box, masses, n_steps=2, noise=noise, sampler="heun",
                              charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=25.0)
    assert torch.allclose(dx_s, dx, atol=1e-10) and torch.allclose(v_s, v, atol=1e-10)

    # 2. Permuting atoms together with types, masses, charges, ff_types, and bond indices permutes outputs
    perm = torch.tensor([4, 2, 0, 5, 1, 3], dtype=torch.long)
    inv_perm = torch.argsort(perm)
    bonds_p = inv_perm[bonds]
    noise_p = noise.view(2, -1, 2, 3)[:, perm].reshape(-1, 2, 3)
    dx_p, v_p = model.predict(pos[:, :, perm], vel[:, :, perm], types[perm], box, masses[perm],
                              n_steps=2, noise=noise_p, sampler="heun",
                              charges=charges[perm], ff_types=ff_types[perm], bonds=bonds_p, stride_fs=25.0)
    assert torch.allclose(dx_p, dx[:, perm], atol=1e-10)
    assert torch.allclose(v_p, v[:, perm], atol=1e-10)


def test_chemistry_edges_smooth_across_cutoff():
    """Cutoff crossings remain C^2 smooth (~h^3) even with nonzero charges and bond embeddings."""
    cutoff = 3.5
    model = make_upgraded_model(head="regression", cutoff=cutoff, target="direct", target_scaling="element_std")
    charges = torch.tensor([0.8, -0.8, 0.4], dtype=F64)
    ff_types = torch.tensor([0, 1, 2], dtype=torch.long)
    bonds = torch.tensor([[0, 1]], dtype=torch.long)
    types = torch.tensor([0, 1, 2], dtype=torch.long)
    masses = torch.tensor([7.0, 16.0, 12.0], dtype=F64)
    base = torch.tensor([[[0.0, 0.0, 0.0], [-1.2, 0.2, 0.0], [cutoff, 0.0, 0.0]]], dtype=F64).expand(1, 3, 3, 3).clone()
    vel = torch.zeros_like(base)

    def eval_at(offset):
        p = base.clone()
        p[:, :, 2, 0] += offset
        dx, v = model.predict(p, vel, types, None, masses, charges=charges, ff_types=ff_types, bonds=bonds)
        return torch.cat([dx.flatten(), v.flatten()])

    jumps = [(eval_at(-h) - eval_at(1e-9)).abs().max().item() for h in (1e-2, 1e-3)]
    assert jumps[0] > 0
    order = torch.log10(torch.tensor(jumps[0] / jumps[1])).item()
    assert order > 2.6, (jumps, order)


# ---------------------------------------------------------------- Heun ODE, mass scaling, gradients, FastRollout

def test_heun_integrates_linear_time_flow_exactly():
    """For dy/dtau = a + b * tau, 1-step Heun is exact while 1-step Euler has error 0.5 * b."""
    model = make_upgraded_model(head="flow", geometry_aware=False, target="direct", target_scaling="element_std")
    a_vec = torch.tensor([0.3, -0.2, 0.5], dtype=F64)
    b_vec = torch.tensor([0.8, -0.4, 0.6], dtype=F64)

    class LinearTimeDenoiser(torch.nn.Module):
        def forward(self, cond, graph, y_tau, tau_per_node, candidate_graph_fn=None):
            out = torch.zeros_like(y_tau)
            out[:, 0, :] = a_vec + tau_per_node[:, None] * b_vec
            out[:, 1, :] = -0.5 * a_vec + tau_per_node[:, None] * (0.25 * b_vec)
            return out

    model.denoiser = LinearTimeDenoiser()
    pos, vel, types, masses, box, charges, ff_types, bonds = molecular_inputs(b=1)
    # Equal masses so remove_com is simple; give atom-varying initial noise with zero COM
    masses = torch.ones_like(masses)
    noise = model.com_free_noise(1, types, masses, generator=torch.Generator().manual_seed(4), like=pos)

    # Constant field across atoms is removed by remove_com; use exact multi-step reference vs 1-step
    class StateDependentDenoiser(torch.nn.Module):
        def forward(self, cond, graph, y_tau, tau_per_node, candidate_graph_fn=None):
            # dy/dtau = (1 + 2 * tau) * y0_fixed(node): linear in tau, non-uniform across atoms
            return (1.0 + 2.0 * tau_per_node[:, None, None]) * noise

    model.denoiser = StateDependentDenoiser()
    # Integral of (1 + 2*tau) from 0 to 1 is 2.0, so y(1) = y(0) + 2*noise = 3*noise; Euler 1-step gives 2*noise!
    dx_euler, v_euler = model.predict(pos, vel, types, box, masses, n_steps=1, noise=noise, sampler="euler",
                                      charges=charges, ff_types=ff_types, bonds=bonds)
    dx_heun, v_heun = model.predict(pos, vel, types, box, masses, n_steps=1, noise=noise, sampler="heun",
                                    charges=charges, ff_types=ff_types, bonds=bonds)
    dx_ref, v_ref = model.predict(pos, vel, types, box, masses, n_steps=200, noise=noise, sampler="heun",
                                  charges=charges, ff_types=ff_types, bonds=bonds)
    assert torch.allclose(dx_heun, dx_ref, atol=1e-12)
    assert torch.allclose(v_heun, v_ref, atol=1e-12)
    assert (dx_euler - dx_ref).abs().max() > 0.1


def test_mass_scaling_equipartition_and_gradients():
    pos, vel, types, masses, box, charges, ff_types, bonds = molecular_inputs()
    model = make_upgraded_model(head="flow").train()
    # Roundtrip of targets_to_y and y_to_targets
    dx_true = torch.randn_like(pos[:, 0]) * 0.1
    v_true = torch.randn_like(vel[:, 0]) * 0.02
    y = model.targets_to_y(dx_true, v_true, vel[:, -1], types, masses=masses, stride_fs=30.0)
    dx_rec, v_rec = model.y_to_targets(y, types, vel[:, -1], masses=masses, stride_fs=30.0)
    assert torch.allclose(dx_rec, dx_true, atol=1e-12)
    assert torch.allclose(v_rec, v_true, atol=1e-12)

    # All new modules receive finite nonzero gradients
    loss = model.loss(pos, vel, types, box, masses, dx_true, v_true,
                      charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=30.0)
    loss.backward()
    assert torch.isfinite(loss)
    for mod_name in ("ff_type_embed", "charge_in", "bond_embed", "stride_in", "denoiser"):
        mod = getattr(model, mod_name)
        grads = [p.grad for p in mod.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.abs().max() > 0 for g in grads), mod_name


def test_set_chemistry_works_with_rollout_and_fast_rollout():
    pos, vel, types, masses, box, charges, ff_types, bonds = molecular_inputs(b=1, periodic=True)
    model = make_upgraded_model(head="flow", geometry_aware=False).float().eval()
    pos, vel, masses, box = pos.float(), vel.float(), masses.float(), box.float()
    model.set_chemistry(charges=charges.float(), ff_types=ff_types, bonds=bonds, n_atoms=pos.shape[2])
    assert not any("_chem" in k for k in model.state_dict())

    out = rollout(model, pos, vel, types, box, masses, 2, sampling_steps=2,
                  excluded_pairs=exclusions_from_bonds(bonds, pos.shape[2]))
    assert out["positions"].shape == (2, 1, pos.shape[2], 3) and torch.isfinite(out["positions"]).all()

    fast = FastRollout(model, sampling_steps=2, neighbor="verlet", sampler="heun")
    fout = fast.rollout(pos, vel, types, box, masses, 2)
    assert fout["positions"].shape == (2, 1, pos.shape[2], 3) and torch.isfinite(fout["positions"]).all()


def test_dataset_reads_and_validates_pff_metadata(tmp_path):
    meta = dict(
        version=1, units=dict(charge="e", distance="nm"),
        charges=[0.8, -0.4, -0.4], ff_types=["Li", "O_carb", "O_eth"],
        molecule_ids=[0, 1, 1], original_indices=[0, 1, 2],
        bonds=[[1, 2]], constraints=[], drude_parent_map=[],
    )
    for i in range(2):
        p = tmp_path / f"traj_{i:03d}.h5"
        w = TrajWriter(p, 3, [6.94, 16.0, 16.0], ["Li", "O", "O"],
                       dict(complete=True, dt_fs=2.0, save_every=1, periodic=True))
        for t in range(8):
            x = np.array([[0.2, 0.2, 0.2], [0.5, 0.5, 0.5], [0.62, 0.5, 0.5]]) + 0.001 * t
            v = np.full((3, 3), 0.01 * (i + 1))
            w.add(x, v, np.zeros((3, 3)), 0.0, 0.0, np.eye(3) * 2.0, t * 0.002)
        w.close()
        with h5py.File(p, "r+") as f:
            write_metadata(f, meta, 3)

    (tmp_path / "config.yaml").write_text(yaml.safe_dump({"split": {"train": [0], "val": [1], "test": []}}))
    ds = TrajWindows(tmp_path, "train", 1, 2, m=1)
    assert torch.allclose(ds.charges, torch.tensor([0.8, -0.4, -0.4], dtype=torch.float32))
    assert ds.ff_type_names == ["Li", "O_carb", "O_eth"]
    assert ds.bonds.tolist() == [[1, 2]]
    st = compute_stats(ds, n_samples=4)
    assert "mass_dx_std" in st and "mass_res_v_std" in st and torch.isfinite(st["mass_dx_std"])

    # Altering charges in val trajectory is caught by preflight
    bad_meta = dict(meta, charges=[1.0, -0.5, -0.5])
    with h5py.File(tmp_path / "traj_001.h5", "r+") as f:
        write_metadata(f, bad_meta, 3)
    with pytest.raises(ValueError, match="charges"):
        TrajWindows(tmp_path, "train", 1, 2, m=1)
