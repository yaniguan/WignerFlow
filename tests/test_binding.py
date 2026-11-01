"""Analytic synthetic binding cases; no simulation engine is needed."""
import numpy as np
import pytest

from equitraj.binding import (
    bond_angle_distributions, compare_binding, compare_bond_angles,
    contact_persistence, exponential_average, hbond_contact_count,
    interaction_energy, ligand_com_displacement, ligand_internal_rmsd,
    ligand_rmsd, mean_difference, protein_backbone_rmsd, unwrap_ligand,
)


POCKET = np.array([0, 1, 2, 3])
LIGAND = np.array([4, 5, 6])


def pose():
    return np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.],
                     [3., 1., 1.], [4., 1., 1.], [3., 2., 1.]])


def rigid_frames():
    frames = []
    for angle in [0., .4, 1.8, 3.2]:
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        frames.append(pose() @ rotation + [angle, -2 * angle, .5 * angle])
    return np.array(frames)


def test_proper_rigid_motion_has_zero_rmsd_and_com():
    x = rigid_frames()
    np.testing.assert_allclose(ligand_rmsd(x, LIGAND, POCKET), 0, atol=1e-14)
    np.testing.assert_allclose(ligand_internal_rmsd(x, LIGAND), 0, atol=1e-14)
    np.testing.assert_allclose(protein_backbone_rmsd(x, POCKET), 0, atol=1e-14)
    c = ligand_com_displacement(x, LIGAND, POCKET)
    np.testing.assert_allclose(c["displacement"], 0, atol=1e-14)
    assert c["first_unbound_frame"] is None
    # Proper rotations cannot superpose a reflected tetrahedron.
    reflected = np.stack([pose(), pose() * [-1, 1, 1]])
    assert protein_backbone_rmsd(reflected, POCKET)[1] > .1


def test_ligand_departure_and_deformation_are_distinct():
    x = np.repeat(pose()[None], 3, axis=0)
    x[1:, LIGAND, 0] += np.array([2., 6.])[:, None]
    np.testing.assert_allclose(ligand_rmsd(x, LIGAND, POCKET), [0, 2, 6], atol=1e-14)
    np.testing.assert_allclose(ligand_internal_rmsd(x, LIGAND), 0, atol=1e-14)
    c = ligand_com_displacement(x, LIGAND, POCKET, unbinding_threshold=5)
    np.testing.assert_allclose(c["displacement"], [0, 2, 6], atol=1e-14)
    assert c["unbound"].tolist() == [False, False, True]
    assert c["first_unbound_frame"] == 2
    x[2, 5, 1] += 1
    assert ligand_internal_rmsd(x, LIGAND)[2] > .1
    m = np.ones(7)
    m[5] = 2
    weighted = ligand_com_displacement(x, LIGAND, POCKET, masses=m)
    assert weighted["vectors"][2, 1] == pytest.approx(.5)


def test_reference_contact_persistence_and_heavy_atom_counts():
    x = np.zeros((4, 2, 3))
    x[:, 1, 0] = [2., 3., 5., 6.]
    result = contact_persistence(x, [1], [0], cutoff=3)
    assert result["pairs"].tolist() == [[1, 0]]
    np.testing.assert_allclose(result["persistence"], [.5])
    assert result["mean_persistence"] == .5
    np.testing.assert_array_equal(hbond_contact_count(x, [1], [0], cutoff=3), [1, 1, 0, 0])
    empty = contact_persistence(x, [1], [0], cutoff=1)
    assert empty["pairs"].shape == (0, 2)
    assert empty["persistence"].size == 0
    assert empty["mean_persistence"] == 0


@pytest.mark.parametrize("cell", [np.diag([20., 20., 20.]),
                                      np.array([[20., 0., 0.], [3., 20., 0.], [1., 2., 20.]])])
def test_periodic_images_preserve_binding_geometry(cell):
    x = rigid_frames()
    wrapped = x.copy()
    # Independent lattice shifts split both selections across periodic images.
    wrapped[:, 0] += cell[1]
    wrapped[:, 2] -= cell[2]
    wrapped[:, 4] += cell[0]
    wrapped[:, 5] -= cell[1]
    wrapped[:, 6] += cell[2]
    original = wrapped.copy()
    np.testing.assert_allclose(ligand_rmsd(wrapped, LIGAND, POCKET, reference=pose(), box=cell),
                               0, atol=2e-14)
    np.testing.assert_allclose(ligand_internal_rmsd(wrapped, LIGAND, reference=pose(), box=cell),
                               0, atol=2e-14)
    np.testing.assert_allclose(protein_backbone_rmsd(wrapped, POCKET, reference=pose(), box=cell),
                               0, atol=2e-14)
    a = contact_persistence(x, LIGAND, POCKET)
    b = contact_persistence(wrapped, LIGAND, POCKET, box=cell)
    np.testing.assert_allclose(a["persistence"], b["persistence"])
    np.testing.assert_array_equal(wrapped, original)


def test_continuous_com_tracks_departure_beyond_half_cell():
    x = np.repeat(pose()[None], 7, axis=0)
    shifts = np.arange(7) * 2.
    x[:, LIGAND, 0] += shifts[:, None]
    wrapped = x % 20
    result = ligand_com_displacement(wrapped, LIGAND, POCKET, box=[20, 20, 20],
                                      unbinding_threshold=5)
    np.testing.assert_allclose(result["displacement"], shifts, atol=1e-14)
    assert result["first_unbound_frame"] == 3
    whole = unwrap_ligand(wrapped, LIGAND, POCKET, [20, 20, 20], continuous=True)
    np.testing.assert_allclose(whole, x, atol=1e-14)


def test_energy_callable_and_stable_proxies():
    x = np.repeat(pose()[None], 3, axis=0)
    x[:, 4, 0] = [1, 2, 3]
    seen = []
    def energy(frame, cell):
        seen.append(cell.copy())
        value = frame[4, 0] * 2
        frame[:] = 0  # The callable must not mutate the input trajectory.
        return value
    values = interaction_energy(x, energy, box=[20, 20, 20])
    np.testing.assert_array_equal(values, [2, 4, 6])
    assert len(seen) == 3 and seen[0].shape == (3, 3)
    assert x[0, 4, 0] == 1
    assert mean_difference(values, [1, 3]) == 2
    expected = -np.log(np.exp(-np.array([0., 1., 2.])).mean())
    assert exponential_average([0, 1, 2], 1) == pytest.approx(expected)
    assert exponential_average([10000, 10001, 10002], 1) == pytest.approx(10000 + expected)
    assert exponential_average([-10000, -10000], 2) == -10000
    with pytest.raises(ValueError, match="scalar series"):
        interaction_energy(x, lambda frame, box: [1, 2])


def test_bond_angle_distributions_and_comparison():
    r = np.array([[[1., 0, 0], [0., 0, 0], [0., 1, 0]]])
    x = np.repeat(r, 2, axis=0)
    x[1, 0, 0] = 3
    bonds = [[0, 1], [1, 2]]
    d = bond_angle_distributions(x, bonds)
    assert d["angles"].tolist() == [[0, 1, 2]]
    np.testing.assert_allclose(d["bond_lengths"], [[1, 1], [3, 1]])
    np.testing.assert_allclose(d["angles_rad"], np.pi / 2)
    c = compare_bond_angles(x, r, bonds)
    np.testing.assert_allclose(c["differences"]["bond_lengths_mean"], [1, 0])
    np.testing.assert_allclose(c["differences"]["bond_lengths_std"], [1, 0])
    np.testing.assert_allclose(c["differences"]["bond_lengths_wasserstein"], [1, 0])
    np.testing.assert_allclose(c["differences"]["angles_rad_mean"], 0)
    wrapped = x.copy()
    wrapped[:, 0] += [10, 0, 0]
    pbc = bond_angle_distributions(wrapped, bonds, box=[10, 10, 10])
    np.testing.assert_allclose(pbc["bond_lengths"], d["bond_lengths"])
    np.testing.assert_allclose(pbc["angles_rad"], d["angles_rad"])
    empty = bond_angle_distributions(x, [])
    assert empty["bond_lengths"].shape == (2, 0)
    assert empty["angles_rad"].shape == (2, 0)
    with pytest.raises(ValueError, match="zero-length"):
        bond_angle_distributions(np.zeros((1, 3, 3)), bonds)


def test_compare_binding_common_pose_unequal_lengths_and_metadata():
    reference = dict(positions=np.repeat(pose()[None], 2, axis=0),
                     ligand_indices=LIGAND, pocket_indices=POCKET)
    x = np.repeat(pose()[None], 3, axis=0)
    x[:, LIGAND, 0] += 6
    report = compare_binding(dict(positions=x), reference, hbond_ligand_indices=LIGAND,
                             hbond_pocket_indices=POCKET, energy=lambda f, b: f[4, 0], beta=1,
                             bonds=[[4, 5], [4, 6]])
    differences = report["differences"]
    assert differences["ligand_rmsd"] == pytest.approx(6)
    assert differences["ligand_internal_rmsd"] == pytest.approx(0, abs=1e-14)
    assert differences["com_displacement"] == pytest.approx(6)
    assert differences["unbound_fraction"] == 1
    assert differences["interaction_energy"] == 6
    assert differences["exponential_energy_proxy"] == pytest.approx(6)
    assert differences["mean_persistence"] == -1
    assert differences["hbond_contact_count"] < 0
    np.testing.assert_allclose(report["geometry"]["differences"]["bond_lengths_mean"], 0)
    identical = compare_binding(reference, reference)
    for value in identical["differences"].values():
        np.testing.assert_allclose(value, 0)


@pytest.mark.parametrize("bad", [np.zeros((2, 3)), np.zeros((0, 7, 3)),
                                  np.full((1, 7, 3), np.nan)])
def test_invalid_positions(bad):
    with pytest.raises(ValueError, match="positions"):
        ligand_rmsd(bad, LIGAND, POCKET)


@pytest.mark.parametrize("indices", [[], [4, 4], [-1], [7], [4.5]])
def test_invalid_selections(indices):
    with pytest.raises(ValueError):
        ligand_rmsd(pose()[None], indices, POCKET)


def test_invalid_options_and_topology():
    x = pose()[None]
    with pytest.raises(ValueError, match="disjoint"):
        ligand_rmsd(x, [0, 4], POCKET)
    with pytest.raises(ValueError, match="box"):
        unwrap_ligand(x, LIGAND, POCKET, [0, 10, 10])
    with pytest.raises(ValueError, match="positive"):
        contact_persistence(x, LIGAND, POCKET, cutoff=0)
    with pytest.raises(ValueError, match="masses"):
        ligand_com_displacement(x, LIGAND, POCKET, masses=np.zeros(7))
    with pytest.raises(ValueError, match="beta"):
        exponential_average([1], 0)
    with pytest.raises(ValueError, match="series"):
        mean_difference([], [1])
    with pytest.raises(ValueError, match="bonds"):
        bond_angle_distributions(x, [[0, 8]])
    with pytest.raises(ValueError, match="both H-bond"):
        compare_binding(x, x, ligand_indices=LIGAND, pocket_indices=POCKET,
                        hbond_ligand_indices=LIGAND)
    with pytest.raises(ValueError, match="beta requires"):
        compare_binding(x, x, ligand_indices=LIGAND, pocket_indices=POCKET, beta=1)
