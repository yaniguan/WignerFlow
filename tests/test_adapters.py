"""Synthetic import tests; optional formats skip without their readers."""
import h5py
import numpy as np
import pytest
import yaml

from equitraj.adapters import (import_trajectory, infer_elements, minimum_image,
                               unwrap_positions, write_data_config)
from equitraj.trajfile import read_traj


def test_npz_units_unwrap_species_and_velocity(tmp_path):
    x = np.zeros((5, 2, 3))
    x[:, :, 0] = (np.arange(5)[:, None] + np.array([8, 3])) % 10
    src = tmp_path / "input.npz"
    np.savez(src, positions=x, box=np.eye(3) * 10, masses=[22.99, 35.45])
    path = import_trajectory(src, tmp_path / "out", unit="angstrom", dt_fs=1, save_every=2)
    d = read_traj(path)
    assert list(d["elements"]) == ["Na", "Cl"]
    assert np.allclose(np.diff(d["positions"][:, :, 0], axis=0), 0.1)
    assert np.allclose(d["velocities"][:, :, 0], 50)  # .1 nm / .002 ps
    assert np.allclose(d["time"], np.arange(5) * .002)
    assert d["attrs"]["complete"] and d["attrs"]["velocities_estimated"]
    assert d["attrs"]["save_every"] == 2 and d["attrs"]["periodic"]
    with pytest.raises(FileExistsError):
        import_trajectory(src, tmp_path / "out", unit="angstrom")
    config = write_data_config(tmp_path / "out", 3)
    assert yaml.safe_load(config.read_text())["split"] == {"train": [0], "val": [1], "test": [2]}
    from equitraj.dataset import TrajWindows
    ds = TrajWindows(tmp_path / "out", "train", stride_steps=2, k=1)
    pos, vel, cell = ds[0]
    assert np.allclose(pos.numpy(), d["positions"][:2] * 10)
    assert np.allclose(vel.numpy(), d["velocities"][:2] * .01)
    assert cell.shape == (3, 3) and ds.elements == ["Cl", "Na"]


def test_generic_velocity_and_drude_filter(tmp_path):
    src = tmp_path / "input.h5"
    with h5py.File(src, "w") as f:
        f["positions"] = np.zeros((3, 4, 3))
        f["velocities"] = np.ones((3, 4, 3)) * .02
        f["elements"] = np.array(["Na", "Cl", "X", "X"], dtype="S3")
        f["masses"] = [22.99, 35.45, .4, 2]
        f["names"] = np.array(["Na", "Cl", "DP", "D2"], dtype="S3")
    path = import_trajectory(src, tmp_path / "out", unit="angstrom")
    d = read_traj(path)
    assert d["positions"].shape == (3, 2, 3)
    assert np.allclose(d["velocities"], 2)
    assert not d["attrs"]["velocities_estimated"]
    assert not d["attrs"]["periodic"]
    assert np.allclose(d["kinetic_energy"], .5 * (22.99 + 35.45) * 3 * 4)
    path = import_trajectory(src, tmp_path / "keep", unit="angstrom", drop_drude=False)
    assert read_traj(path)["positions"].shape[1] == 4


def test_separate_velocity_and_nm(tmp_path):
    src = tmp_path / "input.npz"
    np.savez(src, positions=np.zeros((3, 2, 3)), elements=["Na", "Cl"])
    velocity = tmp_path / "vel.npy"
    np.save(velocity, np.ones((3, 2, 3)) * 2)
    d = read_traj(import_trajectory(src, tmp_path / "out", unit="nm", velocity_source=velocity))
    assert np.allclose(d["velocities"], 2)
    assert np.allclose(d["masses"], [22.99, 35.45])


def test_triclinic_unwrap():
    cell = np.array([[10., 0, 0], [3., 9, 0], [1, 2, 8]])
    frac = np.array([[[.9, .9, .9]], [[1.05, 1.1, 1.05]], [[1.2, 1.3, 1.2]]])
    wrapped = (frac % 1) @ cell
    assert np.allclose(unwrap_positions(wrapped, cell), frac @ cell)
    d = np.array([[.2, -.3, .1]]) @ cell
    assert np.allclose(minimum_image(d + np.array([2, -1, 3]) @ cell, cell), d)
    assert list(infer_elements([22.99, 35.45, .4])) == ["Na", "Cl", "X"]


def test_import_rejects_bad_input_and_split(tmp_path):
    src = tmp_path / "input.npz"
    np.savez(src, positions=np.zeros((3, 2, 3)), masses=[23, 35])
    with pytest.raises(ValueError, match="unit"):
        import_trajectory(src, tmp_path)
    with pytest.raises(ValueError):
        write_data_config(tmp_path, 2, {"train": [0], "val": [0], "test": [1]})


def test_extxyz_roundtrip(tmp_path):
    ase = pytest.importorskip("ase")
    from ase import units
    from ase.io import write
    frames = []
    for t in range(3):
        atoms = ase.Atoms("NaCl", positions=[[t * .01, 0, 0], [3 + t * .01, 0, 0]],
                          cell=[10, 10, 10], pbc=True)
        # ASE internal velocity units: displacement per fs divided by units.fs.
        atoms.set_velocities(np.full((2, 3), .001 / units.fs))
        frames.append(atoms)
    src = tmp_path / "salt.extxyz"
    write(src, frames, format="extxyz")
    d = read_traj(import_trajectory(src, tmp_path / "out"))
    assert list(d["elements"]) == ["Na", "Cl"]
    assert np.allclose(d["positions"], np.array([a.positions for a in frames]) * .1)
    assert np.allclose(d["velocities"], .1)
    assert not d["attrs"]["velocities_estimated"]


def test_ase_missing_momenta_estimates(tmp_path):
    ase = pytest.importorskip("ase")
    from ase.io import write
    frames = [ase.Atoms("Ar", positions=[[t * .02, 0, 0]]) for t in range(4)]
    src = tmp_path / "gas.xyz"
    write(src, frames)
    d = read_traj(import_trajectory(src, tmp_path / "out", dt_fs=2))
    assert d["attrs"]["velocities_estimated"]
    assert np.allclose(d["velocities"][:, 0, 0], 1)


def test_dcd_roundtrip(tmp_path):
    md = pytest.importorskip("mdtraj")
    top = md.Topology()
    chain = top.add_chain()
    res = top.add_residue("ION", chain)
    top.add_atom("Na", md.element.sodium, res)
    top.add_atom("Cl", md.element.chlorine, res)
    x = np.zeros((4, 2, 3), dtype=np.float32)
    x[:, :, 0] = np.arange(4)[:, None] * .01 + [0, .3]
    tr = md.Trajectory(x, top, unitcell_lengths=np.full((4, 3), 2.),
                       unitcell_angles=np.full((4, 3), 90.))
    pdb, dcd = tmp_path / "top.pdb", tmp_path / "traj.dcd"
    tr[0].save_pdb(str(pdb))
    tr.save_dcd(str(dcd))
    d = read_traj(import_trajectory(dcd, tmp_path / "out", topology=pdb, dt_fs=10))
    assert np.allclose(d["positions"], x, atol=1e-6)
    assert np.allclose(d["velocities"][:, :, 0], 1, atol=1e-5)
    assert list(d["elements"]) == ["Na", "Cl"]
