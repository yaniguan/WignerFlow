import copy
import numpy as np
import pytest
from equitraj.topology import validate_metadata, remap_metadata
from equitraj.adapters import import_trajectory
from equitraj.trajfile import read_traj


def metadata():
    return dict(version=1, units=dict(charge="e", distance="nm"),
                charges=[1., -.2, -.8], ff_types=["Li", "D", "O"],
                molecule_ids=[0, 0, 1], original_indices=[10, 11, 12],
                bonds=[[0, 1]], constraints=[[0, 2, .2]],
                drude_parent_map=[[1, 0]])


def test_filter():
    d = remap_metadata(metadata(), np.array([True, False, True]))
    assert d["charges"] == [1, -.8]
    assert d["original_indices"] == [10, 12]
    assert d["constraints"] == [[0, 1, .2]]
    assert d["bonds"] == d["drude_parent_map"] == []
    with pytest.raises(ValueError, match="parent"):
        remap_metadata(metadata(), np.array([False, True, True]))


@pytest.mark.parametrize("key,value", [
    ("version", 2), ("charges", [1]), ("charges", [1, 0, np.nan]),
    ("bonds", [[0, 3]]), ("constraints", [[0, 2, -1]]),
    ("original_indices", [0, 0, 1]), ("molecule_ids", [0., 0., 1.]),
    ("units", {"charge": "e", "distance": "angstrom"})])
def test_invalid(key, value):
    d = copy.deepcopy(metadata())
    d[key] = value
    with pytest.raises(ValueError):
        validate_metadata(d, 3)


def test_import_roundtrip(tmp_path):
    src = tmp_path / "source.npz"
    np.savez(src, positions=np.zeros((3, 3, 3)),
             masses=[7., .4, 16.], elements=["Li", "X", "O"])
    path = import_trajectory(src, tmp_path / "data", unit="nm",
                             particle_metadata=metadata(), periodic=False)
    got = read_traj(path)
    assert got["particle_metadata"] == remap_metadata(metadata(), np.array([True, False, True]))
    legacy = import_trajectory(src, tmp_path / "legacy", unit="nm", periodic=False)
    assert "particle_metadata" not in read_traj(legacy)
    second = import_trajectory(path, tmp_path / "second", unit="nm",
                               periodic=False, drop_drude=False)
    assert read_traj(second)["particle_metadata"] == got["particle_metadata"]
