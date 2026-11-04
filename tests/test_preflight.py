import h5py
import numpy as np
import pytest
import yaml
from equitraj.trajfile import TrajWriter
from equitraj.dataset import TrajWindows


def dataset(tmp_path):
    for i in range(2):
        w = TrajWriter(tmp_path / f"traj_{i:03d}.h5", 2, [7, 16], ["Li", "O"],
                       dict(complete=True, dt_fs=1., save_every=2, periodic=True))
        for t in range(6):
            w.add(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)),
                  0, 0, np.eye(3) * (2 + i), t * .002)
        w.close()
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(
        {"split": {"train": [0], "val": [1], "test": []}}))
    return lambda **kw: TrajWindows(tmp_path, "train", 2, 2, **kw)


def test_constant_different_cells(tmp_path):
    load = dataset(tmp_path)
    ds = load()
    assert len(ds) == 4
    assert ds.meta["elements"].tolist() == ["Li", "O"]


@pytest.mark.parametrize("change,match", [
    ("elements", "order"), ("masses", "masses"), ("cadence", "spacing"),
    ("box", "time-varying"), ("periodic", "periodicity"), ("identity", "identities")])
def test_all_splits_checked_before_cache(tmp_path, change, match):
    load = dataset(tmp_path)
    load()  # create a cache, then corrupt even a trajectory outside train
    with h5py.File(tmp_path / "traj_001.h5", "r+") as f:
        if change == "elements":
            f["elements"][:] = [b"O", b"Li"]
        elif change == "masses":
            f["masses"][0] = 8
        elif change == "cadence":
            f.attrs["dt_fs"] = 2
        elif change == "box":
            f["box"][2] = np.eye(3) * 4
        elif change == "periodic":
            f.attrs["periodic"] = False
        else:
            f.create_dataset("atom_ids", data=[1, 0])
    with pytest.raises(ValueError, match=match):
        load()


def test_window_arguments_and_estimates(tmp_path):
    load = dataset(tmp_path)
    for kw in ({"start_every": 0}, {"m": 0}):
        with pytest.raises(ValueError):
            load(**kw)
    with pytest.raises(ValueError):
        TrajWindows(tmp_path, "test", 2, 2)
    with pytest.raises(ValueError):
        TrajWindows(tmp_path, "train", 3, 2)
    with h5py.File(tmp_path / "traj_001.h5", "r+") as f:
        f.attrs["velocities_estimated"] = True
    load()
    with pytest.raises(ValueError, match="estimated"):
        load(reject_estimated_velocities=True)
