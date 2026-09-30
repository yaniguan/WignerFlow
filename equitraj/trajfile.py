"""HDF5 trajectory files: one file per trajectory.

Datasets (OpenMM units: nm, ps, kJ/mol):
    positions         (n_frames, n_atoms, 3)  unwrapped
    velocities        (n_frames, n_atoms, 3)
    forces            (n_force_frames, n_atoms, 3)  saved every `force_every` frames
    potential_energy  (n_frames,)
    kinetic_energy    (n_frames,)
    box               (n_frames, 3, 3)
    time              (n_frames,)  ps
    masses            (n_atoms,)   amu
    elements          (n_atoms,)   atomic symbols
Attributes: all metadata (system, force field, dt, save interval, seed, ...).
"""

import h5py
import numpy as np


class TrajWriter:
    """Append frames to an HDF5 file. Frames are buffered and written in blocks."""

    def __init__(self, path, n_atoms, masses, elements, attrs, force_every=1, block=1000):
        self.f = h5py.File(path, "w")
        self.n_atoms = n_atoms
        self.force_every = force_every
        self.block = block
        self.n_frames = 0
        self.buf = {k: [] for k in ["positions", "velocities", "forces", "potential_energy", "kinetic_energy", "box", "time"]}

        chunk = (min(block, 256), n_atoms, 3)
        for name in ["positions", "velocities", "forces"]:
            self.f.create_dataset(name, shape=(0, n_atoms, 3), maxshape=(None, n_atoms, 3), dtype="f4", chunks=chunk)
        for name in ["potential_energy", "kinetic_energy", "time"]:
            self.f.create_dataset(name, shape=(0,), maxshape=(None,), dtype="f8")
        self.f.create_dataset("box", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype="f4")
        self.f.create_dataset("masses", data=np.asarray(masses, dtype="f8"))
        self.f.create_dataset("elements", data=np.asarray(elements, dtype="S3"))
        for k, v in attrs.items():
            self.f.attrs[k] = v
        self.f.attrs["force_every"] = force_every

    def add(self, positions, velocities, forces, potential_energy, kinetic_energy, box, time):
        """Add one frame. `forces` is only stored every `force_every` frames."""
        self.buf["positions"].append(positions)
        self.buf["velocities"].append(velocities)
        if self.n_frames % self.force_every == 0:
            self.buf["forces"].append(forces)
        self.buf["potential_energy"].append(potential_energy)
        self.buf["kinetic_energy"].append(kinetic_energy)
        self.buf["box"].append(box)
        self.buf["time"].append(time)
        self.n_frames += 1
        if len(self.buf["positions"]) >= self.block:
            self.flush()

    def flush(self):
        for name, items in self.buf.items():
            if not items:
                continue
            data = np.asarray(items)
            ds = self.f[name]
            n_old = ds.shape[0]
            ds.resize(n_old + len(data), axis=0)
            ds[n_old:] = data
            items.clear()
        self.f.flush()

    def close(self):
        self.flush()
        self.f.close()


def read_traj(path, start=0, stop=None, stride=1):
    """Read a trajectory into a dict of numpy arrays (OpenMM units) plus 'attrs'."""
    with h5py.File(path, "r") as f:
        out = {}
        for name in ["positions", "velocities", "potential_energy", "kinetic_energy", "box", "time"]:
            out[name] = f[name][start:stop:stride]
        out["forces"] = f["forces"][:]
        out["masses"] = f["masses"][:]
        out["elements"] = np.array([e.decode() for e in f["elements"][:]])
        out["attrs"] = dict(f.attrs)
    return out
