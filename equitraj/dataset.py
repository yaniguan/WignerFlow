"""Training windows from ground-truth trajectories.

A window is k context frames plus m future frames, spaced by the stride:
    frames s, s+f, ..., s+(k-1)f  (context)   s+kf, ..., s+(k+m-1)f  (targets)
where f = stride_steps / save_every is the stride in saved frames.
Windows never cross trajectories, and trajectories are split into train/val/test by the
`split` field of the data config (never by frame).

For fast random access, each HDF5 trajectory is converted once to .npy files in Å and Å/fs
(<data_dir>/npy/), which are then memory-mapped.
"""

import os

import h5py
import numpy as np
import torch
import yaml

from equitraj.units import NM_PER_PS_TO_A_PER_FS, NM_TO_A


def npy_cache(h5_path):
    """Convert one trajectory to .npy (positions Å, velocities Å/fs) if not done yet. Returns file paths."""
    d = os.path.join(os.path.dirname(h5_path), "npy")
    base = os.path.join(d, os.path.basename(h5_path).replace(".h5", ""))
    paths = {k: f"{base}_{k}.npy" for k in ["pos", "vel", "meta"]}
    if all(os.path.exists(p) for p in paths.values()):
        return paths
    os.makedirs(d, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        assert f.attrs["complete"], h5_path
        n = f["positions"].shape[0]
        for key, name, scale in [("pos", "positions", NM_TO_A), ("vel", "velocities", NM_PER_PS_TO_A_PER_FS)]:
            out = np.lib.format.open_memmap(paths[key] + ".tmp", mode="w+", dtype=np.float32, shape=f[name].shape)
            for s in range(0, n, 10000):
                out[s:s + 10000] = f[name][s:s + 10000] * scale
            out.flush()
            del out
            os.replace(paths[key] + ".tmp", paths[key])
        meta = {
            "box": f["box"][0] * NM_TO_A,
            "masses": f["masses"][:],
            "elements": np.array([e.decode() for e in f["elements"][:]]),
            "save_every": int(f.attrs["save_every"]),
            "dt_fs": float(f.attrs["dt_fs"]),
            "periodic": bool(f.attrs.get("periodic", True)),  # molecules in vacuum store periodic=False
        }
        np.save(paths["meta"], meta, allow_pickle=True)
    return paths


def preflight(paths, *, reject_estimated_velocities=False):
    """Check HDF5 files before using caches or making training windows.

    Check element order. Optional atom_ids or particle_names also let us check
    the order of atoms with the same element. Without them, we cannot check it.
    Each trajectory can have its own fixed cell.
    """
    reference = None
    for path in paths:
        with h5py.File(path, "r") as f:
            if not f.attrs.get("complete", False):
                raise ValueError(f"{path}: incomplete trajectory")
            x, v = f["positions"], f["velocities"]
            if x.ndim != 3 or x.shape[-1] != 3 or x.shape != v.shape or x.shape[0] < 2 or x.shape[1] < 1:
                raise ValueError(f"{path}: positions/velocities must match (T,N,3), T>=2")
            n = x.shape[1]
            elements, masses = f["elements"][:], f["masses"][:]
            if elements.shape != (n,) or masses.shape != (n,) or not np.isfinite(masses).all() or np.any(masses <= 0):
                raise ValueError(f"{path}: invalid particle elements or positive masses")
            dt, save = float(f.attrs["dt_fs"]), float(f.attrs["save_every"])
            if not np.isfinite(dt) or dt <= 0 or not np.isfinite(save) or save < 1 or save != int(save):
                raise ValueError(f"{path}: invalid cadence")
            times = f["time"][:]
            if times.shape != (len(x),) or not np.isfinite(times).all() or not np.allclose(np.diff(times), dt * save / 1000, rtol=1e-6, atol=1e-10):
                raise ValueError(f"{path}: time spacing disagrees with cadence")
            periodic = bool(f.attrs.get("periodic", True))
            cells = f["box"]
            if cells.shape != (len(x), 3, 3):
                raise ValueError(f"{path}: invalid cell shape")
            first = cells[0]
            if not np.isfinite(first).all() or (periodic and abs(np.linalg.det(first)) < 1e-12):
                raise ValueError(f"{path}: invalid periodic cell")
            for start in range(0, len(x), 10000):
                if not np.allclose(cells[start:start + 10000], first, rtol=1e-6, atol=1e-8):
                    raise ValueError(f"{path}: time-varying cell unsupported")
                if not np.isfinite(x[start:start + 10000]).all() or not np.isfinite(v[start:start + 10000]).all():
                    raise ValueError(f"{path}: nonfinite coordinates or velocities")
            if reject_estimated_velocities and f.attrs.get("velocities_estimated", False):
                raise ValueError(f"{path}: estimated velocities rejected")
            identities = {}
            for key in ("atom_ids", "particle_names"):
                if key in f:
                    identities[key] = f[key][:]
            if reference is not None:
                if not np.array_equal(elements, reference[0]):
                    raise ValueError(f"{path}: atom order/elements differ")
                if not np.allclose(masses, reference[1], rtol=1e-7, atol=1e-8):
                    raise ValueError(f"{path}: masses differ")
                if (dt, save, periodic) != reference[2:5]:
                    raise ValueError(f"{path}: cadence or periodicity differs")
                if identities.keys() != reference[5].keys():
                    raise ValueError(f"{path}: particle identities/order differ")
                for key, value in identities.items():
                    if not np.array_equal(value, reference[5][key]):
                        raise ValueError(f"{path}: particle identities/order differ")
            else:
                reference = (elements, masses, dt, save, periodic, identities)
    if reference is None:
        raise ValueError("no trajectories to validate")


class TrajWindows(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, stride_steps, k, m=1, start_every=1, *, reject_estimated_velocities=False):
        cfg = yaml.safe_load(open(os.path.join(data_dir, "config.yaml")))
        for value in (stride_steps, k, m, start_every):
            if int(value) != value or value < 1:
                raise ValueError("stride, k, m and start_every must be positive integers")
        self.traj_ids = cfg["split"][split]
        if not self.traj_ids:
            raise ValueError(f"empty split: {split}")
        ids = sorted({i for values in cfg["split"].values() for i in values})
        preflight([os.path.join(data_dir, f"traj_{i:03d}.h5") for i in ids],
                  reject_estimated_velocities=reject_estimated_velocities)
        self.pos, self.vel, self.boxes = [], [], []
        for i in self.traj_ids:
            paths = npy_cache(os.path.join(data_dir, f"traj_{i:03d}.h5"))
            self.pos.append(np.load(paths["pos"], mmap_mode="r"))
            self.vel.append(np.load(paths["vel"], mmap_mode="r"))
            meta = np.load(paths["meta"], allow_pickle=True).item()
            self.boxes.append(torch.tensor(meta["box"], dtype=torch.float32))  # NPT gives each trajectory its own box
        self.meta = meta
        self.periodic = meta["periodic"]
        save_every = meta["save_every"]
        if stride_steps % save_every:
            raise ValueError(f"stride {stride_steps} is not a multiple of save_every {save_every}")
        self.f = stride_steps // save_every
        self.k, self.m = k, m
        self.stride_fs = stride_steps * meta["dt_fs"]
        self.elements = sorted(set(meta["elements"]))
        self.types = torch.tensor([self.elements.index(e) for e in meta["elements"]])
        self.masses = torch.tensor(meta["masses"], dtype=torch.float32)
        span = (k + m - 1) * self.f
        self.index = [(t, s) for t in range(len(self.pos)) for s in range(0, len(self.pos[t]) - span, start_every)]
        if not self.index:
            raise ValueError("split contains no complete windows")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        t, s = self.index[i]
        frames = slice(s, s + (self.k + self.m) * self.f, self.f)
        pos = torch.from_numpy(np.array(self.pos[t][frames]))
        vel = torch.from_numpy(np.array(self.vel[t][frames]))
        return pos, vel, self.boxes[t]


def compute_stats(ds, n_samples=2000, seed=0):
    """Per-element scales (isotropic, over xyz), each (n_types,):
        dx_std, v_std:          std of the stride displacement Δx and of the velocity (input scales)
        res_dx_std, res_v_std:  std of Δx - v_t Δt and of v_new - v_t (scales of the residual targets)
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    dx, v, v_next = [], [], []
    for i in idx:
        pos, vel, _ = ds[i]
        dx.append(pos[ds.k] - pos[ds.k - 1])
        v.append(vel[ds.k - 1])
        v_next.append(vel[ds.k])
    dx, v, v_next = torch.stack(dx), torch.stack(v), torch.stack(v_next)  # (S, N, 3)
    res_dx, res_v = dx - v * ds.stride_fs, v_next - v

    def per_type(x):
        return torch.stack([x[:, ds.types == t].pow(2).mean().sqrt() for t in range(len(ds.elements))])

    return {"dx_std": per_type(dx), "v_std": per_type(v), "res_dx_std": per_type(res_dx), "res_v_std": per_type(res_v)}
