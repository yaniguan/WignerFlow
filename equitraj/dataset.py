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


class TrajWindows(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, stride_steps, k, m=1, start_every=1):
        cfg = yaml.safe_load(open(os.path.join(data_dir, "config.yaml")))
        self.traj_ids = cfg["split"][split]
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
        assert stride_steps % save_every == 0, f"stride {stride_steps} is not a multiple of save_every {save_every}"
        self.f = stride_steps // save_every
        self.k, self.m = k, m
        self.stride_fs = stride_steps * meta["dt_fs"]
        self.elements = sorted(set(meta["elements"]))
        self.types = torch.tensor([self.elements.index(e) for e in meta["elements"]])
        self.masses = torch.tensor(meta["masses"], dtype=torch.float32)
        span = (k + m - 1) * self.f
        self.index = [(t, s) for t in range(len(self.pos)) for s in range(0, len(self.pos[t]) - span, start_every)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        t, s = self.index[i]
        frames = slice(s, s + (self.k + self.m) * self.f, self.f)
        pos = torch.from_numpy(np.array(self.pos[t][frames]))
        vel = torch.from_numpy(np.array(self.vel[t][frames]))
        return pos, vel, self.boxes[t]


def compute_stats(ds, n_samples=2000, seed=0):
    """Per-element std of the stride displacement Δx and of the velocity (isotropic: over xyz).

    Returns (dx_std, v_std), each (n_types,). These set the scale of the model's targets.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n_samples, len(ds)), replace=False)
    dx, v = [], []
    for i in idx:
        pos, vel, _ = ds[i]
        dx.append(pos[ds.k] - pos[ds.k - 1])
        v.append(vel[ds.k - 1])
    dx, v = torch.stack(dx), torch.stack(v)  # (S, N, 3)
    n_types = len(ds.elements)
    dx_std = torch.stack([dx[:, ds.types == t].pow(2).mean().sqrt() for t in range(n_types)])
    v_std = torch.stack([v[:, ds.types == t].pow(2).mean().sqrt() for t in range(n_types)])
    return dx_std, v_std
