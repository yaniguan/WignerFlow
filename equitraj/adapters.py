"""External MD import. Output is unwrapped nm, nm/ps, ps, amu.

Finite differences cannot recover crossings larger than half a cell per saved frame.
Cells contain lattice vectors as rows. Import one source per trajectory index.
"""
from pathlib import Path

import numpy as np
import yaml

# Common MD elements; ASE supplies the full periodic table when installed.
_MASSES = dict(H=1.008, He=4.003, Li=6.94, B=10.81, C=12.011, N=14.007,
               O=15.999, F=18.998, Ne=20.180, Na=22.990, Mg=24.305,
               Al=26.982, Si=28.085, P=30.974, S=32.06, Cl=35.45,
               Ar=39.948, K=39.098, Ca=40.078, Fe=55.845, Zn=65.38,
               Br=79.904, I=126.904)
_Z_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F",
    10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S",
    17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 26: "Fe", 30: "Zn", 35: "Br", 53: "I",
}
_COVALENT_RADII_A = dict(H=0.31, Li=1.28, B=0.84, C=0.76, N=0.71, O=0.66, F=0.57,
                         Si=1.11, P=1.07, S=1.05, Cl=1.02, Br=1.20, I=1.39)
KCAL_TO_KJ = 4.184


def infer_elements(masses):
    """Nearest standard atomic mass; sub-amu particles are labeled X."""
    table = _MASSES
    try:
        from ase.data import atomic_masses, chemical_symbols
        table = {s: m for s, m in zip(chemical_symbols[1:], atomic_masses[1:]) if np.isfinite(m)}
    except ImportError:
        pass
    return np.array([min(table, key=lambda s: abs(table[s] - m)) if m >= 1 else "X"
                     for m in masses])


def boxes_array(box, n_frames):
    """Normalize lengths, a constant matrix, or frame matrices to (T,3,3)."""
    b = np.asarray(box, dtype=float)
    if b.shape == (3,):
        b = np.diag(b)
    if b.shape == (3, 3):
        b = np.broadcast_to(b, (n_frames, 3, 3)).copy()
    if (b.shape != (n_frames, 3, 3) or not np.isfinite(b).all()
            or np.any(np.abs(np.linalg.det(b)) < 1e-12)):
        raise ValueError("box must be nonsingular lengths (3,), matrix (3,3), or (T,3,3)")
    return b


def minimum_image(displacement, box):
    """Closest periodic displacement for reduced triclinic cells.

    Check neighboring lattice images as fractional rounding alone is insufficient
    for tilted cells. Strongly skewed, unreduced cells should be reduced first.
    """
    frac = np.asarray(displacement) @ np.linalg.inv(box)
    base = (frac - np.round(frac)) @ box
    if np.allclose(box, np.diag(np.diag(box)), atol=1e-12, rtol=0):
        return base
    best = base.copy()
    norm = np.sum(best ** 2, axis=-1)
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                candidate = base + np.array([i, j, k]) @ box
                d = np.sum(candidate ** 2, axis=-1)
                mask = d < norm
                best = np.where(mask[..., None], candidate, best)
                norm = np.minimum(norm, d)
    return best


def unwrap_positions(positions, box):
    """Accumulate minimum-image increments, retaining the first frame's image.

    For changing cells this uses the later frame's cell for each increment.
    """
    x = np.asarray(positions, dtype=float)
    if box is None:
        return x.copy()
    cells = boxes_array(box, len(x))
    out = x.copy()
    for t in range(1, len(x)):
        out[t] = out[t - 1] + minimum_image(x[t] - x[t - 1], cells[t])
    return out


def _generic(src):
    if Path(src).suffix.lower() == ".npz":
        with np.load(src, allow_pickle=False) as f:
            return {k: f[k] for k in f.files}
    import h5py

    with h5py.File(src, "r") as f:
        data = {k: f[k][:] for k in f if isinstance(f[k], h5py.Dataset)}
        from equitraj.topology import read_metadata
        if "positions" in data:
            data["particle_metadata"] = read_metadata(f, data["positions"].shape[1])
        return data


def _infer_covalent_metadata(pos_angstrom, elements, charges=None):
    """Infer covalent bonds and atom coordination types (`C_3`, `O_1`, ...) from first-frame geometry."""
    n = len(elements)
    rad = np.array([_COVALENT_RADII_A.get(str(e), 0.75) for e in elements], dtype=float)
    d = np.linalg.norm(pos_angstrom[:, None, :] - pos_angstrom[None, :, :], axis=-1)
    ii, jj = np.where(np.triu((d > 0.25) & (d <= 1.25 * (rad[:, None] + rad[None, :])), k=1))
    bonds = [[int(i), int(j)] for i, j in zip(ii, jj)]
    deg = np.zeros(n, dtype=int)
    for i, j in bonds:
        deg[i] += 1
        deg[j] += 1
    ff_types = [f"{elements[i]}_{deg[i]}" for i in range(n)]
    q = [0.0] * n if charges is None else [float(c) for c in charges]
    return {
        "version": 1,
        "units": {"charge": "e", "distance": "nm"},
        "charges": q,
        "ff_types": ff_types,
        "molecule_ids": [0] * n,
        "original_indices": list(range(n)),
        "bonds": bonds,
        "constraints": [],
        "drude_parent_map": [],
    }


def _read_md17(src):
    """Read public sGDML / rMD17 / MD22 `.npz` quantum MD archives (`R`/`coords` in Å, `z` nuclear charges, `E`, `F`)."""
    with np.load(src, allow_pickle=False) as f:
        keys = set(f.files)
        pos_key = "R" if "R" in keys else ("coords" if "coords" in keys else None)
        z_key = "z" if "z" in keys else ("nuclear_charges" if "nuclear_charges" in keys else None)
        if pos_key is None or z_key is None:
            raise ValueError(f"MD17/rMD17 archive {src} must contain ('R' or 'coords') and ('z' or 'nuclear_charges')")
        pos = np.asarray(f[pos_key], dtype=float)
        z = np.asarray(f[z_key], dtype=int).reshape(-1)
        elems = []
        for zi in z:
            if int(zi) in _Z_TO_SYMBOL:
                elems.append(_Z_TO_SYMBOL[int(zi)])
            else:
                from ase.data import chemical_symbols
                elems.append(chemical_symbols[int(zi)])
        pe = None
        if "E" in keys:
            pe = np.asarray(f["E"], dtype=float).reshape(-1) * KCAL_TO_KJ
        elif "energies" in keys:
            pe = np.asarray(f["energies"], dtype=float).reshape(-1) * KCAL_TO_KJ
        forces = None
        if "F" in keys:
            # kcal/(mol * Å) -> kJ/(mol * nm) = * 4.184 * 10 = * 41.84
            forces = np.asarray(f["F"], dtype=float) * (KCAL_TO_KJ * 10.0)
        elif "forces" in keys:
            forces = np.asarray(f["forces"], dtype=float) * (KCAL_TO_KJ * 10.0)
        meta = _infer_covalent_metadata(pos[0], elems)
        out = {"positions": pos, "elements": elems, "particle_metadata": meta}
        if pe is not None:
            out["potential_energy"] = pe
        if forces is not None:
            out["forces"] = forces
        return out, "angstrom", "angstrom/fs"


def _read(src, topology, fmt):
    if fmt == "md17":
        return _read_md17(src)
    if fmt == "mdtraj":
        import mdtraj as md
        tr = md.load(str(src), top=str(topology) if topology else None)
        atoms = list(tr.topology.atoms)
        return dict(positions=tr.xyz, box=tr.unitcell_vectors,
                    elements=[a.element.symbol if a.element else "X" for a in atoms],
                    masses=[a.element.mass if a.element else 0.0 for a in atoms],
                    names=[a.name for a in atoms]), "nm", "nm/ps"
    if fmt == "ase":
        from ase.io import read
        from ase import units
        frames = read(str(src), index=":")
        if len(frames) < 2:
            raise ValueError("at least two ASE frames required")
        first = frames[0]
        if any(a.get_chemical_symbols() != first.get_chemical_symbols() or
               not np.array_equal(a.get_masses(), first.get_masses()) for a in frames):
            raise ValueError("ASE atom order, species and masses must be constant")
        # ASE momenta imply internal time units; custom velocities arrays are
        # interpreted as Å/fs (override velocity_unit for another convention).
        if all(a.has("momenta") for a in frames):
            v = [a.get_velocities() * units.fs for a in frames]
        elif all("velocities" in a.arrays for a in frames):
            v = [a.arrays["velocities"] for a in frames]
        else:
            v = None
        pbc = np.array([a.pbc for a in frames])
        if np.any(pbc) and not np.all(pbc):
            raise ValueError("partially periodic ASE trajectories are unsupported")
        return dict(positions=[a.positions for a in frames], velocities=v,
                    box=[a.cell.array for a in frames] if np.all(pbc) else None,
                    elements=first.get_chemical_symbols(), masses=first.get_masses()), "angstrom", "angstrom/fs"
    return _generic(src), None, None


def write_data_config(out_dir, n_traj, split=None):
    """Write explicit trajectory-ID splits; default reserves last two for val/test."""
    if n_traj < 1:
        raise ValueError("n_traj must be positive")
    if split is None:
        split = {"train": list(range(max(1, n_traj - 2))),
                 "val": [n_traj - 2] if n_traj >= 3 else [],
                 "test": [n_traj - 1] if n_traj >= 2 else []}
    if any(not isinstance(i, (int, np.integer)) for values in split.values() for i in values):
        raise ValueError("split IDs must be integers")
    split = {name: [int(i) for i in values] for name, values in split.items()}
    ids = [i for values in split.values() for i in values]
    if set(split) != {"train", "val", "test"} or sorted(ids) != list(range(n_traj)):
        raise ValueError("split must partition all trajectory IDs into train/val/test")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "config.yaml"
    with path.open("w") as f:
        yaml.safe_dump({"n_traj": n_traj, "split": split}, f)
    return path


def import_trajectory(src, out_dir, *, topology=None, format=None, unit=None,
                      velocity_unit=None, velocity_source=None, dt_fs=1.0,
                      save_every=1, traj_index=0, periodic=None, system="imported",
                      drop_drude=True, unwrap=True, elements=None, masses=None,
                      box=None, overwrite=False, particle_metadata=None):
    """Import DCD/XTC+PDB, ASE-readable files, or generic HDF5/NPZ.

    Generic inputs require unit='angstrom' or 'nm'. Velocities default to
    angstrom/fs or nm/ps respectively; override velocity_unit when necessary.
    dt_fs is the integration timestep; saved frame spacing is dt_fs*save_every.
    Separate velocity sources are .npy, .npz or HDF5 with a velocities key.
    unwrap=False declares that input positions are already unwrapped.
    Name D* or mass <1 amu identifies particles dropped by default. Keeping
    particles preserves labels (unknown/virtual elements use X), without merging
    their mass into parent atoms. Energies/forces unavailable here are NaN.
    Does not overwrite config.yaml: call write_data_config after all imports.
    """
    if (not np.isfinite(dt_fs) or dt_fs <= 0 or save_every < 1
            or int(save_every) != save_every or traj_index < 0 or int(traj_index) != traj_index):
        raise ValueError("positive dt_fs, integer save_every and nonnegative traj_index required")
    suffix = Path(src).suffix.lower()
    fmt = format or ("mdtraj" if suffix in (".dcd", ".xtc", ".trr") else
                     "generic" if suffix in (".npz", ".h5", ".hdf5") else "ase")
    if fmt not in ("mdtraj", "ase", "generic", "md17"):
        raise ValueError("format must be mdtraj, ase, generic, or md17")
    data, native_unit, native_vunit = _read(src, topology, fmt)
    unit = unit or native_unit
    scales = {"angstrom": 0.1, "A": 0.1, "Å": 0.1, "nm": 1.0}
    if unit not in scales:
        raise ValueError("specify position unit: angstrom or nm")
    scale = scales[unit]
    x = np.asarray(data["positions"], dtype=float) * scale
    if x.ndim != 3 or x.shape[-1] != 3 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError("positions must be finite (T,N,3), with at least two frames")
    n = x.shape[1]
    elem = elements if elements is not None else data.get("elements")
    mass = masses if masses is not None else data.get("masses")
    if elem is not None:
        elem = np.asarray(elem).astype(str)
    if mass is None:
        if elem is None:
            raise ValueError("elements or masses required")
        try:
            mass = [_MASSES[e] for e in elem]
        except KeyError:
            from ase.data import atomic_masses, atomic_numbers
            mass = [atomic_masses[atomic_numbers[e]] for e in elem]
    mass = np.asarray(mass, dtype=float)
    if elem is None:
        elem = infer_elements(mass)
    if mass.shape != (n,) or elem.shape != (n,) or np.any(mass < 0) or not np.isfinite(mass).all():
        raise ValueError("elements and nonnegative finite masses must have one entry per atom")
    names = np.asarray(data.get("names", [""] * n)).astype(str)
    if names.shape != (n,):
        raise ValueError("names must have one entry per atom")
    keep = ~((mass < 1) | np.char.startswith(names, "D")) if drop_drude else np.ones(n, bool)
    if not keep.any():
        raise ValueError("particle filter removed all atoms")
    cell = box if box is not None else data.get("box")
    periodic = cell is not None if periodic is None else periodic
    if periodic and cell is None:
        raise ValueError("periodic import requires a box")
    cells = boxes_array(np.asarray(cell) * scale, len(x)) if periodic else np.zeros((len(x), 3, 3))
    x = x[:, keep]
    if periodic and unwrap:
        x = unwrap_positions(x, cells)
    v = data.get("velocities")
    if velocity_source is not None:
        v = (np.load(velocity_source, allow_pickle=False) if Path(velocity_source).suffix == ".npy"
             else _generic(velocity_source)["velocities"])
    estimated = v is None
    if estimated:
        v = np.gradient(x, dt_fs * save_every / 1000, axis=0, edge_order=2 if len(x) > 2 else 1)
    else:
        vu = velocity_unit or native_vunit or ("nm/ps" if unit == "nm" else "angstrom/fs")
        vs = {"nm/ps": 1.0, "angstrom/fs": 100.0, "A/fs": 100.0, "angstrom/ps": 0.1}
        if vu not in vs:
            raise ValueError("unsupported velocity_unit")
        v = np.asarray(v, dtype=float)
        if v.shape != (len(x), n, 3):
            raise ValueError("velocity shape must match source positions")
        v = v[:, keep] * vs[vu]
    if not np.isfinite(v).all():
        raise ValueError("velocities must be finite")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"traj_{traj_index:03d}.h5"
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    attrs = dict(complete=False, dt_fs=dt_fs, save_every=int(save_every), periodic=bool(periodic),
                 system=system, velocities_estimated=estimated, source=str(src),
                 frame_interval_fs=dt_fs * save_every, positions_unwrapped=True,
                 drude_particles_dropped=bool(drop_drude))
    # TrajWindows trusts existing caches without consulting the HDF5 timestamp.
    for field in ("pos", "vel", "meta"):
        (out_dir / "npy" / f"{path.stem}_{field}.npy").unlink(missing_ok=True)
    if particle_metadata is None:
        particle_metadata = data.get("particle_metadata")
    if particle_metadata is not None:
        from equitraj.topology import remap_metadata
        particle_metadata = remap_metadata(particle_metadata, keep)
    from equitraj.trajfile import TrajWriter

    pe_arr = data.get("potential_energy")
    if pe_arr is not None:
        pe_arr = np.asarray(pe_arr, dtype=float).reshape(len(x))
    forces_arr = data.get("forces")
    if forces_arr is not None:
        forces_arr = np.asarray(forces_arr, dtype=float)[:, keep]
    writer = TrajWriter(path, int(keep.sum()), mass[keep], elem[keep], attrs,
                        particle_metadata=particle_metadata)
    try:
        for t in range(len(x)):
            ke = 0.5 * np.sum(mass[keep, None] * v[t] ** 2)
            pe_t = np.nan if pe_arr is None else float(pe_arr[t])
            f_t = np.full_like(x[t], np.nan) if forces_arr is None else forces_arr[t]
            writer.add(x[t], v[t], f_t, pe_t, ke, cells[t],
                       t * dt_fs * save_every / 1000)
        writer.flush()
        writer.f.attrs["complete"] = True
    finally:
        writer.close()
    return path
