"""NumPy binding diagnostics; positions are Å, cells have row lattice vectors.

Atom ordering must agree between trajectories. RMSDs use a fixed reference pose
(default: frame zero). Energies and beta must use reciprocal units. Energy
averages are proxies, not binding free energies. PBC assumes reduced cells,
compact selections smaller than half a cell, and saved motion below half a cell.
"""
from collections.abc import Mapping
from itertools import combinations

import numpy as np

from equitraj.adapters import boxes_array, minimum_image


def _positions(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 3 or x.shape[-1] != 3 or min(x.shape[:2]) == 0 or not np.isfinite(x).all():
        raise ValueError("positions must be finite (T,N,3), with T,N > 0")
    return x


def _indices(indices, n, name="indices", empty=False):
    a = np.asarray(indices)
    if a.ndim != 1 or (a.size and a.dtype.kind not in "iu"):
        raise ValueError(f"{name} must be a one-dimensional integer array")
    a = a.astype(int)
    if (not empty and not len(a)) or np.any((a < 0) | (a >= n)) or len(np.unique(a)) != len(a):
        raise ValueError(f"{name} must contain unique in-range atom indices")
    return a


def _positive(value, name):
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _selections(x, ligand, pocket):
    l = _indices(ligand, x.shape[1], "ligand_indices")
    p = _indices(pocket, x.shape[1], "pocket_indices")
    if np.intersect1d(l, p).size:
        raise ValueError("ligand and pocket selections must be disjoint")
    return l, p


def unwrap_ligand(positions, ligand_indices, pocket_indices, box=None, *, continuous=False):
    """Copy positions, making pocket and ligand whole near the first pocket atom.

    With continuous=True, accumulate ligand-to-pocket relative increments across
    frames; this preserves departures beyond half a cell for COM diagnostics.
    Other atoms are untouched. No bond topology is inferred.
    """
    x = _positions(positions).copy()
    l, p = _selections(x, ligand_indices, pocket_indices)
    if box is None:
        return x
    cells = boxes_array(box, len(x))
    previous = None
    for t, cell in enumerate(cells):
        anchor = x[t, p[0]].copy()
        x[t, p] = anchor + minimum_image(x[t, p] - anchor, cell)
        center = x[t, p].mean(axis=0)
        whole = x[t, l[0]] + minimum_image(x[t, l] - x[t, l[0]], cell)
        relative = minimum_image(whole.mean(axis=0) - center, cell)
        if continuous and previous is not None:
            relative = previous + minimum_image(relative - previous, cell)
        x[t, l] = whole + center + relative - whole.mean(axis=0)
        previous = relative
    return x


def _reference(reference, x):
    r = x[0] if reference is None else np.asarray(reference, dtype=float)
    if r.shape != x.shape[1:] or not np.isfinite(r).all():
        raise ValueError("reference must be a finite (N,3) pose")
    return r


def _aligned(x, r, fit):
    a, b = x[fit].mean(axis=0), r[fit].mean(axis=0)
    u, _, vt = np.linalg.svd((x[fit] - a).T @ (r[fit] - b))
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    return (x - a) @ (u @ correction @ vt) + b


def _rmsd(x, r, fit, measure):
    return np.array([np.sqrt(np.mean(np.sum((_aligned(f, r, fit)[measure] - r[measure]) ** 2, axis=-1)))
                     for f in x])


def ligand_rmsd(positions, ligand_indices, pocket_indices, reference=None, box=None, reference_box=None):
    """Ligand RMSD after proper Kabsch alignment of pocket atoms (no reflection).

    Use at least three noncollinear pocket atoms for a unique orientation.
    reference_box defaults to the first trajectory cell.
    """
    x = _positions(positions)
    l, p = _selections(x, ligand_indices, pocket_indices)
    r = _reference(reference, x)
    rb = reference_box if reference_box is not None else (None if box is None else boxes_array(box, len(x))[0])
    r = unwrap_ligand(r[None], l, p, rb)[0]
    x = unwrap_ligand(x, l, p, box)
    return _rmsd(x, r, p, l)


def ligand_internal_rmsd(positions, ligand_indices, reference=None, box=None, reference_box=None):
    """Ligand RMSD after fitting the ligand itself; removes rigid-body motion."""
    x = _positions(positions)
    l = _indices(ligand_indices, x.shape[1])
    return protein_backbone_rmsd(x, l, reference, box, reference_box)


def protein_backbone_rmsd(positions, backbone_indices, reference=None, box=None, reference_box=None):
    """Self-aligned RMSD of the supplied backbone selection."""
    x = _positions(positions)
    s = _indices(backbone_indices, x.shape[1])
    r = _reference(reference, x).copy()
    x = x.copy()
    if box is not None:
        cells = boxes_array(box, len(x))
        for t, cell in enumerate(cells):
            x[t, s] = x[t, s[0]] + minimum_image(x[t, s] - x[t, s[0]], cell)
        rb = cells[0] if reference_box is None else boxes_array(reference_box, 1)[0]
        r[s] = r[s[0]] + minimum_image(r[s] - r[s[0]], rb)
    elif reference_box is not None:
        rb = boxes_array(reference_box, 1)[0]
        r[s] = r[s[0]] + minimum_image(r[s] - r[s[0]], rb)
    return _rmsd(x, r, s, s)


def _distances(x, a, b, box):
    cells = None if box is None else boxes_array(box, len(x))
    values = []
    for t, frame in enumerate(x):
        dr = frame[a, None] - frame[None, b]
        if cells is not None:
            dr = minimum_image(dr, cells[t])
        values.append(np.linalg.norm(dr, axis=-1))
    return np.asarray(values)


def contact_persistence(positions, ligand_indices, pocket_indices, reference=None, *, cutoff=4.5,
                        box=None, reference_box=None):
    """Persistence of reference contacts (distance <= cutoff), with global pairs.

    Empty reference contact sets have mean persistence 0 and empty fractions.
    """
    x = _positions(positions)
    l, p = _selections(x, ligand_indices, pocket_indices)
    _positive(cutoff, "cutoff")
    r = _reference(reference, x)
    rb = reference_box if reference_box is not None else (None if box is None else boxes_array(box, len(x))[0])
    mask = _distances(r[None], l, p, rb)[0] <= cutoff
    i, j = np.nonzero(mask)
    fractions = (_distances(x, l, p, box) <= cutoff)[:, mask].mean(axis=0)
    return dict(pairs=np.column_stack((l[i], p[j])), persistence=fractions,
                mean_persistence=float(fractions.mean()) if fractions.size else 0.0)


def ligand_com_displacement(positions, ligand_indices, pocket_indices, reference=None, *,
                            masses=None, box=None, reference_box=None, unbinding_threshold=5.0):
    """Pocket-aligned COM displacement and threshold mask, plus first crossing.

    masses is an optional positive (N,) array; absent masses give a centroid.
    Unbinding means displacement >= threshold, without a residence-time rule.
    """
    raw = _positions(positions)
    l, p = _selections(raw, ligand_indices, pocket_indices)
    _positive(unbinding_threshold, "unbinding_threshold")
    weights = np.ones(len(l))
    if masses is not None:
        m = np.asarray(masses, dtype=float)
        if m.shape != (raw.shape[1],) or not np.isfinite(m).all() or np.any(m <= 0):
            raise ValueError("masses must be a finite positive (N,) array")
        weights = m[l]
    r = _reference(reference, raw)
    rb = reference_box if reference_box is not None else (None if box is None else boxes_array(box, len(raw))[0])
    r = unwrap_ligand(r[None], l, p, rb)[0]
    x = unwrap_ligand(raw, l, p, box, continuous=True)
    origin = np.average(r[l], axis=0, weights=weights)
    vectors = np.array([np.average(_aligned(f, r, p)[l], axis=0, weights=weights) - origin for f in x])
    displacement = np.linalg.norm(vectors, axis=-1)
    unbound = displacement >= unbinding_threshold
    hits = np.flatnonzero(unbound)
    return dict(displacement=displacement, vectors=vectors, unbound=unbound,
                first_unbound_frame=int(hits[0]) if len(hits) else None)


def hbond_contact_count(positions, ligand_indices, pocket_indices, *, cutoff=3.5, box=None):
    """Distance-only H-bond-ish counts; supply eligible heavy atoms (e.g. N/O).

    Counts all cross-selection pairs <= cutoff, without hydrogen angles,
    donor/acceptor chemistry, or exclusion of multiple partners.
    """
    x = _positions(positions)
    l, p = _selections(x, ligand_indices, pocket_indices)
    _positive(cutoff, "cutoff")
    return (_distances(x, l, p, box) <= cutoff).sum(axis=(1, 2))


def interaction_energy(positions, energy, *, box=None):
    """Call energy(frame_copy, cell_copy_or_None) once per frame; return scalars.

    The caller supplies the protein-ligand interaction model and energy units.
    """
    x = _positions(positions)
    cells = None if box is None else boxes_array(box, len(x))
    values = np.asarray([energy(f.copy(), None if cells is None else cells[t].copy())
                         for t, f in enumerate(x)], dtype=float)
    return _series(values)


def _series(values):
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not len(v) or not np.isfinite(v).all():
        raise ValueError("energy values must be a nonempty finite scalar series")
    return v


def exponential_average(delta_energy, beta):
    """Stable -log(mean(exp(-beta * delta_energy))) / beta (FEP proxy).

    delta_energy should be paired energy differences sampled in the initial
    ensemble. Applying this to arbitrary interaction energies is only a proxy.
    """
    v = _series(delta_energy)
    beta = _positive(beta, "beta")
    low = v.min()
    with np.errstate(over="ignore", under="ignore"):
        return float(low - np.log(np.exp(-beta * (v - low)).mean()) / beta)


def mean_difference(rollout_energy, reference_energy):
    """Difference of means, rollout minus reference; lengths may differ."""
    return float(_series(rollout_energy).mean() - _series(reference_energy).mean())


def _topology(items, width, n, name):
    a = np.asarray(items)
    if a.size == 0:
        return np.empty((0, width), dtype=int)
    if a.ndim != 2 or a.shape[1] != width or a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer (K,{width}) array")
    if np.any((a < 0) | (a >= n)) or any(len(set(row)) != width for row in a):
        raise ValueError(f"{name} contains invalid atom indices")
    return a.astype(int)


def bond_angle_distributions(positions, bonds, *, angles=None, box=None):
    """Return (T,B) lengths in Å and (T,A) angles in radians.

    If angles is omitted, derive all unique i-j-k angles from the bond graph.
    Zero-length bonds are rejected. Samples are retained instead of binning.
    """
    x = _positions(positions)
    b = _topology(bonds, 2, x.shape[1], "bonds")
    if angles is None:
        neighbors = {}
        for i, j in b:
            neighbors.setdefault(i, set()).add(j)
            neighbors.setdefault(j, set()).add(i)
        angles = [(i, j, k) for j in sorted(neighbors) for i, k in combinations(sorted(neighbors[j]), 2)]
    a = _topology(angles, 3, x.shape[1], "angles")
    cells = None if box is None else boxes_array(box, len(x))
    lengths, theta = [], []
    for t, f in enumerate(x):
        dr = f[b[:, 0]] - f[b[:, 1]]
        u, v = f[a[:, 0]] - f[a[:, 1]], f[a[:, 2]] - f[a[:, 1]]
        if cells is not None:
            dr, u, v = (minimum_image(d, cells[t]) for d in (dr, u, v))
        length = np.linalg.norm(dr, axis=-1)
        denom = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
        if np.any(length == 0) or np.any(denom == 0):
            raise ValueError("zero-length bond prevents geometry analysis")
        lengths.append(length)
        theta.append(np.arccos(np.clip(np.sum(u * v, axis=-1) / denom, -1, 1)))
    return dict(bonds=b, angles=a, bond_lengths=np.asarray(lengths), angles_rad=np.asarray(theta))


def compare_bond_angles(rollout, reference, bonds, *, angles=None, box=None, reference_box=None):
    """Compare means, standard deviations and empirical Wasserstein-1 distances.

    Distribution distances retain length/radian units and support unequal frame
    counts. Each bond or angle is compared separately, without histogram bins.
    """
    r = bond_angle_distributions(rollout, bonds, angles=angles, box=box)
    s = bond_angle_distributions(reference, bonds, angles=r["angles"], box=reference_box)
    differences = {}
    for key in ("bond_lengths", "angles_rad"):
        differences[key + "_mean"] = r[key].mean(axis=0) - s[key].mean(axis=0)
        differences[key + "_std"] = r[key].std(axis=0) - s[key].std(axis=0)
        distances = []
        for u, v in zip(r[key].T, s[key].T):
            u, v = np.sort(u), np.sort(v)
            grid = np.sort(np.concatenate((u, v)))
            cdf_u = np.searchsorted(u, grid[:-1], side="right") / len(u)
            cdf_v = np.searchsorted(v, grid[:-1], side="right") / len(v)
            distances.append(np.sum(np.abs(cdf_u - cdf_v) * np.diff(grid)))
        differences[key + "_wasserstein"] = np.asarray(distances)
    return dict(rollout=r, reference=s, differences=differences)


def compare_binding(rollout, reference, *, ligand_indices=None, pocket_indices=None,
                    backbone_indices=None, box=None, reference_box=None, masses=None,
                    contact_cutoff=4.5, hbond_cutoff=3.5, hbond_ligand_indices=None,
                    hbond_pocket_indices=None, unbinding_threshold=5.0, energy=None,
                    beta=None, bonds=None, angles=None):
    """Compare arrays or mappings containing positions and selection metadata.

    Mappings may provide box, masses, ligand_indices, pocket_indices and
    backbone_indices. Explicit selections take precedence; per-trajectory boxes
    and masses override shared defaults. Both trajectories use reference frame
    zero and its reference contacts. Differences are rollout minus reference
    time means (no frame pairing needed). Backbone defaults to pocket. H-bond
    counts require both explicit eligible-atom selections. Optional energy is
    evaluated with interaction_energy's callable contract.
    """
    rd = rollout if isinstance(rollout, Mapping) else dict(positions=rollout)
    sd = reference if isinstance(reference, Mapping) else dict(positions=reference)
    x, y = _positions(rd["positions"]), _positions(sd["positions"])
    if x.shape[1:] != y.shape[1:]:
        raise ValueError("trajectories must have matching atom counts/order")
    l = ligand_indices if ligand_indices is not None else sd.get("ligand_indices", rd.get("ligand_indices"))
    p = pocket_indices if pocket_indices is not None else sd.get("pocket_indices", rd.get("pocket_indices"))
    bb = backbone_indices if backbone_indices is not None else sd.get("backbone_indices", p)
    xb = rd.get("box", box)
    yb = sd.get("box", reference_box if reference_box is not None else box)
    rb = None if yb is None else boxes_array(yb, len(y))[0]
    if (hbond_ligand_indices is None) != (hbond_pocket_indices is None):
        raise ValueError("both H-bond heavy-atom selections are required")
    report = {}
    for name, data, trajectory, cells in (("rollout", rd, x, xb), ("reference", sd, y, yb)):
        result = dict(ligand_rmsd=ligand_rmsd(trajectory, l, p, y[0], cells, rb),
                      ligand_internal_rmsd=ligand_internal_rmsd(trajectory, l, y[0], cells, rb),
                      backbone_rmsd=protein_backbone_rmsd(trajectory, bb, y[0], cells, rb))
        result["contacts"] = contact_persistence(trajectory, l, p, y[0], cutoff=contact_cutoff,
                                                  box=cells, reference_box=rb)
        result["com"] = ligand_com_displacement(trajectory, l, p, y[0], masses=data.get("masses", masses),
                                                box=cells, reference_box=rb,
                                                unbinding_threshold=unbinding_threshold)
        if hbond_ligand_indices is not None:
            result["hbond_contact_count"] = hbond_contact_count(trajectory, hbond_ligand_indices,
                                                               hbond_pocket_indices, cutoff=hbond_cutoff, box=cells)
        if energy is not None:
            result["interaction_energy"] = interaction_energy(trajectory, energy, box=cells)
        report[name] = result
    def summary(r):
        out = {k: float(r[k].mean()) for k in ("ligand_rmsd", "ligand_internal_rmsd", "backbone_rmsd")}
        out.update(mean_persistence=r["contacts"]["mean_persistence"],
                   com_displacement=float(r["com"]["displacement"].mean()),
                   unbound_fraction=float(r["com"]["unbound"].mean()))
        for k in ("hbond_contact_count", "interaction_energy"):
            if k in r:
                out[k] = float(r[k].mean())
        if beta is not None:
            if "interaction_energy" not in r:
                raise ValueError("beta requires an energy callable")
            out["exponential_energy_proxy"] = exponential_average(r["interaction_energy"], beta)
        return out
    a, b = summary(report["rollout"]), summary(report["reference"])
    report["differences"] = {k: a[k] - b[k] for k in a}
    report["differences"]["contact_persistence"] = (report["rollout"]["contacts"]["persistence"]
                                                     - report["reference"]["contacts"]["persistence"])
    if bonds is not None:
        report["geometry"] = compare_bond_angles(x, y, bonds, angles=angles, box=xb, reference_box=yb)
    return report
