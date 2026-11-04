"""Explicit PFF metadata v1: charge e, constraint distance nm, zero-based indices.

No force-field parameters are inferred or merged when particles are removed.
"""
import json
import numpy as np

VERSION = 1


def validate_metadata(metadata, n_atoms):
    """Return a JSON-compatible validated copy. All v1 fields are required."""
    d = dict(metadata)
    if d.get("version") != VERSION or d.get("units") != {"charge": "e", "distance": "nm"}:
        raise ValueError("expected metadata version 1 and units charge=e, distance=nm")
    for key in ("charges", "ff_types", "molecule_ids", "original_indices"):
        a = np.asarray(d[key])
        if a.shape != (n_atoms,):
            raise ValueError(f"{key} must have one entry per particle")
        if key == "charges" and not np.isfinite(a.astype(float)).all():
            raise ValueError("charges must be finite")
        if key in ("molecule_ids", "original_indices"):
            if a.dtype.kind not in "iu" or np.any(a < 0):
                raise ValueError(f"{key} must be nonnegative integers")
        if key == "ff_types" and (a.dtype.kind not in "US" or np.any(a == "")):
            raise ValueError("ff_types must be nonempty strings")
        d[key] = a.tolist()
    if len(set(d["original_indices"])) != n_atoms:
        raise ValueError("original_indices must be unique")
    for key, width in (("bonds", 2), ("constraints", 3), ("drude_parent_map", 2)):
        rows = np.asarray(d[key])
        if rows.size == 0:
            d[key] = []
            continue
        if rows.ndim != 2 or rows.shape[1] != width:
            raise ValueError(f"invalid {key} shape")
        idx = rows[:, :2].astype(float)
        if not np.isfinite(idx).all() or np.any(idx != np.floor(idx)) or np.any(idx < 0) or np.any(idx >= n_atoms) or np.any(idx[:, 0] == idx[:, 1]):
            raise ValueError(f"invalid {key} particle indices")
        pairs = idx.astype(int)
        canonical = [tuple(sorted(p)) for p in pairs.tolist()]
        if len(set(canonical)) != len(canonical):
            raise ValueError(f"duplicate {key} pairs")
        if key == "constraints":
            lengths = rows[:, 2].astype(float)
            if not np.isfinite(lengths).all() or np.any(lengths <= 0):
                raise ValueError("constraint lengths must be positive nm")
            d[key] = [[int(i), int(j), float(r)] for (i, j), r in zip(pairs, lengths)]
        else:
            d[key] = pairs.tolist()
        if key == "drude_parent_map" and len(set(pairs[:, 0])) != len(pairs):
            raise ValueError("each Drude must have exactly one parent")
    return d


def remap_metadata(metadata, keep):
    """Filter particle fields and remap topology; reject retained dangling Drudes.

    Bonds/constraints touching removed particles are discarded. A retained Drude
    whose parent was removed is an error. original_indices retain provenance.
    """
    keep = np.asarray(keep)
    if keep.ndim != 1 or keep.dtype.kind != "b" or not keep.any():
        raise ValueError("keep must be a nonempty boolean particle mask")
    d = validate_metadata(metadata, len(keep))
    mapping = np.full(len(keep), -1, dtype=int)
    mapping[keep] = np.arange(keep.sum())
    for key in ("charges", "ff_types", "molecule_ids", "original_indices"):
        d[key] = np.asarray(d[key])[keep].tolist()
    for key in ("bonds", "constraints", "drude_parent_map"):
        rows = []
        for row in d[key]:
            i, j = row[:2]
            if key == "drude_parent_map" and keep[i] and not keep[j]:
                raise ValueError("retained Drude has removed parent")
            if keep[i] and keep[j]:
                rows.append([int(mapping[i]), int(mapping[j]), *row[2:]])
        d[key] = rows
    return validate_metadata(d, int(keep.sum()))


def write_metadata(group, metadata, n_atoms):
    d = validate_metadata(metadata, n_atoms)
    group.attrs["pff_metadata"] = json.dumps(d)


def read_metadata(group, n_atoms):
    raw = group.attrs.get("pff_metadata")
    return None if raw is None else validate_metadata(json.loads(raw), n_atoms)
