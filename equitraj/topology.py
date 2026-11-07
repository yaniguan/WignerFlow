"""PFF metadata version 1 uses charges in e and distances in nm. Indices start at 0.

Removing particles does not guess or combine force-field values.
"""
import json
import numpy as np

VERSION = 1


def validate_metadata(metadata, n_atoms):
    """Check all required version 1 fields and return a copy for JSON storage."""
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
        if (not np.isfinite(idx).all()
                or np.any(idx != np.floor(idx))
                or np.any(idx < 0)
                or np.any(idx >= n_atoms)
                or np.any(idx[:, 0] == idx[:, 1])):
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
    """Keep selected particles and update their indices in bonds and other pairs.

    Drop bonds and constraints that include removed particles. Raise an error
    if a kept Drude's parent was removed. Leave original_indices unchanged.
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
    if raw is None:
        return None
    return validate_metadata(json.loads(raw), n_atoms)
