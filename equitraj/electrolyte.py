"""Polarizable electrolyte (`MU-PFF` / WMI `02_Na_MD`) ingestion, coarse-grained reduction,
Onsager / Nernst-Einstein transport evaluation, and differentiable conductivity calibration.

Bridges `02_Na_MD/build_outputs/<system>` (`pol_ff.xml` + `system.pdb` + `virtual_sites.json` +
`config_augmented.json`) directly into WignerFlow at three selectable resolutions:
  1. `molecular`   (8,264 sites ->   610 molecules: COMs of EC, DMC, Na+, PF6-; >1,000 ns/day batched on H100)
  2. `polar_beads` (8,264 sites -> 1,140 beads: splits dipolar solvents into [carbonyl-O, scaffold-COM] + ions)
  3. `all_atom`    (8,264 sites -> 6,144 physical atoms: strips 2,120 `LP*` virtual sites, folds charges)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from equitraj.adapters import minimum_image as minimum_image_np, unwrap_positions
from equitraj.trajfile import TrajWriter
from equitraj.units import NM_PER_PS_TO_A_PER_FS, NM_TO_A

E_CHARGE_C = 1.602176634e-19
KB_J_PER_K = 1.380649e-23
AMU_TO_GRAM = 1.66053906660e-24
# Prefactor converting (N_e * D [Å^2/ps]) / (V [Å^3] * T [K]) -> conductivity in mS/cm (= 10 * S/m):
# e^2 * 1e-8 [m^2/s per Å^2/ps] / (1e-30 [m^3/Å^3] * k_B) * 10 [mS/cm per S/m]
COND_PREFACTOR_MS_CM = (E_CHARGE_C ** 2 * 1e-8 * 10.0) / (1e-30 * KB_J_PER_K)


def load_wmi_build_dir(build_dir, *, target_density_g_cm3=1.2516):
    """Parse a `02_Na_MD/build_outputs/<system>` directory into physical atoms, molecules, and charges.

    Folds massless lone-pair (`LP*` / `virtual_sites.json`) monopole charges `c0` and polarizabilities
    onto their parent heavy atoms before stripping virtual sites so every molecule preserves exact
    net charge (`EC = 0`, `DMC = 0`, `Na+ = +1`, `PF6- = -1`).
    """
    root = Path(build_dir)
    cfg_path = root / "config_augmented.json"
    if not cfg_path.is_file():
        cfg_path = root / "config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}

    ff_xml = root / "system" / "openmm" / "pol_ff.xml"
    pdb_path = root / "system" / "openmm" / "system.pdb"
    vs_path = root / "system" / "virtual_sites.json"
    if not ff_xml.is_file() or not pdb_path.is_file():
        raise FileNotFoundError(f"Missing pol_ff.xml or system.pdb under {root / 'system' / 'openmm'}")

    # 1. Parse pol_ff.xml: AtomTypes, Residues, Multipole c0 charges, and Polarize polarizabilities
    tree = ET.parse(ff_xml)
    ff_root = tree.getroot()
    atom_types = {}
    for el in ff_root.findall("./AtomTypes/Type"):
        tid = el.attrib["name"]
        atom_types[tid] = {
            "class": el.attrib.get("class", tid),
            "element": el.attrib["element"],
            "mass": float(el.attrib["mass"]),
            "c0": 0.0,
            "polarizability": 0.0,
        }
    for el in ff_root.findall("./AmoebaMultipoleForce/Multipole"):
        tid = el.attrib["type"]
        if tid in atom_types:
            atom_types[tid]["c0"] = float(el.attrib.get("c0", 0.0))
    for el in ff_root.findall("./AmoebaMultipoleForce/Polarize"):
        tid = el.attrib["type"]
        if tid in atom_types:
            atom_types[tid]["polarizability"] = float(el.attrib.get("polarizability", 0.0))

    residue_templates = {}
    for res_el in ff_root.findall("./Residues/Residue"):
        rname = res_el.attrib["name"]
        r_atoms = [(a.attrib["name"], a.attrib["type"]) for a in res_el.findall("Atom")]
        name_to_local = {nm: idx for idx, (nm, _) in enumerate(r_atoms)}
        r_bonds = [
            (name_to_local[b.attrib["atomName1"]], name_to_local[b.attrib["atomName2"]])
            for b in res_el.findall("Bond")
        ]
        residue_templates[rname] = {"atoms": r_atoms, "bonds": r_bonds}

    # 2. Parse system.pdb atoms & initial box
    pdb_box_A = None
    raw_atoms = []
    mol_blocks = []
    cur_mol = []
    cur_res_key = None
    with pdb_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("CRYST1"):
                lx, ly, lz = float(line[6:15]), float(line[15:24]), float(line[24:33])
                pdb_box_A = np.diag([lx, ly, lz])
            elif line.startswith(("ATOM  ", "HETATM")):
                aname = line[12:16].strip()
                rname = line[17:21].strip()
                res_seq = line[22:26].strip()
                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                elem = line[76:78].strip() or aname[0]
                g_idx = len(raw_atoms)
                raw_atoms.append({"name": aname, "resname": rname, "xyz": [x, y, z], "pdb_elem": elem})
                if cur_res_key is not None and (rname, res_seq) != cur_res_key and cur_mol:
                    mol_blocks.append(cur_mol)
                    cur_mol = []
                cur_res_key = (rname, res_seq)
                cur_mol.append(g_idx)
            elif line.startswith("TER") and cur_mol:
                mol_blocks.append(cur_mol)
                cur_mol = []
                cur_res_key = None
    if cur_mol:
        mol_blocks.append(cur_mol)

    n_total = len(raw_atoms)
    raw_type_ids = [""] * n_total
    raw_masses = np.zeros(n_total, dtype=np.float64)
    raw_charges = np.zeros(n_total, dtype=np.float64)
    raw_pol = np.zeros(n_total, dtype=np.float64)
    raw_elems = [""] * n_total
    raw_mol_id = np.zeros(n_total, dtype=np.int64)
    mol_resnames = []
    global_bonds_raw = []

    for m_id, atom_indices in enumerate(mol_blocks):
        rname = raw_atoms[atom_indices[0]]["resname"]
        mol_resnames.append(rname)
        tmpl = residue_templates[rname]
        if len(atom_indices) != len(tmpl["atoms"]):
            raise ValueError(f"Molecule {m_id} ({rname}) has {len(atom_indices)} atoms != template {len(tmpl['atoms'])}")
        for local_i, g_i in enumerate(atom_indices):
            _, tid = tmpl["atoms"][local_i]
            info = atom_types[tid]
            raw_type_ids[g_i] = tid
            raw_masses[g_i] = info["mass"]
            raw_charges[g_i] = info["c0"]
            raw_pol[g_i] = info["polarizability"]
            raw_elems[g_i] = info["element"]
            raw_mol_id[g_i] = m_id
        for loc_a, loc_b in tmpl["bonds"]:
            global_bonds_raw.append((atom_indices[loc_a], atom_indices[loc_b]))

    # 3. Fold lone-pair virtual sites (`virtual_sites.json` or `LP*` / `mass < 1`) onto parent heavy atoms
    lp_to_parent = {}
    if vs_path.is_file():
        vs_data = json.loads(vs_path.read_text(encoding="utf-8"))
        for entry in vs_data.get("sites", []):
            lp_i = int(entry["lp_index_1based"]) - 1
            par_i = int(entry["parent_index_1based"]) - 1
            lp_to_parent[lp_i] = par_i
    else:
        adj = [[] for _ in range(n_total)]
        for a, b in global_bonds_raw:
            adj[a].append(b)
            adj[b].append(a)
        for i in range(n_total):
            if raw_atoms[i]["name"].startswith("LP") or raw_masses[i] < 0.5:
                if len(adj[i]) == 1:
                    lp_to_parent[i] = adj[i][0]

    folded_charges = raw_charges.copy()
    folded_pol = raw_pol.copy()
    for lp_i, par_i in lp_to_parent.items():
        folded_charges[par_i] += folded_charges[lp_i]
        folded_pol[par_i] += folded_pol[lp_i]
        folded_charges[lp_i] = 0.0
        folded_pol[lp_i] = 0.0

    keep_mask = np.ones(n_total, dtype=bool)
    for lp_i in lp_to_parent:
        keep_mask[lp_i] = False
    phys_indices = np.nonzero(keep_mask)[0]
    old_to_phys = np.full(n_total, -1, dtype=np.int64)
    old_to_phys[phys_indices] = np.arange(len(phys_indices))

    phys_bonds = []
    for a, b in global_bonds_raw:
        pa, pb = int(old_to_phys[a]), int(old_to_phys[b])
        if pa >= 0 and pb >= 0:
            phys_bonds.append([min(pa, pb), max(pa, pb)])

    # 4. Classify molecule species (`S00`, `S01`, ..., `cation`, `anion`) matching 02_Na_MD/results_mu.json
    unique_res = []
    for rn in mol_resnames:
        if rn not in unique_res:
            unique_res.append(rn)

    res_species = {}
    solvent_counter = 0
    for rn in unique_res:
        first_m = mol_resnames.index(rn)
        m_atoms = [i for i in mol_blocks[first_m] if keep_mask[i]]
        q_net = float(np.round(folded_charges[m_atoms].sum(), 4))
        m_tot = float(raw_masses[m_atoms].sum())
        count = sum(1 for r in mol_resnames if r == rn)
        if q_net > 0.5:
            label, role, short = "cation", "cation", raw_elems[m_atoms[0]] if len(m_atoms) == 1 else "Cat"
        elif q_net < -0.5:
            label, role, short = "anion", "anion", f"{raw_elems[m_atoms[0]]}X"[:3] if len(m_atoms) > 1 else "Ani"
            if rn == "MOL4" and set(raw_elems[i] for i in m_atoms) == {"P", "F"}:
                short = "PF6"
        else:
            label = f"S{solvent_counter:02d}"
            role = "solvent"
            short = label
            solvent_counter += 1
        res_species[rn] = {
            "resname": rn,
            "species": label,
            "short_elem": short,
            "role": role,
            "charge": q_net,
            "mass_amu": m_tot,
            "count": count,
            "n_phys_atoms": len(m_atoms),
        }

    mol_phys_atoms = [[int(old_to_phys[i]) for i in blk if keep_mask[i]] for blk in mol_blocks]
    total_mass_amu = float(raw_masses[phys_indices].sum())
    vol_A3 = (total_mass_amu * AMU_TO_GRAM / float(target_density_g_cm3)) * 1e24
    eq_box_L_A = float(vol_A3 ** (1.0 / 3.0))

    return {
        "build_dir": str(root),
        "config": cfg,
        "temperature_K": float(cfg.get("temperature", 298.15)),
        "n_total_sites": n_total,
        "n_virtual_sites": len(lp_to_parent),
        "n_phys_atoms": len(phys_indices),
        "n_molecules": len(mol_blocks),
        "phys_indices": phys_indices,
        "lp_to_parent": lp_to_parent,
        "pdb_positions_A": np.array([a["xyz"] for a in raw_atoms], dtype=np.float64),
        "pdb_box_A": pdb_box_A,
        "equilibrium_box_L_A": eq_box_L_A,
        "total_mass_amu": total_mass_amu,
        "phys_elements": [raw_elems[i] for i in phys_indices],
        "phys_atom_names": [raw_atoms[i]["name"] for i in phys_indices],
        "phys_ff_types": [f"wmi_{raw_type_ids[i]}" for i in phys_indices],
        "phys_masses": raw_masses[phys_indices],
        "phys_charges": folded_charges[phys_indices],
        "phys_polarizabilities": folded_pol[phys_indices],
        "phys_mol_ids": raw_mol_id[phys_indices],
        "phys_bonds": phys_bonds,
        "mol_resnames": mol_resnames,
        "mol_phys_atoms": mol_phys_atoms,
        "species_table": res_species,
    }


def _unwrap_intramolecular_A(pos_phys_A, boxes_A, mol_phys_atoms):
    """Make every molecule whole relative to its first atom in each frame before COM or affine scaling."""
    out = np.array(pos_phys_A, dtype=np.float64, copy=True)
    t_len = out.shape[0]
    for t in range(t_len):
        cell = boxes_A[t]
        if np.abs(np.linalg.det(cell)) < 1e-12:
            continue
        for m_atoms in mol_phys_atoms:
            if len(m_atoms) > 1:
                anchor = out[t, m_atoms[0]]
                disp = minimum_image_np(out[t, m_atoms[1:]] - anchor[None, :], cell)
                out[t, m_atoms[1:]] = anchor[None, :] + disp
    return out


def reduce_wmi_frames(pos_raw_A, vel_raw_A_fs, boxes_A, wmi, *,
                      resolution="molecular", canonicalize_npt_box=True):
    """Reduce `(T, 8264, 3)` raw WMI frames to `molecular` (610), `polar_beads` (1140), or `all_atom` (6144).

    When `canonicalize_npt_box=True`, fluctuating NPT cells `L(t)` are mapped to the mean-volume cubic
    box `<V>^(1/3)` by scaling molecule centroids while keeping intramolecular bond vectors untouched.
    """
    if resolution not in ("molecular", "polar_beads", "all_atom"):
        raise ValueError(f"resolution must be 'molecular', 'polar_beads', or 'all_atom', got {resolution!r}")
    t_len = pos_raw_A.shape[0]
    if pos_raw_A.shape[1] == wmi["n_total_sites"]:
        keep = wmi["phys_indices"]
        pos_p = np.asarray(pos_raw_A[:, keep], dtype=np.float64)
        vel_p = None if vel_raw_A_fs is None else np.asarray(vel_raw_A_fs[:, keep], dtype=np.float64)
    elif pos_raw_A.shape[1] == wmi["n_phys_atoms"]:
        pos_p = np.asarray(pos_raw_A, dtype=np.float64)
        vel_p = None if vel_raw_A_fs is None else np.asarray(vel_raw_A_fs, dtype=np.float64)
    else:
        raise ValueError(f"frame atom count {pos_raw_A.shape[1]} matches neither {wmi['n_total_sites']} nor {wmi['n_phys_atoms']}")

    boxes = np.asarray(boxes_A, dtype=np.float64)
    if boxes.ndim == 2:
        boxes = np.broadcast_to(boxes, (t_len, 3, 3)).copy()
    pos_p = _unwrap_intramolecular_A(pos_p, boxes, wmi["mol_phys_atoms"])

    masses_p = wmi["phys_masses"]
    charges_p = wmi["phys_charges"]
    mol_atoms = wmi["mol_phys_atoms"]

    # Optional rigid-centroid affine NPT canonicalization to mean volume <V>^(1/3)
    if canonicalize_npt_box:
        vols = np.abs(np.linalg.det(boxes))
        l_bar = float(np.mean(vols) ** (1.0 / 3.0))
        target_box = np.eye(3, dtype=np.float64) * l_bar
        for t in range(t_len):
            l_t = float(vols[t] ** (1.0 / 3.0))
            if l_t > 0 and abs(l_t - l_bar) > 1e-9:
                ratio = l_bar / l_t
                for m_idx in mol_atoms:
                    w = masses_p[m_idx] / masses_p[m_idx].sum()
                    com = (w[:, None] * pos_p[t, m_idx]).sum(axis=0)
                    pos_p[t, m_idx] = com[None, :] * ratio + (pos_p[t, m_idx] - com[None, :])
        boxes = np.broadcast_to(target_box, (t_len, 3, 3)).copy()

    if resolution == "all_atom":
        meta = {
            "version": 1,
            "units": {"charge": "e", "distance": "nm"},
            "charges": charges_p.tolist(),
            "ff_types": list(wmi["phys_ff_types"]),
            "molecule_ids": wmi["phys_mol_ids"].tolist(),
            "original_indices": wmi["phys_indices"].tolist(),
            "bonds": list(wmi["phys_bonds"]),
            "constraints": [],
            "drude_parent_map": [],
        }
        species_per_mol = [wmi["species_table"][rn]["species"] for rn in wmi["mol_resnames"]]
        return {
            "positions_A": pos_p,
            "velocities_A_fs": vel_p,
            "boxes_A": boxes,
            "masses": masses_p,
            "elements": list(wmi["phys_elements"]),
            "particle_metadata": meta,
            "species_meta": {
                "resolution": "all_atom",
                "molecule_ids": wmi["phys_mol_ids"].tolist(),
                "mol_species": species_per_mol,
                "mol_charges": [wmi["species_table"][rn]["charge"] for rn in wmi["mol_resnames"]],
                "species_table": wmi["species_table"],
                "total_mass_amu": wmi["total_mass_amu"],
            },
        }

    if resolution == "molecular":
        n_m = wmi["n_molecules"]
        pos_m = np.zeros((t_len, n_m, 3), dtype=np.float64)
        vel_m = None if vel_p is None else np.zeros((t_len, n_m, 3), dtype=np.float64)
        masses_m = np.zeros(n_m, dtype=np.float64)
        charges_m = np.zeros(n_m, dtype=np.float64)
        elems_m, fftypes_m, mol_species = [], [], []
        for m_id, m_idx in enumerate(mol_atoms):
            rn = wmi["mol_resnames"][m_id]
            sp = wmi["species_table"][rn]
            m_sub = masses_p[m_idx]
            w = m_sub / m_sub.sum()
            pos_m[:, m_id] = np.einsum("a,tad->td", w, pos_p[:, m_idx])
            if vel_m is not None:
                vel_m[:, m_id] = np.einsum("a,tad->td", w, vel_p[:, m_idx])
            masses_m[m_id] = m_sub.sum()
            charges_m[m_id] = sp["charge"]
            elems_m.append(sp["short_elem"])
            fftypes_m.append(sp["species"])
            mol_species.append(sp["species"])
        meta = {
            "version": 1,
            "units": {"charge": "e", "distance": "nm"},
            "charges": charges_m.tolist(),
            "ff_types": fftypes_m,
            "molecule_ids": list(range(n_m)),
            "original_indices": list(range(n_m)),
            "bonds": [],
            "constraints": [],
            "drude_parent_map": [],
        }
        return {
            "positions_A": pos_m,
            "velocities_A_fs": vel_m,
            "boxes_A": boxes,
            "masses": masses_m,
            "elements": elems_m,
            "particle_metadata": meta,
            "species_meta": {
                "resolution": "molecular",
                "molecule_ids": list(range(n_m)),
                "mol_species": mol_species,
                "mol_charges": charges_m.tolist(),
                "species_table": wmi["species_table"],
                "total_mass_amu": wmi["total_mass_amu"],
            },
        }

    # resolution == "polar_beads": split dipolar neutral solvents into [polar_head_O, scaffold_COM] + ions
    pos_b, vel_b = [], ([] if vel_p is not None else None)
    masses_b, charges_b, elems_b, fftypes_b, mol_ids_b, bonds_b, mol_species = [], [], [], [], [], [], []
    for m_id, m_idx in enumerate(mol_atoms):
        rn = wmi["mol_resnames"][m_id]
        sp = wmi["species_table"][rn]
        mol_species.append(sp["species"])
        q_sub = charges_p[m_idx]
        m_sub = masses_p[m_idx]
        if sp["role"] == "solvent" and len(m_idx) >= 4:
            head_loc = int(np.argmin(q_sub))  # carbonyl oxygen (most negative site)
            head_g = m_idx[head_loc]
            rest_g = [g for k_i, g in enumerate(m_idx) if k_i != head_loc]
            m_rest = masses_p[rest_g]
            w_rest = m_rest / m_rest.sum()
            idx_head = len(masses_b)
            idx_tail = idx_head + 1
            pos_b.append(pos_p[:, head_g])
            pos_b.append(np.einsum("a,tad->td", w_rest, pos_p[:, rest_g]))
            if vel_b is not None:
                vel_b.append(vel_p[:, head_g])
                vel_b.append(np.einsum("a,tad->td", w_rest, vel_p[:, rest_g]))
            q_h = float(charges_p[head_g])
            masses_b.extend([float(masses_p[head_g]), float(m_rest.sum())])
            charges_b.extend([q_h, -q_h])
            elems_b.extend(["O", sp["short_elem"]])
            fftypes_b.extend([f"{sp['species']}_O", f"{sp['species']}_R"])
            mol_ids_b.extend([m_id, m_id])
            bonds_b.append([idx_head, idx_tail])
        else:
            w = m_sub / m_sub.sum()
            pos_b.append(np.einsum("a,tad->td", w, pos_p[:, m_idx]))
            if vel_b is not None:
                vel_b.append(np.einsum("a,tad->td", w, vel_p[:, m_idx]))
            masses_b.append(float(m_sub.sum()))
            charges_b.append(float(sp["charge"]))
            elems_b.append(sp["short_elem"])
            fftypes_b.append(sp["species"])
            mol_ids_b.append(m_id)

    pos_out = np.stack(pos_b, axis=1)
    vel_out = None if vel_b is None else np.stack(vel_b, axis=1)
    meta = {
        "version": 1,
        "units": {"charge": "e", "distance": "nm"},
        "charges": charges_b,
        "ff_types": fftypes_b,
        "molecule_ids": mol_ids_b,
        "original_indices": list(range(len(masses_b))),
        "bonds": bonds_b,
        "constraints": [],
        "drude_parent_map": [],
    }
    return {
        "positions_A": pos_out,
        "velocities_A_fs": vel_out,
        "boxes_A": boxes,
        "masses": np.asarray(masses_b, dtype=np.float64),
        "elements": elems_b,
        "particle_metadata": meta,
        "species_meta": {
            "resolution": "polar_beads",
            "molecule_ids": mol_ids_b,
            "mol_species": mol_species,
            "mol_charges": [wmi["species_table"][rn]["charge"] for rn in wmi["mol_resnames"]],
            "species_table": wmi["species_table"],
            "total_mass_amu": wmi["total_mass_amu"],
        },
    }


def _read_openmm_box_log_A(log_path, n_frames):
    """Extract per-frame cubic box lengths from an OpenMM `EnhancedStateDataReporter` CSV log if present."""
    if log_path is None or not Path(log_path).is_file():
        return None
    lx_vals = []
    with Path(log_path).open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(line for line in fh if not line.startswith("##"))
        for row in reader:
            if "Box X (nm)" in row and "Box Y (nm)" in row and "Box Z (nm)" in row:
                lx_vals.append([
                    float(row["Box X (nm)"]) * 10.0,
                    float(row["Box Y (nm)"]) * 10.0,
                    float(row["Box Z (nm)"]) * 10.0,
                ])
    if not lx_vals:
        return None
    arr = np.asarray(lx_vals, dtype=np.float64)
    idx = np.linspace(0, len(arr) - 1, n_frames).round().astype(int)
    out = np.zeros((n_frames, 3, 3), dtype=np.float64)
    for k_d in range(3):
        out[:, k_d, k_d] = arr[idx, k_d]
    return out


def import_wmi_replica(build_dir, traj_path, out_dir, *,
                       traj_index=0, resolution="molecular", dt_frame_fs=200.0,
                       canonicalize_npt_box=True, state_log=None, unwrap=False,
                       velocities_A_fs=None, overwrite=True):
    """Import one `02_Na_MD` replica (`.dcd`, `.npz`, or `.h5`) into WignerFlow `<out_dir>/traj_XXX.h5`.

    Note: `02_Na_MD/md_explicit.py` writes DCD files with `enforcePeriodicBox=False` (already unwrapped),
    so `unwrap=False` is the safe default; pass `unwrap=True` if importing a wrapped trajectory.
    """
    wmi = load_wmi_build_dir(build_dir)
    src = Path(traj_path)
    suf = src.suffix.lower()
    boxes_A = None

    if suf == ".dcd":
        import mdtraj as md
        pdb_file = Path(build_dir) / "system" / "openmm" / "system.pdb"
        tr = md.load_dcd(str(src), top=str(pdb_file))
        pos_A = np.asarray(tr.xyz, dtype=np.float64) * 10.0
        if tr.unitcell_vectors is not None and np.all(np.isfinite(tr.unitcell_vectors)):
            boxes_A = np.asarray(tr.unitcell_vectors, dtype=np.float64) * 10.0
    elif suf == ".npz":
        with np.load(src, allow_pickle=False) as f:
            pos_A = np.asarray(f["positions_A" if "positions_A" in f.files else "positions"], dtype=np.float64)
            if "velocities_A_fs" in f.files and velocities_A_fs is None:
                velocities_A_fs = np.asarray(f["velocities_A_fs"], dtype=np.float64)
            if "boxes_A" in f.files:
                boxes_A = np.asarray(f["boxes_A"], dtype=np.float64)
    elif suf in (".h5", ".hdf5"):
        import h5py
        with h5py.File(src, "r") as f:
            pos_A = np.asarray(f["positions"][:], dtype=np.float64) * NM_TO_A
            if "velocities" in f and velocities_A_fs is None:
                velocities_A_fs = np.asarray(f["velocities"][:], dtype=np.float64) * NM_PER_PS_TO_A_PER_FS
            if "box" in f:
                boxes_A = np.asarray(f["box"][:], dtype=np.float64) * NM_TO_A
    else:
        raise ValueError(f"Unsupported trajectory extension {suf!r}; use .dcd, .npz, or .h5")

    if boxes_A is None or np.any(np.abs(np.linalg.det(boxes_A)) < 1e-8):
        log_boxes = _read_openmm_box_log_A(state_log, len(pos_A))
        boxes_A = log_boxes if log_boxes is not None else np.broadcast_to(
            np.eye(3) * wmi["equilibrium_box_L_A"], (len(pos_A), 3, 3)
        ).copy()

    if unwrap:
        pos_A = unwrap_positions(pos_A, boxes_A)

    red = reduce_wmi_frames(
        pos_A, velocities_A_fs, boxes_A, wmi,
        resolution=resolution, canonicalize_npt_box=canonicalize_npt_box,
    )
    x_nm = red["positions_A"] / NM_TO_A
    cells_nm = red["boxes_A"] / NM_TO_A
    estimated = red["velocities_A_fs"] is None
    if estimated:
        v_nm_ps = np.gradient(x_nm, dt_frame_fs / 1000.0, axis=0, edge_order=2 if len(x_nm) > 2 else 1)
    else:
        v_nm_ps = red["velocities_A_fs"] / NM_PER_PS_TO_A_PER_FS

    out_path_dir = Path(out_dir)
    out_path_dir.mkdir(parents=True, exist_ok=True)
    h5_out = out_path_dir / f"traj_{traj_index:03d}.h5"
    if h5_out.exists() and not overwrite:
        raise FileExistsError(h5_out)
    for field in ("pos", "vel", "meta"):
        (out_path_dir / "npy" / f"{h5_out.stem}_{field}.npy").unlink(missing_ok=True)

    attrs = {
        "complete": False,
        "dt_fs": float(dt_frame_fs),
        "save_every": 1,
        "periodic": True,
        "system": f"wmi_{resolution}",
        "resolution": resolution,
        "velocities_estimated": bool(estimated),
        "temperature_K": wmi["temperature_K"],
        "total_mass_amu": wmi["total_mass_amu"],
    }
    masses = red["masses"]
    with TrajWriter(
        h5_out, n_atoms=len(masses), masses=masses, elements=red["elements"],
        attrs=attrs, particle_metadata=red["particle_metadata"],
    ) as writer:
        for t in range(len(x_nm)):
            ke = 0.5 * float(np.sum(masses[:, None] * v_nm_ps[t] ** 2))
            writer.add(x_nm[t], v_nm_ps[t], np.full_like(x_nm[t], np.nan), np.nan, ke,
                       cells_nm[t], t * dt_frame_fs / 1000.0)
        writer.flush()
        writer.f.attrs["complete"] = True

    sp_json = out_path_dir / "species_meta.json"
    sp_json.write_text(json.dumps(red["species_meta"], indent=2), encoding="utf-8")
    return h5_out


def _molecule_coms_A(positions_A, masses, molecule_ids, n_molecules):
    """Collapse `(R, T, N_sites, 3)` unwrapped rollout coordinates to `(R, T, N_mol, 3)` molecule COMs."""
    pos = np.asarray(positions_A, dtype=np.float64)
    if pos.ndim == 3:
        pos = pos[None]
    r_cnt, t_cnt, n_sites, _ = pos.shape
    mol_ids = np.asarray(molecule_ids, dtype=np.int64)
    if n_sites == n_molecules and np.array_equal(mol_ids, np.arange(n_molecules)):
        return pos
    m = np.asarray(masses, dtype=np.float64)
    out = np.zeros((r_cnt, t_cnt, n_molecules, 3), dtype=np.float64)
    for m_id in range(n_molecules):
        idx = np.nonzero(mol_ids == m_id)[0]
        w = m[idx] / m[idx].sum()
        out[:, :, m_id] = np.einsum("a,rtad->rtd", w, pos[:, :, idx])
    return out


def evaluate_electrolyte_transport(positions_A, masses, box_A, dt_fs, species_meta, *,
                                   temperature_K=298.15, fit_fraction=(0.25, 0.90),
                                   cluster_cutoff_A=6.0, exp_sigma_mS_cm=None):
    """Compute `02_Na_MD/results_mu.json`-compatible transport & speciation metrics across `R` rollouts.

    positions_A: `(R, T, N, 3)` or `(T, N, 3)` unwrapped coordinates in Å.
    Returns density, per-species diffusivities (`1e-10 m^2/s`), Fickian exponent `alpha`,
    Nernst-Einstein `sigma_self`, full Einstein-Helfand / Onsager `sigma_total` (`± SE` across replicas),
    `ionicity = sigma_total / sigma_self`, `transference_plus_NE`, and `SSIP / CIP / AGG` fractions.
    """
    mol_species = list(species_meta["mol_species"])
    mol_charges = np.asarray(species_meta["mol_charges"], dtype=np.float64)
    n_mol = len(mol_species)
    coms = _molecule_coms_A(positions_A, masses, species_meta["molecule_ids"], n_mol)  # (R, T, M, 3)
    r_cnt, t_cnt, _, _ = coms.shape
    if t_cnt < 4:
        raise ValueError("Need at least 4 frames to fit electrolyte transport slopes")

    box_mat = np.asarray(box_A, dtype=np.float64)
    if box_mat.ndim == 3:
        box_mat = box_mat[0]
    vol_A3 = float(abs(np.linalg.det(box_mat)))
    edge_A = float(vol_A3 ** (1.0 / 3.0))
    total_mass = float(species_meta.get("total_mass_amu", np.sum(masses)))
    density_g_cm3 = (total_mass * AMU_TO_GRAM) / (vol_A3 * 1e-24)

    max_lag = max(2, t_cnt // 2)
    lags = np.unique(np.linspace(1, max_lag, min(max_lag, 80)).round().astype(int))
    t_ps = lags * (float(dt_fs) / 1000.0)
    f_lo, f_hi = fit_fraction
    fit_mask = (t_ps >= f_lo * t_ps[-1]) & (t_ps <= f_hi * t_ps[-1])
    if int(fit_mask.sum()) < 2:
        fit_mask = np.ones_like(t_ps, dtype=bool)

    # 1. Per-species MSD & self-diffusion D_s (slope in Å^2/ps -> D = slope / 6; 1 Å^2/ps = 10 * 1e-10 m^2/s)
    unique_sp = []
    for sp in mol_species:
        if sp not in unique_sp:
            unique_sp.append(sp)

    diffusion_data = []
    d_by_sp_A2_ps = {}
    alpha_by_sp = {}
    for sp in unique_sp:
        sp_idx = np.nonzero([s == sp for s in mol_species])[0]
        msd_rep = np.zeros((r_cnt, len(lags)), dtype=np.float64)
        for k_l, lag in enumerate(lags):
            disp = coms[:, lag:, sp_idx] - coms[:, :-lag, sp_idx]
            msd_rep[:, k_l] = np.sum(disp * disp, axis=-1).mean(axis=(1, 2))
        msd_mean = msd_rep.mean(axis=0)
        slopes = [float(np.polyfit(t_ps[fit_mask], msd_rep[r_i, fit_mask], 1)[0]) for r_i in range(r_cnt)]
        d_rep = np.maximum(np.asarray(slopes) / 6.0, 0.0)  # Å^2/ps
        d_mean = float(d_rep.mean())
        d_se = float(d_rep.std(ddof=1) / np.sqrt(r_cnt)) if r_cnt > 1 else 0.0
        d_by_sp_A2_ps[sp] = (d_mean, d_se)

        pos_mask = fit_mask & (msd_mean > 1e-12) & (t_ps > 0)
        alpha = float(np.polyfit(np.log(t_ps[pos_mask]), np.log(msd_mean[pos_mask]), 1)[0]) if pos_mask.sum() >= 2 else 1.0
        alpha_by_sp[sp] = alpha
        diffusion_data.append({
            "species": sp,
            "diffusion_coefficient_1e-10_m2_s": d_mean * 10.0,
            "std_err_1e-10_m2_s": d_se * 10.0,
            "diffusion_cm2_s": d_mean * 1e-4,
            "alpha": alpha,
        })

    # 2. Nernst-Einstein self-conductivity sigma_self and Einstein-Helfand / Onsager sigma_total
    ion_mask = np.abs(mol_charges) > 0.1
    ion_idx = np.nonzero(ion_mask)[0]
    q_ions = mol_charges[ion_idx]
    sum_nq2 = float(np.sum(q_ions ** 2))

    # Per-replica sigma_self and collective charge-displacement slope < |sum_i q_i dR_i(tau)|^2 >
    sigma_self_rep = np.zeros(r_cnt, dtype=np.float64)
    sigma_tot_rep = np.zeros(r_cnt, dtype=np.float64)
    for r_i in range(r_cnt):
        coms_r = coms[r_i]  # (T, M, 3)
        ne_sum = 0.0
        for sp in unique_sp:
            sp_idx = np.nonzero([s == sp for s in mol_species])[0]
            q_sp = mol_charges[sp_idx[0]]
            if abs(q_sp) > 0.1:
                msd_r = np.array([
                    np.sum((coms_r[lag:, sp_idx] - coms_r[:-lag, sp_idx]) ** 2, axis=-1).mean()
                    for lag in lags
                ])
                d_r = max(float(np.polyfit(t_ps[fit_mask], msd_r[fit_mask], 1)[0]) / 6.0, 0.0)
                ne_sum += len(sp_idx) * (q_sp ** 2) * d_r
        sigma_self_rep[r_i] = COND_PREFACTOR_MS_CM * ne_sum / (vol_A3 * temperature_K)

        # Collective dipole displacement M(t) = sum_{i in ions} q_i R_i(t)
        m_traj = np.einsum("i,tid->td", q_ions, coms_r[:, ion_idx])  # (T, 3)
        coll_msd = np.array([
            np.sum((m_traj[lag:] - m_traj[:-lag]) ** 2, axis=-1).mean()
            for lag in lags
        ])
        slope_coll = float(np.polyfit(t_ps[fit_mask], coll_msd[fit_mask], 1)[0])  # Å^2 * e^2 / ps
        d_coll = max(slope_coll / 6.0, 0.0)
        sigma_tot_rep[r_i] = COND_PREFACTOR_MS_CM * d_coll / (vol_A3 * temperature_K)

    sigma_self = float(sigma_self_rep.mean())
    sigma_self_se = float(sigma_self_rep.std(ddof=1) / np.sqrt(r_cnt)) if r_cnt > 1 else 0.0
    sigma_total = float(sigma_tot_rep.mean())
    sigma_total_se = float(sigma_tot_rep.std(ddof=1) / np.sqrt(r_cnt)) if r_cnt > 1 else 0.0
    ionicity = float(sigma_total / sigma_self) if sigma_self > 1e-12 else 0.0

    d_cat = d_by_sp_A2_ps.get("cation", (0.0, 0.0))[0]
    d_ani = d_by_sp_A2_ps.get("anion", (0.0, 0.0))[0]
    t_plus_ne = float(d_cat / (d_cat + d_ani)) if (d_cat + d_ani) > 1e-12 else 0.0

    # 3. Ion speciation (SSIP / CIP / AGG) at `cluster_cutoff_A` (6.0 Å matching 02_Na_MD/results_mu.json)
    cat_idx = np.nonzero(mol_charges > 0.5)[0]
    ani_idx = np.nonzero(mol_charges < -0.5)[0]
    ssip_cnt = cip_cnt = agg_cnt = 0
    total_cat_samples = 0
    sub_frames = np.linspace(t_cnt // 2, t_cnt - 1, min(25, max(1, t_cnt - t_cnt // 2))).round().astype(int)
    if len(cat_idx) > 0 and len(ani_idx) > 0:
        for r_i in range(r_cnt):
            for f_i in sub_frames:
                rc = coms[r_i, f_i, cat_idx]
                ra = coms[r_i, f_i, ani_idx]
                dr = minimum_image_np(rc[:, None, :] - ra[None, :, :], box_mat)
                adj_ca = np.linalg.norm(dr, axis=-1) < cluster_cutoff_A  # (N_cat, N_ani)
                # Connected components on bipartite cation-anion graph
                n_c, n_a = len(cat_idx), len(ani_idx)
                visited_c = np.zeros(n_c, dtype=bool)
                visited_a = np.zeros(n_a, dtype=bool)
                for c0 in range(n_c):
                    if visited_c[c0]:
                        continue
                    comp_c, comp_a = [c0], []
                    visited_c[c0] = True
                    queue = [("c", c0)]
                    while queue:
                        kind, u = queue.pop()
                        if kind == "c":
                            for v_a in np.nonzero(adj_ca[u])[0]:
                                if not visited_a[v_a]:
                                    visited_a[v_a] = True
                                    comp_a.append(int(v_a))
                                    queue.append(("a", int(v_a)))
                        else:
                            for v_c in np.nonzero(adj_ca[:, u])[0]:
                                if not visited_c[v_c]:
                                    visited_c[v_c] = True
                                    comp_c.append(int(v_c))
                                    queue.append(("c", int(v_c)))
                    n_ions_cluster = len(comp_c) + len(comp_a)
                    if len(comp_a) == 0:
                        ssip_cnt += len(comp_c)
                    elif len(comp_c) == 1 and len(comp_a) == 1:
                        cip_cnt += 1
                    else:
                        agg_cnt += len(comp_c)
                    total_cat_samples += len(comp_c)

    denom = max(total_cat_samples, 1)
    f_ssip, f_cip, f_agg = ssip_cnt / denom, cip_cnt / denom, agg_cnt / denom

    out = {
        "system_properties": {
            "density [g/cm3]": density_g_cm3,
            "conductivity_self [mS/cm]": sigma_self,
            "conductivity_self_se [mS/cm]": sigma_self_se,
            "conductivity_total [mS/cm]": sigma_total,
            "conductivity_total_se [mS/cm]": sigma_total_se,
            "ionicity": ionicity,
            "transference_plus_NE": t_plus_ne,
            "alpha_cation": alpha_by_sp.get("cation", 1.0),
        },
        "diffusion_data": diffusion_data,
        "cluster": [
            {"size": 0, "category": "SSIP", "fraction": f_ssip},
            {"size": 1, "category": "CIP", "fraction": f_cip},
            {"size": 2, "category": "AGG", "fraction": f_agg},
        ],
        "_cutoff": float(cluster_cutoff_A),
        "n_rollouts": r_cnt,
        "n_frames": t_cnt,
        "sum_Nq2": sum_nq2,
        "cubic_box_length [angstrom]": edge_A,
        "components": unique_sp,
    }
    if exp_sigma_mS_cm is not None and exp_sigma_mS_cm > 0:
        out["vs_experiment"] = {
            "exp_sigma_mS_cm": float(exp_sigma_mS_cm),
            "rel_err_total_pct": 100.0 * (sigma_total - exp_sigma_mS_cm) / exp_sigma_mS_cm,
            "rel_err_self_pct": 100.0 * (sigma_self - exp_sigma_mS_cm) / exp_sigma_mS_cm,
        }
    return out


def transport_calibration_loss(model, pos, vel, types, box, masses, particle_charges, *,
                               target_sigma_mS_cm, temperature_K=298.15, unroll_steps=4,
                               sampling_steps=2, teacher_model=None, anchor_weight=1.0):
    """Differentiable post-training objective matching collective Einstein-Helfand ion transport to experiment.

    Unrolls `model.predict_with_grad` for `unroll_steps` steps from `(pos, vel)` `(B, k, N, 3)`, computes the
    batch-averaged collective charge displacement `< |sum_i q_i dX_i(unroll)|^2 > / (6 * V * T * dt_total)`,
    penalizes relative log-error against `target_sigma_mS_cm`, and anchors single-step phase-space updates to
    `teacher_model` (pre-trained on `MU-PFF`) so local solvation structure and density do not drift.
    """
    b, k, _, _ = pos.shape
    ctx_x, ctx_v = pos[:, :k].clone(), vel[:, :k].clone()
    x_start = ctx_x[:, -1]
    q = torch.as_tensor(particle_charges, device=pos.device, dtype=pos.dtype)

    anchor_loss = pos.new_tensor(0.0)
    for step_i in range(unroll_steps):
        noise = (model.com_free_noise(b, types, masses, like=ctx_v[:, -1])
                 if getattr(model, "head_type", None) == "flow" else None)
        dx, v_new = model.predict_with_grad(ctx_x, ctx_v, types, box, masses,
                                            n_steps=sampling_steps, noise=noise)
        if step_i == 0 and teacher_model is not None and anchor_weight > 0:
            with torch.no_grad():
                dx_ref, v_ref = teacher_model.predict(ctx_x, ctx_v, types, box, masses,
                                                      n_steps=sampling_steps, noise=noise)
            anchor_loss = ((dx - dx_ref).pow(2).mean() + (model.stride_fs ** 2) * (v_new - v_ref).pow(2).mean())
        ctx_x = torch.cat([ctx_x[:, 1:], (ctx_x[:, -1] + dx)[:, None]], dim=1)
        ctx_v = torch.cat([ctx_v[:, 1:], v_new[:, None]], dim=1)

    total_disp = ctx_x[:, -1] - x_start  # (B, N, 3) in Å
    m_disp = torch.einsum("n,bnd->bd", q, total_disp)  # (B, 3) in e * Å
    dt_total_ps = (unroll_steps * float(model.stride_fs)) / 1000.0
    if box.ndim == 2:
        vol_A3 = torch.abs(torch.linalg.det(box.double())).to(m_disp.dtype).expand(b)
    else:
        vol_A3 = torch.abs(torch.linalg.det(box.double())).to(m_disp.dtype)
    d_coll = (m_disp.pow(2).sum(dim=-1) / (6.0 * max(dt_total_ps, 1e-6)))  # (B,) Å^2 * e^2 / ps
    sigma_est = (COND_PREFACTOR_MS_CM * d_coll / (vol_A3 * float(temperature_K))).mean()
    target = sigma_est.new_tensor(float(target_sigma_mS_cm))
    sigma_loss = (torch.log((sigma_est + 0.1) / (target + 0.1))).pow(2)
    return sigma_loss + float(anchor_weight) * anchor_loss
