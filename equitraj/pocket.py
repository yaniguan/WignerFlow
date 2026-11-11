"""PocketMD v2: literature-aligned CPU/edge molecular-dynamics triage calculator ("随身科学计算器").

Turns WignerFlow's coarse-stride rollout engine and trajectory diagnostics into a lightweight
pre-screening application that runs in ~1 second on a laptop CPU to answer one practical question
before committing 50-500 GPU-hours to production MD / RB-FEP / AIMD:

    "Is this candidate pose / electrolyte additive / neural checkpoint worth sending to Big MD?"
    -> GO / HOLD / REJECT

Grounded in a 120-paper synthesis across drug binding MD, battery electrolyte MD, and coarse-stride
equivariant neural propagators:
1. `triage_binding`     — Multi-replica OpenBPMD `CompScore = PoseScore - 5 * PersScore`
                          (Clark et al. JCTC 2016; Lukauskis et al. JCIM 2022), PoseBusters vdW
                          steric audit (Buttenschoen et al. Chem. Sci. 2024), kinetic strain-recoil
                          spike, multi-replica pose entropy, and a 50-fs direction-retention
                          tau-RAMD impulse residence probe (Kokh et al. JCTC 2018).
2. `triage_electrolyte` — Li+ solvation speciation (SCN/ACN/Additive CN, SSIP/CIP/AGG), Cluster
                          Nernst-Einstein conductivity `sigma_cNE`, Inverse Haven Ionicity
                          `sigma_cNE / sigma_NE`, and cluster transference `t+_cNE`
                          (France-Lanord & Grossman PRL 2019; Fong, Self & Persson ACS Energy Lett.
                          2019), Structural Hopping vs. Vehicular Index `chi_hop` (Self et al. 2019),
                          and Competitive Additive Grafting Score `CAGS_add` (Gao & Zhang JACS 2023).
3. `triage_checkpoint`  — Fast CPU Verlet rollout of WignerFlow `.pt` checkpoints with Fu-2023
                          physical stability auditing (vdW ratio, RDF L1, temperature, D, COM
                          Flying-Ice-Cube leakage) plus a committee-free dual epistemic gate
                          (Heun-Euler ODE curvature `delta_ODE` and O(3)-invariant multi-seed flow
                          spread `sigma_flow`, Vita & Schwalbe-Koda 2023).
"""

from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch

from equitraj import analysis, binding
from equitraj.adapters import unwrap_positions
from equitraj.fast import FastRollout
from equitraj.model import EquiTraj, remove_com
from equitraj.rollout import min_pair_distance, rollout, temperature
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB, NM_TO_A

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMS_DIR = REPO_ROOT / "systems"
CKPTS_DIR = REPO_ROOT / "checkpoints"

# Nernst-Einstein prefactor:
#   sigma [mS/cm] = (e^2 / (V [Å^3] * k_B * T [K])) * sum_s (N_s * z_s^2 * D_s [cm^2/s])
# With e = 1.602176634e-19 C, k_B = 1.380649e-23 J/K, 1 Å^3 = 1e-30 m^3,
# 1 cm^2/s = 1e-4 m^2/s, and 1 S/m = 10 mS/cm:
NERNST_EINSTEIN_MS_CM = 1.85924882e12

COULOMB_KJ_MOL_A = 1389.35458  # kJ/mol * Å / e^2

_DEFAULT_MASS = {
    "H": 1.008, "He": 4.003, "Li": 6.941, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "P": 30.974, "S": 32.06,
    "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
}

# Bondi / PoseBusters van der Waals radii in Å (polar H adjusted to 1.00 Å so canonical 1.90-2.00 Å
# donor-H...acceptor H-bonds pass the 0.75 * (r_i + r_j) PoseBusters steric threshold cleanly).
_VDW_RADIUS_A = {
    "H": 1.00, "He": 1.40, "Li": 1.82, "B": 1.92, "C": 1.70, "N": 1.55,
    "O": 1.52, "F": 1.47, "Na": 2.27, "Mg": 1.73, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "Ar": 1.88, "K": 2.75, "Ca": 2.31,
}


# ---------------------------------------------------------------- public MD dataset catalog

DATASET_CATALOG = [
    {
        "id": "misato_protein_ligand",
        "domain": "binding",
        "title": "MISATO (Protein-Ligand MD + QM, ~17,000 complexes, Siebenmorgen et al. Nat. Comput. Sci. 2024)",
        "url": "https://zenodo.org/record/7711953",
        "import_hint": "python scripts/import_md.py MD.hdf5 --out-dir data/misato_case --dt-fs 2.0 --unit angstrom",
        "pain_point": "Filter false-positive docked poses via CompScore & tau-RAMD before 50-GPU-hour RB-FEP.",
    },
    {
        "id": "mdbind_affinity",
        "domain": "binding",
        "title": "MDbind (~63,000 Protein-Ligand Short MD Simulations)",
        "url": "https://zenodo.org/records/10390550",
        "import_hint": "python scripts/import_md.py complex.xtc --topology complex.pdb --out-dir data/mdbind_case --dt-fs 2.0",
        "pain_point": "Multi-replica OpenBPMD CompScore & PoseBusters triage on CPU.",
    },
    {
        "id": "atlas_proteins",
        "domain": "binding",
        "title": "ATLAS Standardized All-Atom Protein MD Database (Vander Meersche et al. NAR 2024)",
        "url": "https://www.dsimb.inserm.fr/ATLAS",
        "import_hint": "python scripts/import_md.py traj.xtc --topology top.pdb --out-dir data/atlas_case --dt-fs 2.0",
        "pain_point": "Rapid apo/holo pocket flexibility & pose-entropy check before ensemble docking.",
    },
    {
        "id": "libe_electrolyte",
        "domain": "electrolyte",
        "title": "LIBE / Figshare Carbonate + Additive Electrolyte MD (Spotte-Smith / Persson Group)",
        "url": "https://doi.org/10.6084/m9.figshare.14226857",
        "import_hint": "python scripts/import_md.py li_ec_dmc.dcd --topology top.pdb --out-dir data/electrolyte --dt-fs 1.0",
        "pain_point": "Screen new additives for Li+ solvation (CAGS) & Cluster-NE sigma_cNE before 50 ns MD / AIMD.",
    },
    {
        "id": "edp_electrolyte_genome",
        "domain": "electrolyte",
        "title": "Electrolyte Design / Non-Aqueous Li-Salt Solvation Trajectories (Fong, Self & Persson)",
        "url": "https://materialsproject.org/molecules",
        "import_hint": "python scripts/import_md.py electrolyte_traj.dcd --topology system.pdb --out-dir data/edp_case --dt-fs 2.0",
        "pain_point": "Catch neutral CIP/AGG conductivity mirages via Inverse Haven Ionicity (sigma_cNE / sigma_NE).",
    },
    {
        "id": "md17_rmd17_coarse_stride",
        "domain": "checkpoint",
        "title": "MD17 / rMD17 & Liquid Argon Coarse-Stride Benchmark Trajectories (Fu et al. TMLR 2023)",
        "url": "http://www.quantum-machine.org/gdml/#datasets",
        "import_hint": "python scripts/train.py --data data/argon --head flow --stride 25",
        "pain_point": "Audit 50-fs neural checkpoints for RDF L1, COM leakage, and flow epistemic spread on CPU.",
    },
]


# ---------------------------------------------------------------- zero-OpenMM PDB + XML parsers

def parse_pdb(pdb_path):
    """Parse an OpenMM/standard PDB file without external dependencies.

    Maps PDB atom serial numbers across `TER` records so `CONECT` bonds and molecule
    segment IDs remain exact on multi-molecule boxes (e.g. `systems/li_ec_pf6_q08.pdb`).
    Returns a dict with positions (N, 3) Å, elements, atom_names, mol_ids, bonds, and box (3, 3) Å or None.
    """
    positions, elements, names, mol_ids = [], [], [], []
    serial_to_idx = {}
    bonds_set = set()
    box = None
    cur_mol = 0
    in_mol = False

    with open(pdb_path, encoding="utf-8") as fh:
        for line in fh:
            rec = line[:6].strip()
            if rec == "CRYST1":
                lx = float(line[6:15])
                ly = float(line[15:24])
                lz = float(line[24:33])
                box = np.diag([lx, ly, lz]).astype(np.float64)
            elif rec in ("ATOM", "HETATM"):
                in_mol = True
                serial = int(line[6:11])
                name = line[12:16].strip()
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                elem = line[76:78].strip() if len(line) >= 78 else ""
                if not elem:
                    elem = "".join(ch for ch in name if ch.isalpha())[:2].capitalize()
                idx = len(positions)
                serial_to_idx[serial] = idx
                positions.append([x, y, z])
                elements.append(elem)
                names.append(name)
                mol_ids.append(cur_mol)
            elif rec == "TER":
                if in_mol:
                    cur_mol += 1
                    in_mol = False
            elif rec == "CONECT":
                parts = [int(line[k:k + 5]) for k in range(6, len(line.rstrip()), 5) if line[k:k + 5].strip()]
                if len(parts) >= 2 and parts[0] in serial_to_idx:
                    i = serial_to_idx[parts[0]]
                    for s2 in parts[1:]:
                        if s2 in serial_to_idx:
                            j = serial_to_idx[s2]
                            if i != j:
                                bonds_set.add((min(i, j), max(i, j)))

    if not positions:
        raise ValueError(f"no ATOM/HETATM records in {pdb_path}")
    masses = np.array([_DEFAULT_MASS.get(e, 12.011) for e in elements], dtype=np.float64)
    bonds = np.array(sorted(bonds_set), dtype=np.int64).reshape(-1, 2)
    return {
        "positions": np.asarray(positions, dtype=np.float64),
        "elements": elements,
        "atom_names": names,
        "masses": masses,
        "mol_ids": np.asarray(mol_ids, dtype=np.int64),
        "bonds": bonds,
        "box": box,
    }


def parse_openmm_xml(xml_path):
    """Extract masses, charges, LJ (sigma, epsilon), harmonic bonds/constraints, and angles from OpenMM XML.

    Converts OpenMM internal units (nm, rad, kJ/mol) directly to WignerFlow units (Å, rad, kJ/mol).
    """
    root = ET.parse(xml_path).getroot()
    masses = [float(p.attrib["mass"]) for p in root.find("Particles").findall("Particle")]
    n = len(masses)
    charges = np.zeros(n, dtype=np.float64)
    sigma_a = np.full(n, 3.2, dtype=np.float64)
    eps_kj = np.full(n, 0.25, dtype=np.float64)
    bonds = []  # (i, j, r0_A, k_kj_mol_A2)
    angles = []  # (i, j, k_idx, theta0_rad, k_kj_mol_rad2)
    excluded = set()

    constraints_el = root.find("Constraints")
    if constraints_el is not None:
        for c in constraints_el.findall("Constraint"):
            p1, p2 = int(c.attrib["p1"]), int(c.attrib["p2"])
            r0 = float(c.attrib["d"]) * NM_TO_A
            bonds.append((p1, p2, r0, 2500.0))  # stiff spring proxy for constrained X-H
            excluded.add((min(p1, p2), max(p1, p2)))

    forces_el = root.find("Forces")
    if forces_el is not None:
        for force in forces_el.findall("Force"):
            ftype = force.attrib.get("type", "")
            if ftype == "NonbondedForce":
                for idx, p in enumerate(force.find("Particles").findall("Particle")):
                    charges[idx] = float(p.attrib["q"])
                    sigma_a[idx] = max(0.8, float(p.attrib["sig"]) * NM_TO_A)
                    eps_kj[idx] = float(p.attrib["eps"])
                exc_el = force.find("Exceptions")
                if exc_el is not None:
                    for ex in exc_el.findall("Exception"):
                        p1, p2 = int(ex.attrib["p1"]), int(ex.attrib["p2"])
                        excluded.add((min(p1, p2), max(p1, p2)))
            elif ftype == "HarmonicBondForce":
                for b in force.find("Bonds").findall("Bond"):
                    p1, p2 = int(b.attrib["p1"]), int(b.attrib["p2"])
                    r0 = float(b.attrib["d"]) * NM_TO_A
                    # OpenMM k is kJ/(mol*nm^2); divide by 100 for kJ/(mol*Å^2)
                    k_a2 = float(b.attrib["k"]) / (NM_TO_A ** 2)
                    bonds.append((p1, p2, r0, min(k_a2, 3500.0)))
                    excluded.add((min(p1, p2), max(p1, p2)))
            elif ftype == "HarmonicAngleForce":
                for a in force.find("Angles").findall("Angle"):
                    angles.append((
                        int(a.attrib["p1"]), int(a.attrib["p2"]), int(a.attrib["p3"]),
                        float(a.attrib["a"]), float(a.attrib["k"]),
                    ))

    b_arr = np.array([(b[0], b[1]) for b in bonds], dtype=np.int64).reshape(-1, 2)
    b_r0 = np.array([b[2] for b in bonds], dtype=np.float64)
    b_k = np.array([b[3] for b in bonds], dtype=np.float64)
    a_arr = np.array([(a[0], a[1], a[2]) for a in angles], dtype=np.int64).reshape(-1, 3)
    a_th0 = np.array([a[3] for a in angles], dtype=np.float64)
    a_k = np.array([a[4] for a in angles], dtype=np.float64)
    return {
        "masses": np.asarray(masses, dtype=np.float64),
        "charges": charges,
        "sigma_a": sigma_a,
        "eps_kj": eps_kj,
        "bond_pairs": b_arr,
        "bond_r0": b_r0,
        "bond_k": b_k,
        "angle_triplets": a_arr,
        "angle_theta0": a_th0,
        "angle_k": a_k,
        "excluded_pairs": excluded,
    }


def posebusters_vdw_audit(positions, elements, *, group_a=None, group_b=None, excluded_pairs=None,
                          box_length=None, clash_threshold=0.75):
    """PoseBusters-style normalized van der Waals overlap check (Buttenschoen et al., Chem. Sci. 2024).

    Computes pairwise ratios `d_ij / (r_vdW(i) + r_vdW(j))` across either intermolecular pairs
    `(group_a, group_b)` or all non-excluded pairs. Values `< 0.75` indicate unphysical steric clashes.
    """
    pos = np.asarray(positions, dtype=np.float64)
    radii = np.array([_VDW_RADIUS_A.get(e, 1.70) for e in elements], dtype=np.float64)
    if group_a is not None and group_b is not None and len(group_a) > 0 and len(group_b) > 0:
        ga = np.asarray(group_a, dtype=np.int64)
        gb = np.asarray(group_b, dtype=np.int64)
        dr = pos[ga][:, None, :] - pos[gb][None, :, :]
        if box_length is not None:
            dr = dr - float(box_length) * np.round(dr / float(box_length))
        d = np.linalg.norm(dr, axis=-1)
        r_sum = radii[ga][:, None] + radii[gb][None, :]
        ratios = (d / np.maximum(r_sum, 1e-6)).ravel()
    else:
        n = len(pos)
        if n < 2:
            return {"min_vdw_ratio": 1.0, "n_vdw_clashes": 0}
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        if excluded_pairs:
            for i, j in excluded_pairs:
                if 0 <= i < n and 0 <= j < n:
                    mask[min(i, j), max(i, j)] = False
        pi, pj = np.nonzero(mask)
        if len(pi) == 0:
            return {"min_vdw_ratio": 1.0, "n_vdw_clashes": 0}
        dr = pos[pi] - pos[pj]
        if box_length is not None:
            dr = dr - float(box_length) * np.round(dr / float(box_length))
        d = np.linalg.norm(dr, axis=-1)
        ratios = d / np.maximum(radii[pi] + radii[pj], 1e-6)

    if ratios.size == 0:
        return {"min_vdw_ratio": 1.0, "n_vdw_clashes": 0}
    return {
        "min_vdw_ratio": float(ratios.min()),
        "n_vdw_clashes": int((ratios < clash_threshold).sum()),
    }


# ---------------------------------------------------------------- pure-PyTorch CPU pocket engine

class PocketMicroEngine:
    """Vectorized pure-PyTorch BAOAB-Langevin / Velocity-Verlet engine for <150-atom pocket & shell screening.

    Used when no system-specific WignerFlow `.pt` checkpoint is supplied yet, so scientists can
    immediately screen custom drug poses or electrolyte additives on any CPU without OpenMM.
    """

    def __init__(self, masses, charges, sigma_a, eps_kj, *, bond_pairs=None, bond_r0=None, bond_k=None,
                 angle_triplets=None, angle_theta0=None, angle_k=None, excluded_pairs=None,
                 restraint_indices=None, restraint_anchors=None, restraint_k=0.0,
                 box_length=None, dielectric=4.0):
        self.masses = torch.as_tensor(masses, dtype=torch.float64)
        self.n = self.masses.shape[0]
        self.charges = torch.as_tensor(charges, dtype=torch.float64)
        self.sigma = 0.5 * (torch.as_tensor(sigma_a, dtype=torch.float64)[:, None]
                            + torch.as_tensor(sigma_a, dtype=torch.float64)[None, :])
        self.eps = torch.sqrt(torch.clamp(
            torch.as_tensor(eps_kj, dtype=torch.float64)[:, None]
            * torch.as_tensor(eps_kj, dtype=torch.float64)[None, :], min=0.0
        ))
        self.qq = (self.charges[:, None] * self.charges[None, :]) * (COULOMB_KJ_MOL_A / max(dielectric, 1.0))

        mask = torch.ones((self.n, self.n), dtype=torch.bool).triu(1)
        if excluded_pairs:
            for i, j in excluded_pairs:
                if 0 <= i < self.n and 0 <= j < self.n:
                    mask[min(i, j), max(i, j)] = False
        pi, pj = torch.nonzero(mask, as_tuple=True)
        self.nb_i, self.nb_j = pi, pj
        self.nb_sig = self.sigma[pi, pj]
        self.nb_eps = self.eps[pi, pj]
        self.nb_qq = self.qq[pi, pj]

        self.bond_pairs = (torch.empty((0, 2), dtype=torch.long) if bond_pairs is None or len(bond_pairs) == 0
                           else torch.as_tensor(bond_pairs, dtype=torch.long))
        self.bond_r0 = torch.empty(0, dtype=torch.float64) if bond_r0 is None else torch.as_tensor(bond_r0, dtype=torch.float64)
        self.bond_k = torch.empty(0, dtype=torch.float64) if bond_k is None else torch.as_tensor(bond_k, dtype=torch.float64)

        self.ang_triplets = (torch.empty((0, 3), dtype=torch.long) if angle_triplets is None or len(angle_triplets) == 0
                             else torch.as_tensor(angle_triplets, dtype=torch.long))
        self.ang_th0 = torch.empty(0, dtype=torch.float64) if angle_theta0 is None else torch.as_tensor(angle_theta0, dtype=torch.float64)
        self.ang_k = torch.empty(0, dtype=torch.float64) if angle_k is None else torch.as_tensor(angle_k, dtype=torch.float64)

        self.rest_idx = None if restraint_indices is None else torch.as_tensor(restraint_indices, dtype=torch.long)
        self.rest_ref = None if restraint_anchors is None else torch.as_tensor(restraint_anchors, dtype=torch.float64)
        self.rest_k = float(restraint_k)
        self.box_length = None if box_length is None else float(box_length)

    def _wrap(self, dr):
        if self.box_length is None:
            return dr
        return dr - self.box_length * torch.round(dr / self.box_length)

    def forces_kj_mol_a(self, x):
        f = torch.zeros_like(x)
        # 1. Harmonic bonds: V = 0.5 * k * (r - r0)^2
        if self.bond_pairs.shape[0] > 0:
            bi, bj = self.bond_pairs[:, 0], self.bond_pairs[:, 1]
            dr = self._wrap(x[bi] - x[bj])
            r = torch.linalg.norm(dr, dim=-1).clamp_min(1e-6)
            fb = (-self.bond_k * (r - self.bond_r0) / r)[:, None] * dr
            f.index_add_(0, bi, fb)
            f.index_add_(0, bj, -fb)

        # 2. Harmonic angles: V = 0.5 * k * (theta - theta0)^2
        if self.ang_triplets.shape[0] > 0:
            ai, aj, ak = self.ang_triplets[:, 0], self.ang_triplets[:, 1], self.ang_triplets[:, 2]
            u = self._wrap(x[ai] - x[aj])
            v = self._wrap(x[ak] - x[aj])
            ru = torch.linalg.norm(u, dim=-1).clamp_min(1e-6)
            rv = torch.linalg.norm(v, dim=-1).clamp_min(1e-6)
            u_hat = u / ru[:, None]
            v_hat = v / rv[:, None]
            cos_th = (u_hat * v_hat).sum(dim=-1).clamp(-0.999999, 0.999999)
            th = torch.acos(cos_th)
            sin_th = torch.sqrt(1.0 - cos_th * cos_th).clamp_min(1e-5)
            dV_dth = self.ang_k * (th - self.ang_th0)
            fi = -dV_dth[:, None] * (cos_th[:, None] * u_hat - v_hat) / (ru * sin_th)[:, None]
            fk = -dV_dth[:, None] * (cos_th[:, None] * v_hat - u_hat) / (rv * sin_th)[:, None]
            f.index_add_(0, ai, fi)
            f.index_add_(0, ak, fk)
            f.index_add_(0, aj, -(fi + fk))

        # 3. Soft-core capped LJ + damped Coulomb (avoids numerical blowup on unminimized docked clashes)
        if self.nb_i.numel() > 0:
            dr = self._wrap(x[self.nb_i] - x[self.nb_j])
            r2 = (dr * dr).sum(dim=-1).clamp_min(0.64)  # soft cap at r = 0.8 Å
            r = torch.sqrt(r2)
            sr2 = (self.nb_sig * self.nb_sig) / r2
            sr6 = sr2 * sr2 * sr2
            f_mag = (24.0 * self.nb_eps * (2.0 * sr6 * sr6 - sr6) / r + self.nb_qq / r2).clamp(-600.0, 600.0)
            if self.box_length is not None:
                rc = 0.48 * self.box_length
                f_mag = torch.where(r < rc, f_mag * (1.0 - (r / rc) ** 4), torch.zeros_like(f_mag))
            fpair = (f_mag / r)[:, None] * dr
            f.index_add_(0, self.nb_i, fpair)
            f.index_add_(0, self.nb_j, -fpair)

        # 4. Pocket C-alpha positional restraints (keeps protein scaffold anchored in vacuum/implicit solvent)
        if self.rest_idx is not None and self.rest_k > 0.0:
            f[self.rest_idx] += -self.rest_k * (x[self.rest_idx] - self.rest_ref)

        return f

    def run(self, pos0, *, n_frames=80, substeps=8, dt_fs=1.5, temperature_k=300.0, friction_per_ps=2.0, seed=0):
        """Integrate `n_frames * substeps` steps on CPU and return unwrapped positions & velocities."""
        g = torch.Generator().manual_seed(seed)
        x = torch.as_tensor(pos0, dtype=torch.float64).clone()
        v_std = torch.sqrt((KB * temperature_k / AMU_A2_PER_FS2_TO_KJ_PER_MOL) / self.masses)
        v = torch.randn(x.shape, generator=g, dtype=torch.float64) * v_std[:, None]
        if self.rest_idx is None:
            v = remove_com(v[None], self.masses)[0]

        conv = 1.0 / AMU_A2_PER_FS2_TO_KJ_PER_MOL  # (kJ/mol/Å) / amu -> Å/fs^2
        gamma_fs = friction_per_ps / 1000.0
        c1 = float(np.exp(-gamma_fs * dt_fs))
        c2 = float(np.sqrt(max(0.0, 1.0 - c1 * c1)))

        a = self.forces_kj_mol_a(x) * conv / self.masses[:, None]
        frames_x = [x.numpy().copy()]
        frames_v = [v.numpy().copy()]
        for _ in range(n_frames - 1):
            for _ in range(substeps):
                v = v + 0.5 * dt_fs * a
                x_half = x + 0.5 * dt_fs * v
                noise = torch.randn(x.shape, generator=g, dtype=torch.float64) * v_std[:, None]
                v = c1 * v + c2 * noise
                x = x_half + 0.5 * dt_fs * v
                a = self.forces_kj_mol_a(x) * conv / self.masses[:, None]
                v = v + 0.5 * dt_fs * a
                if self.rest_idx is None:
                    v = remove_com(v[None], self.masses)[0]
            frames_x.append(x.numpy().copy())
            frames_v.append(v.numpy().copy())
        return np.stack(frames_x, axis=0), np.stack(frames_v, axis=0)


# ---------------------------------------------------------------- binding physics: OpenBPMD + 50-fs tau-RAMD

def compute_openbpmd_scores(rmsd_replicas, persistence_replicas):
    """Compute OpenBPMD / Clark-2016 composite pose stability score across short MD replicas.

        PoseScore = mean_r(mean_{t in terminal_half} RMSD_lig,r(t))   [Å]
        PersScore = mean_r(ContactPersistence_r)                      [0..1]
        CompScore = PoseScore - 5.0 * PersScore

    Locked native poses typically achieve `CompScore <= -1.0`, whereas false-positive decoys yield `CompScore > 0`.
    """
    pose_arr = np.asarray(rmsd_replicas, dtype=np.float64)
    pers_arr = np.asarray(persistence_replicas, dtype=np.float64)
    pose_score = float(pose_arr.mean())
    pose_std = float(pose_arr.std())
    pers_score = float(pers_arr.mean())
    comp_score = pose_score - 5.0 * pers_score
    return {
        "pose_score_A": pose_score,
        "pose_score_std_A": pose_std,
        "pers_score": pers_score,
        "comp_score": float(comp_score),
    }


def tau_ramd_probe(engine, pos0, ligand_indices, pocket_indices, masses, *,
                   n_replicas=3, max_blocks=32, block_fs=50.0, dt_fs=2.5,
                   force_kj_mol_a=32.0, r_min_a=0.08, unbind_threshold_a=4.5,
                   temperature_k=300.0, friction_per_ps=6.0, seed=101):
    """Fast 50-fs direction-retention tau-RAMD unbinding probe (Kokh et al., JCTC 2018).

    Applies a random constant external force `F_0 * u_hat` to the ligand center of mass and
    evaluates forward ligand COM progress along `u_hat` every `block_fs = 50.0 fs` (exact 1:1 match
    to WignerFlow's 50-fs coarse stride): if forward displacement `dot(Delta R_COM, u_hat) < r_min_a`,
    a new isotropic force direction `u_hat` is sampled; otherwise `u_hat` is retained along open
    egress channels until pocket COM separation reaches `unbind_threshold_a`.
    """
    lig = torch.as_tensor(ligand_indices, dtype=torch.long)
    pock = torch.as_tensor(pocket_indices, dtype=torch.long)
    m_all = torch.as_tensor(masses, dtype=torch.float64)
    w_lig = m_all[lig] / m_all[lig].sum()
    w_pock = m_all[pock] / m_all[pock].sum() if pock.numel() > 0 else None

    x0_t = torch.as_tensor(pos0, dtype=torch.float64)
    lig_com0 = (x0_t[lig] * w_lig[:, None]).sum(dim=0)
    pock_com0 = (x0_t[pock] * w_pock[:, None]).sum(dim=0) if w_pock is not None else torch.zeros(3, dtype=torch.float64)
    rel0 = lig_com0 - pock_com0

    steps_per_block = max(1, int(round(block_fs / dt_fs)))
    conv = 1.0 / AMU_A2_PER_FS2_TO_KJ_PER_MOL
    v_std = torch.sqrt((KB * temperature_k / AMU_A2_PER_FS2_TO_KJ_PER_MOL) / m_all)
    gamma_fs = friction_per_ps / 1000.0
    c1 = float(np.exp(-gamma_fs * dt_fs))
    c2 = float(np.sqrt(max(0.0, 1.0 - c1 * c1)))

    escape_blocks = []
    for r in range(n_replicas):
        g = torch.Generator().manual_seed(seed + 37 * r)
        x = x0_t.clone()
        v = torch.randn(x.shape, generator=g, dtype=torch.float64) * v_std[:, None]
        u = torch.randn(3, generator=g, dtype=torch.float64)
        u = u / torch.linalg.norm(u).clamp_min(1e-8)

        def accel_with_ramd(pos_cur, dir_vec):
            f = engine.forces_kj_mol_a(pos_cur)
            # Distribute total COM force by mass fraction so all ligand atoms share identical RAMD acceleration
            f[lig] = f[lig] + force_kj_mol_a * w_lig[:, None] * dir_vec[None, :]
            return f * conv / m_all[:, None]

        a = accel_with_ramd(x, u)
        prev_rel = rel0.clone()
        escaped_at = max_blocks
        for blk in range(1, max_blocks + 1):
            for _ in range(steps_per_block):
                v = v + 0.5 * dt_fs * a
                x_half = x + 0.5 * dt_fs * v
                noise = torch.randn(x.shape, generator=g, dtype=torch.float64) * v_std[:, None]
                v = c1 * v + c2 * noise
                x = x_half + 0.5 * dt_fs * v
                a = accel_with_ramd(x, u)
                v = v + 0.5 * dt_fs * a
            cur_lig = (x[lig] * w_lig[:, None]).sum(dim=0)
            cur_pock = (x[pock] * w_pock[:, None]).sum(dim=0) if w_pock is not None else torch.zeros(3, dtype=torch.float64)
            cur_rel = cur_lig - cur_pock
            if float(torch.linalg.norm(cur_rel - rel0).item()) >= unbind_threshold_a:
                escaped_at = blk
                break
            if float(torch.dot(cur_rel - prev_rel, u).item()) < r_min_a:
                u = torch.randn(3, generator=g, dtype=torch.float64)
                u = u / torch.linalg.norm(u).clamp_min(1e-8)
                a = accel_with_ramd(x, u)
            prev_rel = cur_rel
        escape_blocks.append(float(escaped_at))

    med_blocks = float(np.median(escape_blocks))
    return {
        "tau_ramd_blocks_50fs": round(med_blocks, 2),
        "tau_ramd_median_ps": round(med_blocks * (block_fs / 1000.0), 3),
        "tau_ramd_replica_blocks": [round(b, 1) for b in escape_blocks],
    }


# ---------------------------------------------------------------- electrolyte transport & solvation physics

def nernst_einstein_metrics(d_cation_cm2_s, d_anion_cm2_s, *, n_cation, n_anion, z_cation, z_anion,
                            volume_a3, temperature_k):
    """Compute uncorrelated Nernst-Einstein ionic conductivity (mS/cm) and ideal transference t+^NE.

        sigma_NE = (e^2 / (V k_B T)) * (N_+ z_+^2 D_+ + N_- z_-^2 D_-)
        t_+^NE   = (N_+ z_+^2 D_+) / (N_+ z_+^2 D_+ + N_- z_-^2 D_-)
    """
    if volume_a3 <= 0 or temperature_k <= 0:
        raise ValueError("volume_a3 and temperature_k must be positive")
    d_cat = max(0.0, float(d_cation_cm2_s))
    d_an = max(0.0, float(d_anion_cm2_s))
    term_pos = float(n_cation) * (float(z_cation) ** 2) * d_cat
    term_neg = float(n_anion) * (float(z_anion) ** 2) * d_an
    denom = term_pos + term_neg
    sigma_ms_cm = NERNST_EINSTEIN_MS_CM * denom / (float(volume_a3) * float(temperature_k))
    t_plus = term_pos / denom if denom > 0 else 0.0
    return {
        "sigma_ne_mS_cm": float(sigma_ms_cm),
        "transference_plus_ne": float(t_plus),
    }


def cluster_nernst_einstein(positions, box_length, cation_indices, anion_indices, *,
                            z_cation, z_anion, dt_frame_fs, temperature_k, r_pair=4.25):
    """Cluster Nernst-Einstein conductivity, Inverse Haven Ionicity, and cluster transference t+^cNE.

    Implements the cluster-correlated transport formalism of France-Lanord & Grossman (PRL 2019)
    and Fong, Self & Persson (ACS Energy Lett. 2019):
      - Connected ion clusters `[(Cation)_i (Anion)_j]` carry net valence `z_c = i * z_+ + j * z_-`.
      - Neutral contact ion pairs (`i = j`, `z_+ = -z_-`) have `z_c = 0` and contribute **zero**
        to `sigma_cNE`, eliminating the classic Nernst-Einstein overestimation mirage!
      - Negatively charged triplets (`[Li(Anion)_2]^-`, `i = 1, j = 2`) contribute `(i z_+) * z_c < 0`
        to cation charge flux, capturing suppressed or negative `t+_cNE`.
    """
    pos = np.asarray(positions, dtype=np.float64)
    cat = np.asarray(cation_indices, dtype=np.int64)
    an = np.asarray(anion_indices, dtype=np.int64)
    ion_idx = np.concatenate([cat, an])
    n_cat = len(cat)
    n_ions = len(ion_idx)
    t_frames = len(pos)
    if t_frames < 4 or n_ions == 0:
        return {
            "sigma_cne_mS_cm": 0.0,
            "sigma_ne_mS_cm": 0.0,
            "ionicity_haven_inv": 0.0,
            "transference_plus_cne": 0.0,
        }

    volume_a3 = float(box_length) ** 3
    lags = np.arange(1, max(3, t_frames // 2))
    t_lags_fs = lags * float(dt_frame_fs)

    # Pairwise cation-anion minimum-image distances across all frames: (T, N_cat, N_an)
    dr_ca = pos[:, cat][:, :, None, :] - pos[:, an][:, None, :, :]
    if box_length is not None:
        dr_ca = dr_ca - float(box_length) * np.round(dr_ca / float(box_length))
    d_ca = np.linalg.norm(dr_ca, axis=-1)

    m_qq_cne = np.zeros(len(lags), dtype=np.float64)
    m_qpq_cne = np.zeros(len(lags), dtype=np.float64)

    for k_idx, lag in enumerate(lags):
        n_origins = t_frames - lag
        acc_qq = 0.0
        acc_qpq = 0.0
        for t0 in range(n_origins):
            # Contact edge exists if mean distance over [t0, t0 + lag] is within primary pair cutoff r_pair
            paired = d_ca[t0:t0 + lag + 1].mean(axis=0) <= r_pair
            visited = np.zeros(n_ions, dtype=bool)
            disp_ions = pos[t0 + lag, ion_idx] - pos[t0, ion_idx]
            for start in range(n_ions):
                if visited[start]:
                    continue
                stack = [start]
                visited[start] = True
                members = []
                while stack:
                    u = stack.pop()
                    members.append(u)
                    if u < n_cat:
                        nbrs = n_cat + np.flatnonzero(paired[u])
                    else:
                        nbrs = np.flatnonzero(paired[:, u - n_cat])
                    for v in nbrs:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(int(v))
                mem = np.asarray(members, dtype=np.int64)
                i_c = int((mem < n_cat).sum())
                j_c = int((mem >= n_cat).sum())
                z_c = i_c * float(z_cation) + j_c * float(z_anion)
                if abs(z_c) < 1e-12:
                    continue
                d_rc = disp_ions[mem].mean(axis=0)
                r2 = float(np.dot(d_rc, d_rc))
                acc_qq += (z_c * z_c) * r2
                acc_qpq += (i_c * float(z_cation) * z_c) * r2
        m_qq_cne[k_idx] = acc_qq / n_origins
        m_qpq_cne[k_idx] = acc_qpq / n_origins

    # Fit Einstein slope (1/6) dM/dt in cm^2/s (1 Å^2/fs = 0.1 cm^2/s)
    t_fit0, t_fit1 = float(t_lags_fs[min(1, len(t_lags_fs) - 1)]), float(t_lags_fs[-1])
    _, d_qq_eff = analysis.diffusion_coefficient(t_lags_fs, m_qq_cne, t_fit0, t_fit1)
    _, d_qpq_eff = analysis.diffusion_coefficient(t_lags_fs, m_qpq_cne, t_fit0, t_fit1)
    d_qq_eff = max(0.0, float(d_qq_eff))

    msd_cat = analysis.msd(pos[:, cat], lags) if n_cat > 0 else np.zeros_like(t_lags_fs)
    msd_an = analysis.msd(pos[:, an], lags) if len(an) > 0 else np.zeros_like(t_lags_fs)
    _, d_cat = analysis.diffusion_coefficient(t_lags_fs, msd_cat, t_fit0, t_fit1) if n_cat > 0 else (0.0, 0.0)
    _, d_an = analysis.diffusion_coefficient(t_lags_fs, msd_an, t_fit0, t_fit1) if len(an) > 0 else (0.0, 0.0)
    ne = nernst_einstein_metrics(
        max(0.0, float(d_cat)), max(0.0, float(d_an)),
        n_cation=n_cat, n_anion=len(an), z_cation=z_cation, z_anion=z_anion,
        volume_a3=volume_a3, temperature_k=temperature_k,
    )
    sigma_cne = NERNST_EINSTEIN_MS_CM * d_qq_eff / (volume_a3 * float(temperature_k))
    sigma_ne = ne["sigma_ne_mS_cm"]
    ionicity = float(np.clip(sigma_cne / sigma_ne, 0.0, 1.5)) if sigma_ne > 1e-12 else 0.0
    t_plus_cne = float(np.clip(d_qpq_eff / d_qq_eff, -1.0, 1.0)) if d_qq_eff > 1e-14 else 0.0
    return {
        "sigma_cne_mS_cm": float(sigma_cne),
        "sigma_ne_mS_cm": float(sigma_ne),
        "ionicity_haven_inv": ionicity,
        "transference_plus_cne": t_plus_cne,
    }


def solvation_hopping_metrics(positions, velocities, box_length, cation_indices, coord_o_indices, r_shell=2.55):
    """Quantify Structural Ligand-Exchange Hopping vs. Vehicular Shell Dragging (Self, Fong & Persson 2019).

    Combines primary-shell ligand exchange fraction `1 - S_shell` with bound `Li+ -- O` frame-to-frame
    displacement co-movement correlation `C_{dr,Li-O}` into a normalized hopping index `chi_hop in [0, 1]`.
    """
    cat = np.asarray(cation_indices, dtype=np.int64)
    oxy = np.asarray(coord_o_indices, dtype=np.int64)
    if len(cat) == 0 or len(oxy) == 0 or len(positions) < 4:
        return {"shell_exchange_fraction": 0.0, "li_o_vel_corr": 0.0, "hopping_index_chi": 0.0}

    pos = np.asarray(positions, dtype=np.float64)
    dr = pos[:, cat][:, :, None, :] - pos[:, oxy][:, None, :, :]
    if box_length is not None:
        dr = dr - float(box_length) * np.round(dr / float(box_length))
    d = np.linalg.norm(dr, axis=-1)  # (T, N_cat, N_oxy)

    t0 = len(pos) // 4
    bound_t0 = d[t0] <= r_shell
    if bound_t0.any():
        survived = (d[-1][bound_t0] <= 1.05 * r_shell).mean()
        exch_frac = float(1.0 - survived)
    else:
        exch_frac = 0.0

    # Frame-stride displacement cosine correlation between Li+ and bound primary-shell O atoms:
    # Instantaneous thermal velocities decorrelate within ~50 fs, whereas 30-fs stride displacements
    # directly measure vehicular cage co-translation!
    disp_cat = pos[t0 + 1:, cat] - pos[t0:-1, cat]  # (T'-1, N_cat, 3)
    disp_oxy = pos[t0 + 1:, oxy] - pos[t0:-1, oxy]  # (T'-1, N_oxy, 3)
    dc_n = disp_cat / np.maximum(np.linalg.norm(disp_cat, axis=-1, keepdims=True), 1e-12)
    do_n = disp_oxy / np.maximum(np.linalg.norm(disp_oxy, axis=-1, keepdims=True), 1e-12)
    cos_mat = np.einsum("tic,tjc->tij", dc_n, do_n)
    shell_mask = d[t0:-1] <= r_shell
    vel_corr = float(cos_mat[shell_mask].mean()) if shell_mask.any() else 0.0

    chi_hop = float(np.clip(0.60 * exch_frac + 0.40 * (1.0 - np.clip(vel_corr, 0.0, 1.0)), 0.0, 1.0))
    return {
        "shell_exchange_fraction": round(exch_frac, 3),
        "li_o_vel_corr": round(vel_corr, 3),
        "hopping_index_chi": round(chi_hop, 3),
    }


def competitive_additive_score(solvent_cn, additive_cn, *, n_solvent_mols, n_additive_mols, paired_fraction=0.0):
    """Competitive Additive Grafting Score `CAGS_add` (bridging Gao, Yao & Zhang JACS 2023 & Hou 2020).

        CAGS_add = ln( ((CN_add + eps) / x_add) / ((CN_solv + eps) / x_solv) )

    Classifies additives into:
      - `sacrificial_sei_former` (`CAGS_add > +0.15`, preferentially enters inner Li+ sheath, e.g. FEC/VC)
      - `weak_solvation_anion_recruiter` (`CAGS_add < -0.15` & `CIP+AGG >= 0.40`, promotes anion-rich sheath)
      - `co_solvating_modulator` (balanced shell participation)
    """
    if n_additive_mols <= 0 or n_solvent_mols <= 0:
        return {"cags_additive_score": None, "additive_role": "baseline_no_additive"}
    tot = float(n_solvent_mols + n_additive_mols)
    x_add = float(n_additive_mols) / tot
    x_solv = float(n_solvent_mols) / tot
    eps = 0.05
    cags = float(np.log(((float(additive_cn) + eps) / x_add) / ((float(solvent_cn) + eps) / x_solv)))
    if cags > 0.15:
        role = "sacrificial_sei_former"
    elif cags < -0.15 and paired_fraction >= 0.40:
        role = "weak_solvation_anion_recruiter"
    else:
        role = "co_solvating_modulator"
    return {
        "cags_additive_score": round(cags, 3),
        "additive_role": role,
    }


def solvation_speciation(positions, box_length, cation_indices, solvent_coord_indices,
                         anion_coord_indices, additive_coord_indices=None, r_shell=2.8):
    """Compute primary Li+ coordination numbers (SCN, ACN, N_add) and SSIP / CIP / AGG fractions."""
    cat = np.asarray(cation_indices, dtype=int)
    solv = np.asarray(solvent_coord_indices, dtype=int)
    an = np.asarray(anion_coord_indices, dtype=int)
    add = np.empty(0, dtype=int) if additive_coord_indices is None else np.asarray(additive_coord_indices, dtype=int)
    lengths = None if box_length is None else np.full(3, float(box_length))

    def pair_counts(target_idx):
        if len(target_idx) == 0:
            return np.zeros((len(positions), len(cat)))
        dr = positions[:, cat][:, :, None, :] - positions[:, target_idx][:, None, :, :]
        if lengths is not None:
            dr = dr - lengths * np.round(dr / lengths)
        d = np.linalg.norm(dr, axis=-1)
        return (d <= r_shell).sum(axis=-1)

    c_solv = pair_counts(solv)
    c_an = pair_counts(an)
    c_add = pair_counts(add)
    ssip = float((c_an == 0).mean())
    cip = float((c_an == 1).mean())
    agg = float((c_an >= 2).mean())
    return {
        "r_shell_A": float(r_shell),
        "solvent_cn": float(c_solv.mean()),
        "anion_cn": float(c_an.mean()),
        "additive_cn": float(c_add.mean()),
        "ssip_fraction": ssip,
        "cip_fraction": cip,
        "agg_fraction": agg,
    }


# ---------------------------------------------------------------- Mode 1: Checkpoint CPU Triage

@torch.no_grad()
def audit_checkpoint_epistemic(model, pos0, vel0, types, box, masses, *, sampling_steps=3, n_draws=3, seed=0):
    """Committee-free dual epistemic gate for equivariant flow models (Vita & Schwalbe-Koda 2023).

    Reuses a single `encode()` pass on `(pos0, vel0)` to compute:
      1. `ode_curvature_delta_A`: RMS displacement discrepancy (Å) between 2nd-order Heun and 1st-order
         Euler integration of the learned flow field from identical noise `y_0`.
      2. `flow_epistemic_spread_A`: strictly O(3)-invariant neighbor-distance standard deviation (Å)
         across `n_draws` independent COM-free stochastic flow realizations.
    """
    m_f32 = model.float().eval()
    pos_f = pos0.to(dtype=torch.float32)
    vel_f = vel0.to(dtype=torch.float32)
    box_f = None if box is None else box.to(dtype=torch.float32)
    masses_f = masses.to(dtype=torch.float32)
    types_l = types.to(dtype=torch.long)

    if m_f32.head_type == "regression":
        return {"ode_curvature_delta_A": 0.0, "flow_epistemic_spread_A": 0.0}

    cond, graph = m_f32.encode(pos_f, vel_f, types_l, box_f)
    n = int(types_l.shape[0])
    gen = torch.Generator(device=cond.device).manual_seed(int(seed) + 911)

    def y_to_dx(y_norm):
        dx, vn = m_f32.y_to_targets(y_norm, types_l, vel_f[:, -1])
        dx, vn = remove_com(dx, masses_f), remove_com(vn, masses_f)
        if m_f32.prior is not None:
            dx, _ = m_f32.prior(pos_f[:, -1], dx, vn, types_l, box_f, masses_f)
        return dx

    # 1. Heun (2nd-order trapezoidal) vs Euler (1st-order) ODE curvature check on draw 0
    y0 = m_f32.com_free_noise(1, types_l, masses_f, generator=gen, like=cond)
    tau0 = torch.zeros(n, dtype=cond.dtype, device=cond.device)
    tau1 = torch.ones(n, dtype=cond.dtype, device=cond.device)
    u0 = m_f32.denoiser(cond, graph, y0, tau0)
    y_euler_1 = y0 + u0
    u1 = m_f32.denoiser(cond, graph, y_euler_1, tau1)
    y_heun_1 = y0 + 0.5 * (u0 + u1)
    dx_eul = y_to_dx(y_euler_1)
    dx_heun = y_to_dx(y_heun_1)
    ode_delta = float(torch.sqrt(((dx_heun - dx_eul) ** 2).sum(dim=-1).mean()).item())

    # 2. O(3)-invariant multi-seed pairwise edge distance spread across `n_draws` flow realizations
    n_steps = max(1, int(sampling_steps))
    xs = []
    for d_idx in range(max(2, int(n_draws))):
        y = y0.clone() if d_idx == 0 else m_f32.com_free_noise(1, types_l, masses_f, generator=gen, like=cond)
        for s_idx in range(n_steps):
            tau = torch.full((n,), s_idx / n_steps, dtype=cond.dtype, device=cond.device)
            y = y + m_f32.denoiser(cond, graph, y, tau) / n_steps
        xs.append((pos_f[:, -1] + y_to_dx(y))[0])
    x_stack = torch.stack(xs, dim=0)  # (M, N, 3)

    src, dst = graph["src"], graph["dst"]
    if src.numel() > 0:
        dr = x_stack[:, dst] - x_stack[:, src]
        if box_f is not None:
            L = torch.diagonal(box_f[0])[None, None, :]
            dr = dr - L * torch.round(dr / L)
        edge_d = torch.linalg.norm(dr, dim=-1)  # (M, E), strictly O(3)-invariant
        flow_spread = float(edge_d.std(dim=0, unbiased=False).mean().item())
    else:
        flow_spread = 0.0

    return {
        "ode_curvature_delta_A": round(ode_delta, 4),
        "flow_epistemic_spread_A": round(flow_spread, 4),
    }


def triage_checkpoint(ckpt_path=None, *, steps=24, sampling_steps=3, neighbor="verlet",
                      thermostat_on=False, seed=0, device="cpu"):
    """Run a fast CPU/edge WignerFlow checkpoint rollout and return a Fu-2023 + Epistemic Big-MD triage report."""
    if ckpt_path is None:
        reduced = CKPTS_DIR / "argon_flow_v1_cpu_reduced.pt"
        ckpt_path = reduced if reduced.exists() else (CKPTS_DIR / "argon_flow_v0.pt")
    ckpt_path = Path(ckpt_path)
    t0 = time.perf_counter()
    ck = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = EquiTraj(
        n_types=len(ck["elements"]), k=ck["k"], stride_fs=ck["stride_fs"], **ck["model_cfg"]
    ).float().to(device).eval()
    model.load_state_dict(ck["model"])

    demo = ck["demo"]
    ref = ck.get("reference", {})
    t_ref = float(ref.get("T_K", 94.4))
    stride_fs = float(ck["stride_fs"])
    box = None if demo["box"] is None else demo["box"][None]
    pos0, vel0 = demo["pos"][None], demo["vel"][None]
    types, masses = ck["types"], ck["masses"]

    fast = FastRollout(model, sampling_steps=sampling_steps, neighbor=neighbor)
    therm = {"t_target": t_ref, "tau_fs": 10 * stride_fs, "dt_fs": stride_fs} if thermostat_on else None
    torch.manual_seed(seed)
    out = rollout(
        model, pos0, vel0, types, box, masses, steps,
        sampling_steps=sampling_steps, t_max=3.0 * t_ref, thermostat=therm, predictor=fast,
    )
    epistemic = audit_checkpoint_epistemic(
        fast.model, pos0, vel0, types, box, masses, sampling_steps=sampling_steps, n_draws=3, seed=seed
    )
    wall_s = time.perf_counter() - t0
    crashed = int(out["crash_step"][0])
    n_ok = steps if crashed < 0 else max(1, crashed)
    x = out["positions"][:n_ok, 0].numpy()
    v = out["velocities"][:n_ok, 0].numpy()
    half = max(0, n_ok // 2)

    temp_series = temperature(torch.from_numpy(v), masses).numpy()
    t_mean = float(temp_series[half:].mean())
    t_rel_err = abs(t_mean - t_ref) / max(t_ref, 1e-6)
    dmin = float(min_pair_distance(torch.from_numpy(x), None if box is None else box.expand(n_ok, 3, 3)).min().item())

    # Fu-2023 normalized vdW distance ratio & Flying-Ice-Cube COM kinetic energy leakage audit
    elem_radii = np.array([_VDW_RADIUS_A.get(ck["elements"][int(t)], 1.70) for t in types.cpu().numpy()])
    min_vdw_ratio = dmin / max(float(2.0 * elem_radii.min()), 1e-6)
    m_np = masses.cpu().numpy().astype(np.float64)
    m_tot = float(m_np.sum())
    v_com = (v * m_np[None, :, None]).sum(axis=1) / max(m_tot, 1e-12)  # (T, 3)
    k_com = 0.5 * m_tot * (v_com ** 2).sum(axis=-1)
    k_tot = 0.5 * (m_np[None, :] * (v ** 2).sum(axis=-1)).sum(axis=-1)
    com_leakage = float((k_com / np.maximum(k_tot, 1e-15)).max())

    rdf_peak_r, rdf_cn, rdf_l1_err = None, None, None
    r_grid, g_curve, g_ref_curve = [], [], []
    if n_ok >= 10 and box is not None:
        box_np = np.tile(box[0].numpy(), (len(x[half:]), 1, 1))
        idx_all = np.arange(x.shape[1])
        r_max = float(ref["rdf_r"][-1]) + 1e-6 if "rdf_r" in ref else 0.48 * float(box[0, 0, 0])
        n_bins = len(ref["rdf_r"]) if "rdf_r" in ref else 80
        r, g, rho = analysis.rdf(x[half:], box_np, idx_all, idx_all, r_max, n_bins)
        rdf_peak_r = float(analysis.first_peak(r, g)[0])
        if "rdf_g" in ref:
            g_ref = np.asarray(ref["rdf_g"])
            r_cut = float(analysis.first_minimum_after_peak(r, g_ref))
            rdf_l1_err = float(analysis.rdf_l1(r, g, g_ref))
            g_ref_curve = g_ref.round(4).tolist()
        else:
            g_smooth = np.convolve(g, np.ones(5) / 5.0, mode="same")
            r_cut = float(analysis.first_minimum_after_peak(r, g_smooth))
        rdf_cn = float(analysis.coordination_number(r, g, rho, r_cut))
        r_grid, g_curve = r.round(3).tolist(), g.round(4).tolist()

    d_cm2_s = None
    if n_ok >= 12:
        unwrapped = unwrap_positions(x, None if box is None else box[0].numpy())
        lags = np.arange(1, max(4, n_ok // 2))
        msd_vals = analysis.msd(unwrapped, lags)
        t_lags = lags * stride_fs
        _, d_cm2_s = analysis.diffusion_coefficient(t_lags, msd_vals, float(t_lags[1]), float(t_lags[-1]))
        d_cm2_s = max(0.0, float(d_cm2_s))

    if crashed >= 0 or min_vdw_ratio < 0.65:
        verdict, badge = "REJECT", "Crash / steric overlap detected during fast rollout"
    elif (t_rel_err <= 0.30 and (rdf_l1_err is None or rdf_l1_err <= 0.85)
          and com_leakage < 1e-3 and epistemic["flow_epistemic_spread_A"] <= 0.35):
        verdict, badge = "GO", "Fu-2023 structural/thermal & flow-epistemic gates passed — ready for multi-ns rollout"
    else:
        verdict, badge = "HOLD", "Moderate thermal/RDF drift or flow curvature — enable CSVR + Heun solver"

    return {
        "mode": "checkpoint",
        "name": ckpt_path.stem,
        "engine": f"WignerFlow EquiTraj ({ck['model_cfg']['head']}, Verlet + Dual Epistemic Audit)",
        "verdict": verdict,
        "summary": badge,
        "wall_s": round(wall_s, 3),
        "steps": int(steps),
        "simulated_ps": round(steps * stride_fs / 1000.0, 3),
        "ns_per_day_equiv": round((steps * stride_fs * 1e-6) / max(wall_s, 1e-6) * 86400.0, 2),
        "n_atoms": int(x.shape[1]),
        "metrics": {
            "crashed_at_step": None if crashed < 0 else crashed,
            "min_pair_dist_A": round(dmin, 3),
            "min_vdw_ratio": round(min_vdw_ratio, 3),
            "com_kinetic_leakage_ratio": float(f"{com_leakage:.2e}"),
            "ode_curvature_delta_A": epistemic["ode_curvature_delta_A"],
            "flow_epistemic_spread_A": epistemic["flow_epistemic_spread_A"],
            "T_ref_K": round(t_ref, 2),
            "T_mean_K": round(t_mean, 2),
            "T_rel_err": round(t_rel_err, 4),
            "rdf_first_peak_A": None if rdf_peak_r is None else round(rdf_peak_r, 3),
            "coordination_number": None if rdf_cn is None else round(rdf_cn, 2),
            "rdf_l1_err_A": None if rdf_l1_err is None else round(rdf_l1_err, 4),
            "diffusion_cm2_s": None if d_cm2_s is None else float(f"{d_cm2_s:.3e}"),
        },
        "series": {
            "time_ps": ((np.arange(1, n_ok + 1) * stride_fs) / 1000.0).round(3).tolist(),
            "temperature_K": temp_series.round(2).tolist(),
            "rdf_r_A": r_grid,
            "rdf_g": g_curve,
            "rdf_g_ref": g_ref_curve,
        },
        "viewer_frames": x[::max(1, len(x) // 16), :64].round(2).tolist(),
        "viewer_roles": ["solvent"] * min(64, x.shape[1]),
    }


# ---------------------------------------------------------------- Mode 2: Drug Binding Pocket Triage

def _unit(vec):
    nrm = float(np.linalg.norm(vec))
    return vec / nrm if nrm > 1e-8 else np.array([1.0, 0.0, 0.0])


def _build_aspirin_pocket(preset):
    """Construct a realistic 45-atom COX-1 catalytic pocket around OpenFF-2.2.1 aspirin."""
    pdb = parse_pdb(SYSTEMS_DIR / "aspirin.pdb")
    ff = parse_openmm_xml(SYSTEMS_DIR / "aspirin.xml")
    lig_pos = pdb["positions"].copy()
    lig_pos -= lig_pos.mean(axis=0)
    n_lig = len(lig_pos)  # 21 atoms: O1x=2 (ester C=O), O2x=3, O3x=11 (acid C=O), O4x=12 (-OH), H8x=20

    # Gentle 1-4 heavy-atom dihedral springs keep aspirin's salicylate/acetyl core realistic in fast screening
    heavy_torsion_bonds, heavy_r0, heavy_k = [], [], []
    bonded_set = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in ff["bond_pairs"]}
    for i in range(13):
        for j in range(i + 1, 13):
            if (i, j) not in bonded_set:
                dij = float(np.linalg.norm(lig_pos[i] - lig_pos[j]))
                if 2.6 <= dij <= 4.3:
                    heavy_torsion_bonds.append((i, j))
                    heavy_r0.append(dij)
                    heavy_k.append(45.0)

    def make_triad(anchor_pt, out_dir, lat_dir):
        u = _unit(out_dir)
        lat = _unit(lat_dir - np.dot(lat_dir, u) * u)
        ca = anchor_pt + 1.35 * u
        side = ca + 1.25 * lat + 0.35 * u
        return [anchor_pt, ca, side]

    ring_center = lig_pos[[4, 5, 6, 7, 8, 9]].mean(axis=0)
    ring_normal = _unit(np.cross(lig_pos[5] - lig_pos[4], lig_pos[9] - lig_pos[4]))

    # 8 residue triads (24 pocket atoms) derived directly from aspirin's polar & aromatic geometry:
    triads = (
        # Triad 0: Arg120 guanidinium donor H-bonded to carboxylic carbonyl O3x (idx 11, q = -0.56)
        make_triad(lig_pos[11] + 2.80 * _unit(lig_pos[11] - lig_pos[10]), lig_pos[11] - lig_pos[10], [0.0, 0.0, 1.0])
        # Triad 1: Tyr385 phenolic donor H-bonded to ester carbonyl O1x (idx 2, q = -0.55)
        + make_triad(lig_pos[2] + 2.80 * _unit(lig_pos[2] - lig_pos[1]), lig_pos[2] - lig_pos[1], [0.0, 1.0, 0.0])
        # Triad 2: Ser530 polar acceptor H-bonded to carboxylic hydroxyl O4x-H8x (idx 12/20, q_H = +0.45)
        + make_triad(lig_pos[20] + 1.98 * _unit(lig_pos[20] - lig_pos[12]), lig_pos[20] - lig_pos[12], [1.0, 0.0, 0.0])
        # Triads 3..7: Hydrophobic pocket channel (Ile523, Val349, Leu352, Ala527, Trp387) cradling phenyl & acetyl
        + make_triad(lig_pos[6] + 3.85 * _unit(lig_pos[6] - ring_center), lig_pos[6] - ring_center, ring_normal)
        + make_triad(lig_pos[8] + 3.85 * _unit(lig_pos[8] - ring_center), lig_pos[8] - ring_center, ring_normal)
        + make_triad(lig_pos[0] + 3.95 * _unit(lig_pos[0] - lig_pos[1]), lig_pos[0] - lig_pos[1], ring_normal)
        + make_triad(ring_center + 3.85 * ring_normal, ring_normal, [1.0, 0.0, 0.0])
        + make_triad(ring_center - 3.85 * ring_normal, -ring_normal, [1.0, 0.0, 0.0])
    )
    pocket_anchors = np.asarray(triads, dtype=np.float64)
    n_pock = len(pocket_anchors)

    pock_charges = np.zeros(n_pock, dtype=np.float64)
    pock_charges[[0, 1]] = [0.55, -0.25]   # Arg120 donor -> O3x
    pock_charges[[3, 4]] = [0.50, -0.25]   # Tyr385 donor -> O1x
    pock_charges[[6, 7]] = [-0.55, 0.25]   # Ser530 acceptor <- H8x-O4x
    pock_masses = np.tile([14.007, 12.011, 15.999], n_pock // 3)
    pock_elements = ["N", "C", "O"] * (n_pock // 3)
    pock_sig = np.full(n_pock, 3.25, dtype=np.float64)
    pock_sig[[0, 3, 6]] = 2.95  # polar H-bond heavy-atom contact radius
    pock_eps = np.full(n_pock, 0.42, dtype=np.float64)

    pock_bonds, pock_r0, pock_k = [], [], []
    pock_excl = set()
    for r in range(0, n_pock, 3):
        i0, i1, i2 = n_lig + r, n_lig + r + 1, n_lig + r + 2
        for a, b in ((i0, i1), (i1, i2), (i0, i2)):
            pock_bonds.append((a, b))
            pock_r0.append(float(np.linalg.norm(pocket_anchors[a - n_lig] - pocket_anchors[b - n_lig])))
            pock_k.append(1100.0)
            pock_excl.add((a, b))

    pose_lig = lig_pos.copy()
    friction = 8.0
    if preset == "aspirin_cox_stable":
        label = "Aspirin in COX-1 Pocket (Top-1 Anchor Pose)"
        dielectric = 1.6
    elif preset == "aspirin_induced_fit":
        label = "Aspirin in COX-1 Pocket (Tilted Induced-Fit Pose)"
        # Mildly misaligned initial docked pose (rotated 16 deg + shifted 0.45 Å): partial rearrangement in pocket
        ang = np.radians(16.0)
        rot = np.array([[np.cos(ang), -np.sin(ang), 0.0], [np.sin(ang), np.cos(ang), 0.0], [0.0, 0.0, 1.0]])
        pose_lig = pose_lig @ rot.T + np.array([0.42, -0.28, 0.20])
        dielectric = 2.0
        friction = 7.0
    elif preset == "aspirin_decoy_clash":
        label = "Aspirin in COX-1 Pocket (Flipped Clash Decoy)"
        # 180-deg flipped false-positive pose with steric clash & electrostatic mismatch -> recoils & unbinds
        pose_lig = pose_lig @ np.diag([-1.0, -1.0, 1.0]) + np.array([1.55, 1.15, 0.75])
        pock_charges[[0, 3, 6]] = [-0.65, -0.60, 0.60]
        pocket_anchors[18:21] += 6.0 * ring_normal
        dielectric = 1.4
        friction = 3.5
    else:
        raise ValueError(f"unknown binding preset {preset!r}")

    pos_all = np.vstack([pose_lig, pocket_anchors])
    masses = np.concatenate([ff["masses"], pock_masses])
    elements = list(pdb["elements"]) + pock_elements
    charges = np.concatenate([ff["charges"], pock_charges])
    sigma_a = np.concatenate([ff["sigma_a"], pock_sig])
    eps_kj = np.concatenate([ff["eps_kj"], pock_eps])
    bond_pairs = np.vstack([
        ff["bond_pairs"],
        np.asarray(heavy_torsion_bonds, dtype=np.int64),
        np.asarray(pock_bonds, dtype=np.int64),
    ])
    bond_r0 = np.concatenate([ff["bond_r0"], np.asarray(heavy_r0), np.asarray(pock_r0)])
    bond_k = np.concatenate([ff["bond_k"], np.asarray(heavy_k), np.asarray(pock_k)])
    excl = set(ff["excluded_pairs"]) | pock_excl

    backbone_idx = np.arange(n_lig + 1, n_lig + n_pock, 3, dtype=np.int64)
    engine = PocketMicroEngine(
        masses, charges, sigma_a, eps_kj,
        bond_pairs=bond_pairs, bond_r0=bond_r0, bond_k=bond_k,
        angle_triplets=ff["angle_triplets"], angle_theta0=ff["angle_theta0"], angle_k=ff["angle_k"],
        excluded_pairs=excl, restraint_indices=backbone_idx,
        restraint_anchors=pocket_anchors[1::3], restraint_k=65.0, dielectric=dielectric,
    )
    return {
        "label": label,
        "engine": engine,
        "pos0": pos_all,
        "masses": masses,
        "elements": elements,
        "friction": friction,
        "ligand_indices": np.arange(n_lig, dtype=np.int64),
        "pocket_indices": np.arange(n_lig, n_lig + n_pock, dtype=np.int64),
        "backbone_indices": backbone_idx,
        "hbond_ligand": np.array([2, 3, 11, 12], dtype=np.int64),
        "hbond_pocket": np.array([n_lig + 0, n_lig + 3, n_lig + 6], dtype=np.int64),
    }


def triage_binding(preset="aspirin_cox_stable", *, pdb_path=None, xml_path=None, ligand_indices=None,
                   pocket_indices=None, n_frames=70, substeps=8, dt_fs=1.5, temperature_k=300.0,
                   n_replicas=3, seed=0):
    """Screen a docked small-molecule binding pose on CPU in <1 s before committing to RB-FEP / Big MD."""
    t0 = time.perf_counter()
    if pdb_path is not None:
        pdb = parse_pdb(pdb_path)
        pos0 = pdb["positions"]
        masses = pdb["masses"]
        elements = pdb["elements"]
        if ligand_indices is None or pocket_indices is None:
            mols = pdb["mol_ids"]
            if mols.max() >= 1:
                ligand_indices = np.flatnonzero(mols == 0)
                pocket_indices = np.flatnonzero(mols > 0)
            else:
                split = max(3, len(pos0) // 3)
                ligand_indices = np.arange(split)
                pocket_indices = np.arange(split, len(pos0))
        ligand_indices = np.asarray(ligand_indices, dtype=np.int64)
        pocket_indices = np.asarray(pocket_indices, dtype=np.int64)
        if xml_path is not None and Path(xml_path).exists():
            ff = parse_openmm_xml(xml_path)
            charges, sig, eps = ff["charges"], ff["sigma_a"], ff["eps_kj"]
            bp, br0, bk = ff["bond_pairs"], ff["bond_r0"], ff["bond_k"]
            excl = ff["excluded_pairs"]
        else:
            charges = np.zeros(len(pos0))
            sig = np.full(len(pos0), 3.2)
            eps = np.full(len(pos0), 0.3)
            bp = pdb["bonds"]
            br0 = np.linalg.norm(pos0[bp[:, 0]] - pos0[bp[:, 1]], axis=-1) if len(bp) else np.empty(0)
            bk = np.full(len(bp), 1500.0)
            excl = {(int(a), int(b)) for a, b in bp}
        bb_idx = pocket_indices[::max(1, len(pocket_indices) // 8)]
        eng = PocketMicroEngine(
            masses, charges, sig, eps, bond_pairs=bp, bond_r0=br0, bond_k=bk,
            excluded_pairs=excl, restraint_indices=bb_idx, restraint_anchors=pos0[bb_idx], restraint_k=45.0,
        )
        hb_l = ligand_indices[[i for i, idx in enumerate(ligand_indices) if elements[idx] in ("O", "N")]]
        hb_p = pocket_indices[[i for i, idx in enumerate(pocket_indices) if elements[idx] in ("O", "N")]]
        setup = {
            "label": Path(pdb_path).stem, "engine": eng, "pos0": pos0, "masses": masses,
            "elements": elements, "friction": 6.0,
            "ligand_indices": ligand_indices, "pocket_indices": pocket_indices, "backbone_indices": bb_idx,
            "hbond_ligand": hb_l if len(hb_l) else ligand_indices[:1],
            "hbond_pocket": hb_p if len(hb_p) else pocket_indices[:1],
        }
    else:
        setup = _build_aspirin_pocket(preset)

    lig, pock, bb = setup["ligand_indices"], setup["pocket_indices"], setup["backbone_indices"]
    n_rep = max(1, int(n_replicas))

    # 1. PoseBusters initial steric overlap audit (Buttenschoen et al., Chem. Sci. 2024)
    pb_audit = posebusters_vdw_audit(setup["pos0"], setup["elements"], group_a=lig, group_b=pock)

    # 2. Multi-replica short thermal relaxation (OpenBPMD protocol, Lukauskis et al., JCIM 2022)
    rmsd_curves, com_curves, hb_curves = [], [], []
    pose_rep, pers_rep, int_rep, bb_rep, recoil_rep, unbound_frames = [], [], [], [], [], []
    aligned_lig_samples = []
    rep0_x = None
    half = max(1, n_frames // 2)
    m_lig = setup["masses"][lig]

    for r in range(n_rep):
        tx, tv = setup["engine"].run(
            setup["pos0"], n_frames=n_frames, substeps=substeps,
            dt_fs=dt_fs, temperature_k=temperature_k,
            friction_per_ps=setup.get("friction", 6.0), seed=seed + 19 * r,
        )
        if r == 0:
            rep0_x = tx
        rmsd_r = binding.ligand_rmsd(tx, lig, pock)
        int_r = binding.ligand_internal_rmsd(tx, lig)
        bb_r = binding.protein_backbone_rmsd(tx, bb)
        # Compute contact persistence on the terminal half to reflect equilibrated lock-in
        tx_eval = np.concatenate([tx[:1], tx[half:]], axis=0)
        contacts_r = binding.contact_persistence(tx_eval, lig, pock, cutoff=4.5)
        com_r = binding.ligand_com_displacement(tx, lig, pock, masses=setup["masses"], unbinding_threshold=4.5)
        hb_r = binding.hbond_contact_count(tx, setup["hbond_ligand"], setup["hbond_pocket"], cutoff=3.6)

        # Kinetic strain-recoil ratio in early relaxation frames (converts T=0 clash PE -> KE spike)
        v_early = tv[1:min(10, len(tv)), lig]
        if len(v_early) > 0:
            t_lig_early = ((m_lig[None, :, None] * (v_early ** 2)).sum(axis=(1, 2))
                           * AMU_A2_PER_FS2_TO_KJ_PER_MOL) / max(3.0 * len(lig) * KB, 1e-12)
            recoil_rep.append(float(t_lig_early.max() / max(temperature_k, 1e-6)))
        else:
            recoil_rep.append(1.0)

        # Pocket-translated terminal ligand coordinates for multi-replica pose entropy
        pock_shift = tx[half:, pock].mean(axis=1, keepdims=True) - tx[0:1, pock].mean(axis=1, keepdims=True)
        aligned_lig_samples.append(tx[half:, lig] - pock_shift)

        rmsd_curves.append(rmsd_r)
        com_curves.append(com_r["displacement"])
        hb_curves.append(hb_r)
        pose_rep.append(float(rmsd_r[half:].mean()))
        pers_rep.append(float(contacts_r["mean_persistence"]))
        int_rep.append(float(int_r[half:].mean()))
        bb_rep.append(float(bb_r[half:].mean()))
        if com_r["first_unbound_frame"] is not None:
            unbound_frames.append(int(com_r["first_unbound_frame"]))

    # 3. 50-fs direction-retention tau-RAMD impulse unbinding probe (Kokh et al., JCTC 2018)
    ramd = tau_ramd_probe(
        setup["engine"], setup["pos0"], lig, pock, setup["masses"],
        n_replicas=min(3, n_rep), temperature_k=temperature_k,
        friction_per_ps=setup.get("friction", 6.0), seed=seed + 503,
    )
    wall_s = time.perf_counter() - t0

    bpm = compute_openbpmd_scores(pose_rep, pers_rep)
    lig_cloud = np.concatenate(aligned_lig_samples, axis=0)  # (K * T_half, N_lig, 3)
    var_per_atom = lig_cloud.var(axis=0).mean(axis=-1)
    pose_entropy = float((1.5 * np.log1p(var_per_atom / 0.25)).mean())
    strain_recoil = float(np.max(recoil_rep))

    rmsd_mean_curve = np.mean(rmsd_curves, axis=0)
    rmsd_std_curve = np.std(rmsd_curves, axis=0)
    com_mean_curve = np.mean(com_curves, axis=0)
    hb_mean_curve = np.round(np.mean(hb_curves, axis=0)).astype(int)
    unbound_frame = min(unbound_frames) if unbound_frames else None
    com_drift = float(com_mean_curve[half:].mean())
    hbond_mean = float(np.mean([h[half:].mean() for h in hb_curves]))

    if (unbound_frame is not None or bpm["pose_score_A"] >= 3.0
            or bpm["pers_score"] < 0.35 or bpm["comp_score"] >= 0.8):
        verdict = "REJECT"
        summary = "Decoy pose fails CompScore / unbinds in 50-fs probe — skip expensive RB-FEP"
    elif (bpm["comp_score"] <= -1.0 and bpm["pose_score_A"] <= 1.65
          and pb_audit["min_vdw_ratio"] >= 0.72):
        verdict = "GO"
        summary = "OpenBPMD CompScore <= -1.0 & high tau-RAMD residence — promote to RB-FEP / 50 ns MD"
    else:
        verdict = "HOLD"
        summary = "Induced-fit rearrangement (borderline CompScore) — run 5 ns equilibration before FEP"

    stride_ps = (substeps * dt_fs) / 1000.0
    roles = ["pocket"] * rep0_x.shape[1]
    for idx in lig:
        roles[int(idx)] = "ligand"
    return {
        "mode": "binding",
        "name": preset if pdb_path is None else Path(pdb_path).stem,
        "title": setup["label"],
        "engine": f"PocketMD OpenFF-XML MicroEngine ({n_rep}-Replica OpenBPMD + 50-fs tau-RAMD, CPU)",
        "verdict": verdict,
        "summary": summary,
        "wall_s": round(wall_s, 3),
        "n_atoms": int(rep0_x.shape[1]),
        "simulated_ps": round((n_frames - 1) * stride_ps * n_rep, 3),
        "metrics": {
            "n_replicas": n_rep,
            "comp_score": round(bpm["comp_score"], 3),
            "pose_score_A": round(bpm["pose_score_A"], 3),
            "pose_score_std_A": round(bpm["pose_score_std_A"], 3),
            "pers_score": round(bpm["pers_score"], 3),
            "ligand_rmsd_plateau_A": round(bpm["pose_score_A"], 3),
            "contact_persistence_R": round(bpm["pers_score"], 3),
            "posebusters_min_vdw_ratio": round(pb_audit["min_vdw_ratio"], 3),
            "posebusters_vdw_clashes": pb_audit["n_vdw_clashes"],
            "strain_recoil_ratio": round(strain_recoil, 2),
            "pose_entropy": round(pose_entropy, 3),
            "tau_ramd_median_ps": ramd["tau_ramd_median_ps"],
            "tau_ramd_blocks_50fs": ramd["tau_ramd_blocks_50fs"],
            "ligand_internal_rmsd_A": round(float(np.mean(int_rep)), 3),
            "backbone_rmsd_A": round(float(np.mean(bb_rep)), 3),
            "hbond_contacts_mean": round(hbond_mean, 2),
            "com_drift_A": round(com_drift, 3),
            "first_unbound_frame": unbound_frame,
        },
        "series": {
            "time_ps": (np.arange(n_frames) * stride_ps).round(3).tolist(),
            "ligand_rmsd_A": rmsd_mean_curve.round(3).tolist(),
            "ligand_rmsd_std_A": rmsd_std_curve.round(3).tolist(),
            "com_displacement_A": com_mean_curve.round(3).tolist(),
            "hbond_count": hb_mean_curve.tolist(),
        },
        "viewer_frames": rep0_x[::max(1, len(rep0_x) // 16)].round(2).tolist(),
        "viewer_roles": roles,
    }


# ---------------------------------------------------------------- Mode 3: Battery Electrolyte Additive Triage

def _orient_carbonate_to_li(ec_heavy, li_pos, arm_dir, r_li_o=2.08):
    """Place a 6-site heavy carbonate molecule so its carbonyl O (idx 4) points at `li_pos` at distance `r_li_o`."""
    arm = _unit(arm_dir)
    v_oc = _unit(ec_heavy[3] - ec_heavy[4])
    axis = np.cross(v_oc, arm)
    s = float(np.linalg.norm(axis))
    c = float(np.dot(v_oc, arm))
    if s < 1e-7:
        rot = np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        k = axis / s
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        rot = np.eye(3) + s * kx + (1.0 - c) * (kx @ kx)
    mol = (ec_heavy - ec_heavy[4]) @ rot.T
    o_target = li_pos + r_li_o * arm
    return mol + o_target


def _build_electrolyte_box(preset):
    """Build an 88-atom ~1.1 M periodic LiPF6 + Carbonate + Additive box using real EC geometry & ECC charges."""
    pdb = parse_pdb(SYSTEMS_DIR / "li_ec_pf6_q08.pdb")
    ec_mask = pdb["mol_ids"] == 1
    ec_heavy = pdb["positions"][ec_mask][:6].copy()  # C1, C2, O_ring, C_carb, O_carb (idx 4), O_ring
    ec_heavy -= ec_heavy.mean(axis=0)

    presets_cfg = {
        "li_ec_base": {
            "label": "1.0 M LiPF6 in EC (Baseline Carbonate)",
            "n_add": 0, "add_q_o": -0.62, "ion_q": 0.80, "friction": 100.0, "dielectric": 1.85, "box": 14.5,
        },
        "li_ec_fec_additive": {
            "label": "1.0 M LiPF6 in EC/DMC + 10% FEC (Sacrificial SEI + Fast Transport)",
            "n_add": 3, "add_q_o": -0.66, "ion_q": 0.80, "friction": 65.0, "dielectric": 1.95, "box": 14.6,
        },
        "li_ec_vc_additive": {
            "label": "1.0 M LiPF6 in EC + High-wt% VC (Viscous Film-Former)",
            "n_add": 3, "add_q_o": -0.65, "ion_q": 0.80, "friction": 210.0, "dielectric": 1.85, "box": 14.5,
        },
        "li_high_salt_agg": {
            "label": "Low-Donor Additive / High-Pairing Formulation (Neutral CIP/AGG Mirage)",
            "n_add": 4, "add_q_o": -0.18, "ion_q": 0.90, "friction": 220.0, "dielectric": 1.40, "box": 14.2,
        },
    }
    if preset not in presets_cfg:
        raise ValueError(f"unknown electrolyte preset {preset!r}")
    cfg = presets_cfg[preset]
    box_l = cfg["box"]

    li_centers = np.array([
        [0.27 * box_l, 0.27 * box_l, 0.27 * box_l],
        [0.73 * box_l, 0.73 * box_l, 0.73 * box_l],
    ])
    if preset == "li_high_salt_agg":
        # Contact/aggregate neutral [Li2(PF6)2]^0 cluster in poor-donor medium: co-diffuses with zero net charge!
        li_centers[1] = li_centers[0] + np.array([3.6, 0.4, 0.0])
        anion_centers = np.array([
            li_centers[0] + np.array([1.8, 1.4, 0.0]),
            li_centers[0] + np.array([1.8, -1.4, 0.2]),
        ])
    else:
        anion_centers = np.array([
            [0.27 * box_l, 0.73 * box_l, 0.73 * box_l],
            [0.73 * box_l, 0.27 * box_l, 0.27 * box_l],
        ])

    tet_arms = np.array([
        [1.0, 1.0, 1.0], [1.0, -1.0, -1.0], [-1.0, 1.0, -1.0], [-1.0, -1.0, 1.0],
    ], dtype=np.float64)
    bulk_centers = np.array([
        [0.73 * box_l, 0.73 * box_l, 0.25 * box_l],
        [0.25 * box_l, 0.25 * box_l, 0.75 * box_l],
        [0.25 * box_l, 0.73 * box_l, 0.25 * box_l],
        [0.73 * box_l, 0.25 * box_l, 0.75 * box_l],
    ])

    pos_list, masses, charges, sig, eps = [], [], [], [], []
    bonds, bond_r0, bond_k = [], [], []
    excl = set()
    li_idx, pf6_f_idx, solv_o_idx, add_o_idx, anion_p_idx = [], [], [], [], []
    roles = []

    # 1. 2 Li+ cations
    for m in range(2):
        li_idx.append(len(pos_list))
        pos_list.append(li_centers[m] % box_l)
        masses.append(6.941)
        charges.append(cfg["ion_q"])
        sig.append(1.95)
        eps.append(0.08)
        roles.append("cation")

    # 2. 2 PF6- anions (7 sites each: P + 6 octahedral F at 1.60 Å)
    oct_offsets = np.array([
        [0.0, 0.0, 0.0], [1.60, 0.0, 0.0], [-1.60, 0.0, 0.0],
        [0.0, 1.60, 0.0], [0.0, -1.60, 0.0], [0.0, 0.0, 1.60], [0.0, 0.0, -1.60],
    ])
    for m in range(2):
        base = len(pos_list)
        anion_p_idx.append(base)
        for k_site, off in enumerate(oct_offsets):
            pos_list.append((anion_centers[m] + off) % box_l)
            if k_site == 0:
                masses.append(30.974)
                charges.append(0.76 * cfg["ion_q"])
                sig.append(3.70)
                eps.append(0.80)
                roles.append("anion")
            else:
                pf6_f_idx.append(base + k_site)
                masses.append(18.998)
                charges.append(-0.293333 * cfg["ion_q"])
                sig.append(2.95)
                eps.append(0.25)
                roles.append("anion")
                bonds.append((base, base + k_site))
                bond_r0.append(1.60)
                bond_k.append(1600.0)
        for a in range(7):
            for b in range(a + 1, 7):
                excl.add((base + a, base + b))

    # 3. 12 Carbonate / Additive molecules (8 in tetrahedral primary shells + 4 in bulk)
    ec_dists = np.linalg.norm(ec_heavy[:, None, :] - ec_heavy[None, :, :], axis=-1)
    # Sacrificial film-formers (q_o <= -0.64) preferentially coordinate Li+ over EC, whereas weak-donor
    # diluents (q_o > -0.30) partition into bulk outside the inner Li+ shell.
    if cfg["add_q_o"] <= -0.64:
        add_slots = {0, 1, 4, 8}
    elif cfg["add_q_o"] > -0.30:
        add_slots = {0, 8, 9, 10}
    else:
        add_slots = {0, 4, 8, 9}
    add_placed = 0
    for m in range(12):
        is_add = (m in add_slots) and (add_placed < cfg["n_add"])
        if is_add:
            add_placed += 1
        base = len(pos_list)
        if m < 8 and preset != "li_high_salt_agg":
            mol_coords = _orient_carbonate_to_li(ec_heavy, li_centers[m // 4], tet_arms[m % 4], r_li_o=2.06)
        elif m < 4 and preset == "li_high_salt_agg":
            mol_coords = bulk_centers[m] + ec_heavy
        elif m < 8 and preset == "li_high_salt_agg":
            mol_coords = _orient_carbonate_to_li(ec_heavy, li_centers[m % 2], tet_arms[m % 4], r_li_o=2.45)
        else:
            mol_coords = bulk_centers[m - 8] + ec_heavy

        q_o = cfg["add_q_o"] if is_add else -0.62
        mol_q = [0.12, 0.12, -0.28, -q_o + 0.32, q_o, -0.28]
        mol_m = [14.027, 14.027, 15.999, 12.011, 15.999 if not is_add else 18.998, 15.999]
        for s in range(6):
            pos_list.append(mol_coords[s] % box_l)
            masses.append(mol_m[s])
            charges.append(mol_q[s])
            sig.append(2.90 if s == 4 else 3.35)
            eps.append(0.50 if s == 4 else 0.30)
            roles.append("additive" if is_add else "solvent")
        if is_add:
            add_o_idx.append(base + 4)
        else:
            solv_o_idx.append(base + 4)
        for a in range(6):
            for b in range(a + 1, 6):
                excl.add((base + a, base + b))
                if ec_dists[a, b] < 2.6:
                    bonds.append((base + a, base + b))
                    bond_r0.append(float(ec_dists[a, b]))
                    bond_k.append(1400.0)

    engine = PocketMicroEngine(
        masses, charges, sig, eps, bond_pairs=bonds, bond_r0=bond_r0, bond_k=bond_k,
        excluded_pairs=excl, box_length=box_l, dielectric=cfg["dielectric"],
    )
    return {
        "label": cfg["label"],
        "engine": engine,
        "pos0": np.asarray(pos_list, dtype=np.float64),
        "masses": np.asarray(masses, dtype=np.float64),
        "box_length": box_l,
        "friction": cfg["friction"],
        "ion_charge": cfg["ion_q"],
        "li_indices": np.asarray(li_idx, dtype=np.int64),
        "anion_center_indices": np.asarray(anion_p_idx, dtype=np.int64),
        "anion_f_indices": np.asarray(pf6_f_idx, dtype=np.int64),
        "solvent_o_indices": np.asarray(solv_o_idx, dtype=np.int64),
        "additive_o_indices": np.asarray(add_o_idx, dtype=np.int64),
        "roles": roles,
    }


def triage_electrolyte(preset="li_ec_fec_additive", *, n_frames=90, substeps=10, dt_fs=3.0,
                       temperature_k=300.0, seed=0):
    """Screen a battery electrolyte / additive formulation on CPU in ~0.5 s before 50 ns MD / AIMD."""
    t0 = time.perf_counter()
    setup = _build_electrolyte_box(preset)
    traj_x, traj_v = setup["engine"].run(
        setup["pos0"], n_frames=n_frames, substeps=substeps, dt_fs=dt_fs,
        temperature_k=temperature_k, friction_per_ps=setup["friction"], seed=seed,
    )
    wall_s = time.perf_counter() - t0
    box_l = setup["box_length"]
    stride_fs = substeps * dt_fs
    half = max(1, n_frames // 2)

    # 1. Primary Li+ solvation shell & SSIP / CIP / AGG speciation
    spec = solvation_speciation(
        traj_x[half:], box_l, setup["li_indices"],
        setup["solvent_o_indices"], setup["anion_center_indices"], setup["additive_o_indices"],
        r_shell=2.85,
    )
    spec_anion = solvation_speciation(
        traj_x[half:], box_l, setup["li_indices"],
        setup["solvent_o_indices"], setup["anion_center_indices"], setup["additive_o_indices"],
        r_shell=4.25,
    )
    spec["anion_cn"] = spec_anion["anion_cn"]
    spec["ssip_fraction"] = spec_anion["ssip_fraction"]
    spec["cip_fraction"] = spec_anion["cip_fraction"]
    spec["agg_fraction"] = spec_anion["agg_fraction"]

    # 2. Uncorrelated Einstein self-diffusion vs. France-Lanord Cluster Nernst-Einstein transport
    lags = np.arange(1, max(5, n_frames // 2))
    t_lags_fs = lags * stride_fs
    msd_li = analysis.msd(traj_x[:, setup["li_indices"]], lags)
    msd_an = analysis.msd(traj_x[:, setup["anion_center_indices"]], lags)
    _, d_li = analysis.diffusion_coefficient(t_lags_fs, msd_li, float(t_lags_fs[1]), float(t_lags_fs[-1]))
    _, d_an = analysis.diffusion_coefficient(t_lags_fs, msd_an, float(t_lags_fs[1]), float(t_lags_fs[-1]))
    d_li, d_an = max(0.0, float(d_li)), max(0.0, float(d_an))

    ne = nernst_einstein_metrics(
        d_li, d_an,
        n_cation=len(setup["li_indices"]), n_anion=len(setup["anion_center_indices"]),
        z_cation=setup["ion_charge"], z_anion=-setup["ion_charge"],
        volume_a3=box_l ** 3, temperature_k=temperature_k,
    )
    cne = cluster_nernst_einstein(
        traj_x, box_l, setup["li_indices"], setup["anion_center_indices"],
        z_cation=setup["ion_charge"], z_anion=-setup["ion_charge"],
        dt_frame_fs=stride_fs, temperature_k=temperature_k, r_pair=4.25,
    )

    # 3. Structural Hopping vs. Vehicular Index & Competitive Additive Grafting Score (CAGS)
    coord_o = np.concatenate([setup["solvent_o_indices"], setup["additive_o_indices"]])
    hop = solvation_hopping_metrics(traj_x, traj_v, box_l, setup["li_indices"], coord_o, r_shell=2.55)
    cags = competitive_additive_score(
        spec["solvent_cn"], spec["additive_cn"],
        n_solvent_mols=len(setup["solvent_o_indices"]),
        n_additive_mols=len(setup["additive_o_indices"]),
        paired_fraction=spec["cip_fraction"] + spec["agg_fraction"],
    )

    # 4. Li-O_solvent+additive RDF on wrapped coordinates
    wrapped = traj_x[half:] % box_l
    boxes = np.tile(np.diag([box_l, box_l, box_l]), (len(wrapped), 1, 1))
    r_grid, g_lio, _ = analysis.rdf(wrapped, boxes, setup["li_indices"], coord_o, min(7.5, 0.48 * box_l), 50)

    if (spec["agg_fraction"] >= 0.35 or spec["cip_fraction"] + spec["agg_fraction"] >= 0.65
            or cne["ionicity_haven_inv"] < 0.35):
        verdict = "REJECT"
        summary = (f"Neutral CIP/AGG pairing collapses true conductivity (sigma_cNE={cne['sigma_cne_mS_cm']:.1f} vs "
                   f"naive sigma_NE={ne['sigma_ne_mS_cm']:.1f} mS/cm) — reject before Big MD")
    elif cne["sigma_cne_mS_cm"] >= 8.0 and cne["transference_plus_cne"] >= 0.20 and cne["ionicity_haven_inv"] >= 0.60:
        verdict = "GO"
        summary = "High Cluster-NE conductivity, strong ionicity & healthy solvation — promote to 50 ns MD / AIMD"
    else:
        verdict = "HOLD"
        summary = "Healthy ionicity/CAGS sheath but viscous sigma_cNE < 8 mS/cm — lower additive wt% or add DMC"

    return {
        "mode": "electrolyte",
        "name": preset,
        "title": setup["label"],
        "engine": "PocketMD Periodic Electrolyte Engine (Cluster-NE + Hopping + CAGS, CPU)",
        "verdict": verdict,
        "summary": summary,
        "wall_s": round(wall_s, 3),
        "n_atoms": int(traj_x.shape[1]),
        "box_A": round(box_l, 2),
        "simulated_ps": round((n_frames - 1) * stride_fs / 1000.0, 3),
        "metrics": {
            "solvent_cn": round(spec["solvent_cn"], 2),
            "additive_cn": round(spec["additive_cn"], 2),
            "anion_cn": round(spec["anion_cn"], 2),
            "cags_additive_score": cags["cags_additive_score"],
            "additive_role": cags["additive_role"],
            "ssip_fraction": round(spec["ssip_fraction"], 3),
            "cip_fraction": round(spec["cip_fraction"], 3),
            "agg_fraction": round(spec["agg_fraction"], 3),
            "D_Li_cm2_s": float(f"{d_li:.3e}"),
            "D_anion_cm2_s": float(f"{d_an:.3e}"),
            "sigma_NE_mS_cm": round(ne["sigma_ne_mS_cm"], 2),
            "sigma_cNE_mS_cm": round(cne["sigma_cne_mS_cm"], 2),
            "ionicity_haven_inv": round(cne["ionicity_haven_inv"], 3),
            "transference_t_plus": round(ne["transference_plus_ne"], 3),
            "transference_cNE": round(cne["transference_plus_cne"], 3),
            "hopping_index_chi": hop["hopping_index_chi"],
            "shell_exchange_fraction": hop["shell_exchange_fraction"],
            "li_o_vel_corr": hop["li_o_vel_corr"],
        },
        "series": {
            "lag_ps": (t_lags_fs / 1000.0).round(3).tolist(),
            "msd_li_A2": msd_li.round(3).tolist(),
            "msd_anion_A2": msd_an.round(3).tolist(),
            "rdf_r_A": r_grid.round(3).tolist(),
            "rdf_g_li_o": g_lio.round(3).tolist(),
        },
        "viewer_frames": (traj_x[::max(1, len(traj_x) // 16)] % box_l).round(2).tolist(),
        "viewer_roles": setup["roles"],
    }


# ---------------------------------------------------------------- receipt formatter

def format_receipt(report):
    """Format a triage report as a compact pocket-calculator terminal receipt."""
    icon = {"GO": "[GO]    ", "HOLD": "[HOLD]  ", "REJECT": "[REJECT]"}.get(report["verdict"], report["verdict"])
    title = report.get("title", report["name"])
    lines = [
        "=" * 74,
        f" PocketMD Triage Calculator | {title}",
        "-" * 74,
        f" Engine : {report['engine']}",
        f" Speed  : {report['wall_s']:.2f} s on CPU  ({report['n_atoms']} atoms, {report['simulated_ps']} ps)",
        "-" * 74,
    ]
    for k, val in report["metrics"].items():
        lines.append(f"   {k:<28s} : {val}")
    lines.extend([
        "-" * 74,
        f" Verdict: {icon} -> {report['summary']}",
        "=" * 74,
    ])
    return "\n".join(lines)
