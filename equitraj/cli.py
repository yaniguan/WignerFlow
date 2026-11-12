"""Unified command-line interface (`wignerflow-pocket`) for PocketMD CPU triage & rollout.

Provides zero-boilerplate terminal commands modeled after ASE / MDTraj / OpenMM CLI ergonomics:
  - `wignerflow-pocket binding`     : Multi-replica OpenBPMD CompScore, PoseBusters vdW & 50-fs tau-RAMD pose triage
  - `wignerflow-pocket electrolyte` : Periodic Li+ solvation shell, Cluster-NE sigma_cNE, Haven ionicity & CAGS triage
  - `wignerflow-pocket checkpoint`  : Fast CPU Verlet rollout & dual epistemic audit of WignerFlow `.pt` checkpoints
  - `wignerflow-pocket rollout`     : Standalone pocket MD trajectory rollout from preset, `.xyz`, or `.pdb` (+ `.xml`)
  - `wignerflow-pocket doctor`      : Environment & optional-dependency check + sub-0.2s live CPU throughput benchmark
  - `wignerflow-pocket catalog`     : List public MD benchmark datasets with one-line import hints
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from equitraj.pocket import (
    CKPTS_DIR,
    DATASET_CATALOG,
    SYSTEMS_DIR,
    PocketMicroEngine,
    _DEFAULT_MASS,
    format_receipt,
    parse_openmm_xml,
    parse_pdb,
    posebusters_vdw_audit,
    triage_binding,
    triage_checkpoint,
    triage_electrolyte,
)
from equitraj.rollout import temperature

BINDING_PRESETS = (
    "aspirin_cox_stable",
    "aspirin_induced_fit",
    "aspirin_decoy_clash",
)

ELECTROLYTE_PRESETS = (
    "li_ec_fec_additive",
    "li_ec_base",
    "li_ec_vc_additive",
    "li_high_salt_agg",
)

ROLLOUT_PRESETS = (
    "ethanol",
    "aspirin",
    "ala2",
    "li_ec_pf6",
)

# Covalent radii in Å (Cordero et al., Dalton Trans. 2008) for automatic `.xyz` / `.pdb` bond inference
_COV_RADIUS_A = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Na": 1.66, "Mg": 1.41, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
}

# Built-in 9-atom equilibrium ethanol geometry (Å) so `wignerflow-pocket rollout` works anywhere with zero external files
_ETHANOL_ELEMENTS = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
_ETHANOL_POS_A = np.array([
    [-1.1867, -0.2472,  0.0000],
    [ 0.0048,  0.5960,  0.0000],
    [ 1.1879, -0.2071,  0.0000],
    [-1.2016, -0.8884,  0.8869],
    [-1.2016, -0.8884, -0.8869],
    [-2.0982,  0.3556,  0.0000],
    [-0.0112,  1.2462,  0.8837],
    [-0.0112,  1.2462, -0.8837],
    [ 1.9845,  0.3411,  0.0000],
], dtype=np.float64)


def parse_xyz(xyz_path: str | Path) -> dict:
    """Parse a standard single-frame `.xyz` coordinate file (Å) into elements, masses, and positions."""
    lines = [ln.rstrip() for ln in Path(xyz_path).read_text(encoding="utf-8").splitlines()]
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) < 3:
        raise ValueError(f"XYZ file {xyz_path} has too few lines")
    try:
        n_atoms = int(lines[0].strip().split()[0])
    except ValueError as exc:
        raise ValueError(f"First line of XYZ file {xyz_path} must be atom count") from exc
    if len(lines) < 2 + n_atoms:
        raise ValueError(f"XYZ file {xyz_path} declares {n_atoms} atoms but has fewer coordinate lines")

    elements, coords = [], []
    for row in lines[2:2 + n_atoms]:
        parts = row.split()
        if len(parts) < 4:
            raise ValueError(f"Malformed XYZ atom line in {xyz_path}: {row!r}")
        elem = "".join(ch for ch in parts[0] if ch.isalpha())[:2].capitalize()
        elements.append(elem)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])

    pos = np.asarray(coords, dtype=np.float64)
    masses = np.array([_DEFAULT_MASS.get(e, 12.011) for e in elements], dtype=np.float64)
    return {
        "elements": elements,
        "positions": pos,
        "masses": masses,
        "box": None,
    }


def write_xyz_trajectory(xyz_path: str | Path, elements: list[str], positions: np.ndarray, stride_fs: float) -> None:
    """Write a multi-frame `(T, N, 3)` trajectory in standard XYZ format (Å)."""
    out_p = Path(xyz_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    n_atoms = len(elements)
    chunks = []
    for frame_idx, frame in enumerate(positions):
        t_ps = (frame_idx * stride_fs) / 1000.0
        chunks.append(f"{n_atoms}")
        chunks.append(f"frame={frame_idx} time_ps={t_ps:.4f}")
        for elem, (x, y, z) in zip(elements, frame):
            chunks.append(f"{elem:<2s} {x:12.6f} {y:12.6f} {z:12.6f}")
    out_p.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def infer_harmonic_topology(positions: np.ndarray, elements: list[str], box_length: float | None = None) -> dict:
    """Infer covalent harmonic bonds, valence angles, and 1-2/1-3 exclusions from 3D coordinates."""
    pos = np.asarray(positions, dtype=np.float64)
    n = len(pos)
    rcov = np.array([_COV_RADIUS_A.get(e, 0.77) for e in elements], dtype=np.float64)

    bond_pairs, bond_r0, bond_k = [], [], []
    adj: list[list[int]] = [[] for _ in range(n)]
    excluded = set()

    for i in range(n):
        for j in range(i + 1, n):
            # Skip Li+ ionic contacts so cations diffuse freely instead of getting harmonically tethered
            if elements[i] in ("Li", "Na", "K", "Ar") or elements[j] in ("Li", "Na", "K", "Ar"):
                continue
            dr = pos[i] - pos[j]
            if box_length is not None:
                dr = dr - box_length * np.round(dr / box_length)
            d = float(np.linalg.norm(dr))
            cutoff = 1.24 * (rcov[i] + rcov[j])
            if 0.45 <= d <= cutoff:
                bond_pairs.append((i, j))
                bond_r0.append(d)
                is_xh = elements[i] == "H" or elements[j] == "H"
                bond_k.append(2200.0 if is_xh else 1800.0)
                adj[i].append(j)
                adj[j].append(i)
                excluded.add((i, j))

    ang_triplets, ang_th0, ang_k = [], [], []
    for center in range(n):
        nbrs = adj[center]
        for a_idx in range(len(nbrs)):
            for b_idx in range(a_idx + 1, len(nbrs)):
                i, k_atom = nbrs[a_idx], nbrs[b_idx]
                u = pos[i] - pos[center]
                v = pos[k_atom] - pos[center]
                if box_length is not None:
                    u = u - box_length * np.round(u / box_length)
                    v = v - box_length * np.round(v / box_length)
                nu = float(np.linalg.norm(u))
                nv = float(np.linalg.norm(v))
                if nu > 1e-6 and nv > 1e-6:
                    cos_th = float(np.clip(np.dot(u, v) / (nu * nv), -0.999999, 0.999999))
                    ang_triplets.append((i, center, k_atom))
                    ang_th0.append(float(np.arccos(cos_th)))
                    ang_k.append(380.0)
                    excluded.add((min(i, k_atom), max(i, k_atom)))

    return {
        "bond_pairs": np.asarray(bond_pairs, dtype=np.int64).reshape(-1, 2),
        "bond_r0": np.asarray(bond_r0, dtype=np.float64),
        "bond_k": np.asarray(bond_k, dtype=np.float64),
        "angle_triplets": np.asarray(ang_triplets, dtype=np.int64).reshape(-1, 3),
        "angle_theta0": np.asarray(ang_th0, dtype=np.float64),
        "angle_k": np.asarray(ang_k, dtype=np.float64),
        "excluded_pairs": excluded,
    }


def run_pocket_rollout(
    *,
    preset: str = "ethanol",
    xyz_path: str | None = None,
    pdb_path: str | None = None,
    xml_path: str | None = None,
    n_frames: int = 60,
    substeps: int = 8,
    dt_fs: float = 1.5,
    temperature_k: float = 300.0,
    friction_per_ps: float = 2.0,
    seed: int = 0,
    out_npz: str | None = None,
    out_xyz: str | None = None,
) -> dict:
    """Run a fast standalone CPU BAOAB trajectory rollout and return a structured PocketMD stability receipt."""
    t0 = time.perf_counter()
    box_l: float | None = None

    if xyz_path is not None:
        parsed = parse_xyz(xyz_path)
        name = Path(xyz_path).stem
        title = f"Custom XYZ Rollout ({Path(xyz_path).name})"
        pos0, elements, masses = parsed["positions"], parsed["elements"], parsed["masses"]
        topo = infer_harmonic_topology(pos0, elements, box_length=None)
        charges = np.zeros(len(pos0), dtype=np.float64)
        sigma_a = np.full(len(pos0), 3.15, dtype=np.float64)
        eps_kj = np.full(len(pos0), 0.25, dtype=np.float64)
    elif pdb_path is not None:
        pdb = parse_pdb(pdb_path)
        name = Path(pdb_path).stem
        title = f"Custom PDB Rollout ({Path(pdb_path).name})"
        pos0, elements, masses = pdb["positions"], pdb["elements"], pdb["masses"]
        box_l = float(pdb["box"][0, 0]) if pdb["box"] is not None else None
        if xml_path is not None and Path(xml_path).exists():
            ff = parse_openmm_xml(xml_path)
            masses = ff["masses"]
            charges, sigma_a, eps_kj = ff["charges"], ff["sigma_a"], ff["eps_kj"]
            topo = ff
        else:
            topo = infer_harmonic_topology(pos0, elements, box_length=box_l)
            charges = np.zeros(len(pos0), dtype=np.float64)
            sigma_a = np.full(len(pos0), 3.15, dtype=np.float64)
            eps_kj = np.full(len(pos0), 0.25, dtype=np.float64)
    else:
        preset_map = {
            "aspirin": ("Aspirin (OpenFF-2.2.1 Vacuum)", SYSTEMS_DIR / "aspirin.pdb", SYSTEMS_DIR / "aspirin.xml"),
            "ala2": ("Alanine Dipeptide (AMBER Vacuum)", SYSTEMS_DIR / "ala2.pdb", SYSTEMS_DIR / "ala2.xml"),
            "li_ec_pf6": ("LiPF6 in EC Periodic Cell (ECC q=0.8)", SYSTEMS_DIR / "li_ec_pf6_q08.pdb", SYSTEMS_DIR / "li_ec_pf6_q08.xml"),
        }
        if preset == "ethanol":
            name = "ethanol"
            title = "Ethanol CH3CH2OH (9-Atom Pocket Preset)"
            elements = list(_ETHANOL_ELEMENTS)
            pos0 = _ETHANOL_POS_A.copy()
            masses = np.array([_DEFAULT_MASS[e] for e in elements], dtype=np.float64)
            topo = infer_harmonic_topology(pos0, elements, box_length=None)
            charges = np.array([-0.18, 0.14, -0.68, 0.06, 0.06, 0.06, 0.06, 0.06, 0.42], dtype=np.float64)
            sigma_a = np.array([3.40, 3.40, 3.05, 2.50, 2.50, 2.50, 2.50, 2.50, 1.20], dtype=np.float64)
            eps_kj = np.array([0.45, 0.45, 0.70, 0.06, 0.06, 0.06, 0.06, 0.06, 0.02], dtype=np.float64)
        elif preset in preset_map:
            title, p_pdb, p_xml = preset_map[preset]
            name = preset
            pdb = parse_pdb(p_pdb)
            ff = parse_openmm_xml(p_xml)
            pos0, elements, masses = pdb["positions"], pdb["elements"], ff["masses"]
            box_l = float(pdb["box"][0, 0]) if pdb["box"] is not None else None
            charges, sigma_a, eps_kj = ff["charges"], ff["sigma_a"], ff["eps_kj"]
            topo = ff
        else:
            raise ValueError(f"Unknown rollout preset {preset!r}; choose from {ROLLOUT_PRESETS}")

    engine = PocketMicroEngine(
        masses, charges, sigma_a, eps_kj,
        bond_pairs=topo["bond_pairs"], bond_r0=topo["bond_r0"], bond_k=topo["bond_k"],
        angle_triplets=topo["angle_triplets"], angle_theta0=topo["angle_theta0"], angle_k=topo["angle_k"],
        excluded_pairs=topo["excluded_pairs"], box_length=box_l, dielectric=1.5 if box_l is None else 2.0,
    )
    traj_x, traj_v = engine.run(
        pos0, n_frames=n_frames, substeps=substeps, dt_fs=dt_fs,
        temperature_k=temperature_k, friction_per_ps=friction_per_ps, seed=seed,
    )
    wall_s = time.perf_counter() - t0

    total_steps = max(1, (n_frames - 1) * substeps)
    sim_ps = (total_steps * dt_fs) / 1000.0
    stride_fs = substeps * dt_fs
    half = max(1, n_frames // 2)

    m_t = torch.from_numpy(masses)
    temp_series = temperature(torch.from_numpy(traj_v), m_t).numpy()
    t_mean = float(temp_series[half:].mean())
    t_rel_err = abs(t_mean - temperature_k) / max(temperature_k, 1e-6)

    vdw_terminal = posebusters_vdw_audit(
        traj_x[-1], elements, excluded_pairs=topo["excluded_pairs"], box_length=box_l
    )

    if len(topo["bond_pairs"]) > 0:
        bi, bj = topo["bond_pairs"][:, 0], topo["bond_pairs"][:, 1]
        dr_b = traj_x[:, bi] - traj_x[:, bj]
        if box_l is not None:
            dr_b = dr_b - box_l * np.round(dr_b / box_l)
        d_b = np.linalg.norm(dr_b, axis=-1)
        bond_rms_drift = float(np.sqrt(np.mean((d_b - topo["bond_r0"][None, :]) ** 2)))
    else:
        bond_rms_drift = 0.0

    m_tot = float(masses.sum())
    v_com = (traj_v * masses[None, :, None]).sum(axis=1) / max(m_tot, 1e-12)
    k_com = 0.5 * m_tot * (v_com ** 2).sum(axis=-1)
    k_tot = 0.5 * (masses[None, :] * (traj_v ** 2).sum(axis=-1)).sum(axis=-1)
    com_leakage = float((k_com / np.maximum(k_tot, 1e-15)).max())

    # Mass-weighted radius of gyration across terminal half (for non-periodic molecules)
    com_pos = (traj_x[half:] * masses[None, :, None]).sum(axis=1, keepdims=True) / max(m_tot, 1e-12)
    rel_pos = traj_x[half:] - com_pos
    rg_a = float(np.sqrt(((masses[None, :] * (rel_pos ** 2).sum(axis=-1)).sum(axis=-1) / max(m_tot, 1e-12)).mean()))

    if out_npz is not None:
        out_p = Path(out_npz)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_p,
            positions_A=traj_x,
            velocities_A_fs=traj_v,
            elements=np.asarray(elements),
            masses_amu=masses,
            dt_frame_fs=np.array(stride_fs),
            box_A=np.array(-1.0 if box_l is None else box_l),
        )
    if out_xyz is not None:
        write_xyz_trajectory(out_xyz, elements, traj_x, stride_fs=stride_fs)

    if vdw_terminal["min_vdw_ratio"] < 0.65 or bond_rms_drift > 0.18 or not np.isfinite(t_mean):
        verdict = "REJECT"
        summary = "Unphysical steric clash or bond blowup detected during pocket rollout"
    elif t_rel_err <= 0.35 and bond_rms_drift <= 0.08 and com_leakage < 1e-6:
        verdict = "GO"
        summary = "Thermally equilibrated, zero-COM-leakage CPU trajectory ready for downstream screening"
    else:
        verdict = "HOLD"
        summary = "Mild thermal/bond transient — increase frames or lower dt_fs for stiff modes"

    ns_per_day = (sim_ps * 1e-3) / max(wall_s, 1e-6) * 86400.0
    metrics = {
        "n_bonds": int(len(topo["bond_pairs"])),
        "n_angles": int(len(topo["angle_triplets"])),
        "total_md_steps": int(total_steps),
        "ns_per_day_cpu": round(ns_per_day, 2),
        "T_target_K": round(float(temperature_k), 2),
        "T_mean_K": round(t_mean, 2),
        "T_rel_err": round(t_rel_err, 4),
        "bond_rms_drift_A": round(bond_rms_drift, 4),
        "min_vdw_ratio": round(vdw_terminal["min_vdw_ratio"], 3),
        "vdw_clashes": int(vdw_terminal["n_vdw_clashes"]),
        "radius_of_gyration_A": round(rg_a, 3),
        "com_kinetic_leakage_ratio": float(f"{com_leakage:.2e}"),
    }
    if out_npz is not None:
        metrics["saved_npz"] = str(out_npz)
    if out_xyz is not None:
        metrics["saved_xyz"] = str(out_xyz)

    return {
        "mode": "rollout",
        "name": name,
        "title": title,
        "engine": f"PocketMD BAOAB-Langevin CPU Engine (dt={dt_fs} fs, {substeps}x stride)",
        "verdict": verdict,
        "summary": summary,
        "wall_s": round(wall_s, 3),
        "n_atoms": int(traj_x.shape[1]),
        "simulated_ps": round(sim_ps, 3),
        "metrics": metrics,
    }


def _pkg_version(pkg_name: str, import_name: str | None = None) -> str:
    mod_name = import_name or pkg_name
    if importlib.util.find_spec(mod_name) is None:
        return "not-installed"
    try:
        return importlib.metadata.version(pkg_name)
    except importlib.metadata.PackageNotFoundError:
        mod = __import__(mod_name)
        return str(getattr(mod, "__version__", "installed"))


def run_doctor() -> dict:
    """Inspect Python/PyTorch/optional MD dependencies and execute a live <0.2s CPU throughput check."""
    t0 = time.perf_counter()
    # Live 25-frame x 8-substep = 192-step BAOAB micro-benchmark on 9-atom ethanol
    bench = run_pocket_rollout(preset="ethanol", n_frames=25, substeps=8, dt_fs=1.5, seed=7)
    bench_s = max(time.perf_counter() - t0, 1e-6)
    steps = int(bench["metrics"]["total_md_steps"])
    steps_per_s = steps / bench_s
    atom_steps_per_s = (steps * bench["n_atoms"]) / bench_s

    ckpts_present = sorted(p.name for p in CKPTS_DIR.glob("*.pt")) if CKPTS_DIR.exists() else []
    systems_present = sorted(p.name for p in SYSTEMS_DIR.glob("*.pdb")) if SYSTEMS_DIR.exists() else []

    core_ok = all(
        _pkg_version(p, imp) != "not-installed"
        for p, imp in (("torch", "torch"), ("numpy", "numpy"), ("scipy", "scipy"), ("e3nn", "e3nn"))
    )
    verdict = "GO" if (core_ok and bench["verdict"] in ("GO", "HOLD")) else "REJECT"
    summary = (
        f"CPU pocket engine ready ({steps_per_s:,.0f} steps/s, {len(systems_present)} built-in systems, "
        f"{len(ckpts_present)} checkpoints)"
        if verdict == "GO"
        else "Missing core runtime dependencies or failed live sanity rollout"
    )

    return {
        "mode": "doctor",
        "name": "wignerflow-pocket-doctor",
        "title": "WignerFlow PocketMD Environment & CPU Throughput Doctor",
        "engine": f"Python {platform.python_version()} | PyTorch {torch.__version__} ({torch.get_num_threads()} CPU threads)",
        "verdict": verdict,
        "summary": summary,
        "wall_s": round(bench_s, 3),
        "n_atoms": bench["n_atoms"],
        "simulated_ps": bench["simulated_ps"],
        "metrics": {
            "os_platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "cpu_logical_cores": int(os.cpu_count() or 1),
            "torch_num_threads": int(torch.get_num_threads()),
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": _pkg_version("torch", "torch"),
            "e3nn_version": _pkg_version("e3nn", "e3nn"),
            "numpy_version": _pkg_version("numpy", "numpy"),
            "scipy_version": _pkg_version("scipy", "scipy"),
            "openmm_optional": _pkg_version("openmm", "openmm"),
            "rdkit_optional": _pkg_version("rdkit", "rdkit"),
            "mdtraj_optional": _pkg_version("mdtraj", "mdtraj"),
            "builtin_systems_count": len(systems_present),
            "builtin_checkpoints_count": len(ckpts_present),
            "live_bench_steps_per_s": round(steps_per_s, 1),
            "live_bench_atom_steps_per_s": round(atom_steps_per_s, 1),
            "live_bench_ns_per_day": bench["metrics"]["ns_per_day_cpu"],
        },
    }


def _emit_report(report: dict, *, as_json: bool = False, out_path: str | None = None) -> int:
    """Print human-readable terminal receipt or JSON payload and optionally write to `--out`."""
    rendered = json.dumps(report, indent=2) if as_json else format_receipt(report)
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(rendered)
    if out_path is not None:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(report, indent=2) if (as_json or out_p.suffix.lower() == ".json") else rendered
        out_p.write_text(payload + "\n", encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="wignerflow-pocket",
        description=(
            "WignerFlow PocketMD: quick CPU molecular dynamics triage & rollout calculator "
            "for drug binding poses, battery electrolytes, and equivariant checkpoints."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # 1. binding
    p_bind = sub.add_parser(
        "binding",
        help="Triage a protein-ligand pose via multi-replica OpenBPMD CompScore, PoseBusters vdW & 50-fs tau-RAMD",
    )
    p_bind.add_argument("--preset", default="aspirin_cox_stable", choices=BINDING_PRESETS,
                        help="Built-in COX-1 aspirin pose preset (ignored when --pdb is supplied)")
    p_bind.add_argument("--pdb", default=None, help="Path to custom complex PDB file (ligand + pocket)")
    p_bind.add_argument("--xml", default=None, help="Optional OpenMM XML force-field file matching --pdb")
    p_bind.add_argument("--replicas", type=int, default=3, help="Short thermal relaxation replicas (default: 3)")
    p_bind.add_argument("--frames", type=int, default=70, help="Saved trajectory frames per replica (default: 70)")
    p_bind.add_argument("--substeps", type=int, default=8, help="BAOAB substeps per frame (default: 8)")
    p_bind.add_argument("--dt-fs", type=float, default=1.5, help="Integration timestep in fs (default: 1.5)")
    p_bind.add_argument("--temp-k", type=float, default=300.0, help="Bath temperature in K (default: 300.0)")
    p_bind.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0)")
    p_bind.add_argument("--list-presets", action="store_true", help="List built-in binding presets and exit")
    p_bind.add_argument("--json", action="store_true", help="Emit machine-readable JSON report to stdout")
    p_bind.add_argument("--out", default=None, help="Optional file path to save receipt or .json report")

    # 2. electrolyte
    p_elec = sub.add_parser(
        "electrolyte",
        help="Triage a Li+ battery electrolyte/additive box via Cluster-NE sigma_cNE, Haven ionicity & CAGS",
    )
    p_elec.add_argument("--preset", default="li_ec_fec_additive", choices=ELECTROLYTE_PRESETS,
                        help="Built-in periodic LiPF6 carbonate/additive formulation preset")
    p_elec.add_argument("--frames", type=int, default=90, help="Saved trajectory frames (default: 90)")
    p_elec.add_argument("--substeps", type=int, default=10, help="BAOAB substeps per frame (default: 10)")
    p_elec.add_argument("--dt-fs", type=float, default=3.0, help="Integration timestep in fs (default: 3.0)")
    p_elec.add_argument("--temp-k", type=float, default=300.0, help="Bath temperature in K (default: 300.0)")
    p_elec.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0)")
    p_elec.add_argument("--list-presets", action="store_true", help="List built-in electrolyte presets and exit")
    p_elec.add_argument("--json", action="store_true", help="Emit machine-readable JSON report to stdout")
    p_elec.add_argument("--out", default=None, help="Optional file path to save receipt or .json report")

    # 3. checkpoint
    p_ckpt = sub.add_parser(
        "checkpoint",
        help="Audit a WignerFlow `.pt` neural propagator checkpoint on CPU (Fu-2023 + Dual Epistemic Gate)",
    )
    p_ckpt.add_argument("--ckpt", default=None, help="Path to `.pt` checkpoint (defaults to built-in Argon flow ckpt)")
    p_ckpt.add_argument("--steps", type=int, default=24, help="Coarse-stride rollout steps (default: 24)")
    p_ckpt.add_argument("--sampling-steps", type=int, default=3, help="Flow ODE integration steps (default: 3)")
    p_ckpt.add_argument("--neighbor", default="verlet", choices=("verlet", "dense", "cell"),
                        help="Neighbor graph backend (default: verlet)")
    p_ckpt.add_argument("--thermostat", action="store_true", help="Enable CSVR thermostat during rollout")
    p_ckpt.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    p_ckpt.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0)")
    p_ckpt.add_argument("--json", action="store_true", help="Emit machine-readable JSON report to stdout")
    p_ckpt.add_argument("--out", default=None, help="Optional file path to save receipt or .json report")

    # 4. rollout
    p_roll = sub.add_parser(
        "rollout",
        help="Run a fast standalone CPU trajectory rollout from a preset, `.xyz`, or `.pdb` (+ optional `.xml`)",
    )
    p_roll.add_argument("--preset", default="ethanol", choices=ROLLOUT_PRESETS,
                        help="Built-in molecular/periodic system preset when --xyz/--pdb are omitted")
    p_roll.add_argument("--xyz", default=None, help="Custom single-frame `.xyz` input file (Å)")
    p_roll.add_argument("--pdb", default=None, help="Custom `.pdb` structure file (Å)")
    p_roll.add_argument("--xml", default=None, help="Optional OpenMM `.xml` force field matching --pdb")
    p_roll.add_argument("--frames", type=int, default=60, help="Saved trajectory frames (default: 60)")
    p_roll.add_argument("--substeps", type=int, default=8, help="BAOAB substeps per frame (default: 8)")
    p_roll.add_argument("--dt-fs", type=float, default=1.5, help="Timestep in fs (default: 1.5)")
    p_roll.add_argument("--temp-k", type=float, default=300.0, help="Bath temperature in K (default: 300.0)")
    p_roll.add_argument("--friction", type=float, default=2.0, help="Langevin friction in ps^-1 (default: 2.0)")
    p_roll.add_argument("--seed", type=int, default=0, help="Deterministic RNG seed (default: 0)")
    p_roll.add_argument("--out-npz", default=None, help="Optional `.npz` path to export trajectory coordinates & velocities")
    p_roll.add_argument("--out-xyz", default=None, help="Optional multi-frame `.xyz` path for VMD/Ovito/PyMOL")
    p_roll.add_argument("--list-presets", action="store_true", help="List built-in rollout presets and exit")
    p_roll.add_argument("--json", action="store_true", help="Emit machine-readable JSON report to stdout")
    p_roll.add_argument("--out", default=None, help="Optional file path to save receipt or .json report")

    # 5. doctor
    p_doc = sub.add_parser(
        "doctor",
        help="Verify Python/PyTorch/optional MD packages and benchmark live CPU rollout speed",
    )
    p_doc.add_argument("--json", action="store_true", help="Emit machine-readable JSON report to stdout")
    p_doc.add_argument("--out", default=None, help="Optional file path to save receipt or .json report")

    # 6. catalog
    p_cat = sub.add_parser(
        "catalog",
        help="List curated public drug-binding, electrolyte & coarse-stride MD datasets",
    )
    p_cat.add_argument("--domain", default="all", choices=("all", "binding", "electrolyte", "checkpoint"),
                       help="Filter datasets by application domain")
    p_cat.add_argument("--json", action="store_true", help="Emit catalog as JSON")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "binding":
        if args.list_presets:
            print("\n".join(BINDING_PRESETS))
            return 0
        report = triage_binding(
            preset=args.preset,
            pdb_path=args.pdb,
            xml_path=args.xml,
            n_frames=args.frames,
            substeps=args.substeps,
            dt_fs=args.dt_fs,
            temperature_k=args.temp_k,
            n_replicas=args.replicas,
            seed=args.seed,
        )
        return _emit_report(report, as_json=args.json, out_path=args.out)

    if args.command == "electrolyte":
        if args.list_presets:
            print("\n".join(ELECTROLYTE_PRESETS))
            return 0
        report = triage_electrolyte(
            preset=args.preset,
            n_frames=args.frames,
            substeps=args.substeps,
            dt_fs=args.dt_fs,
            temperature_k=args.temp_k,
            seed=args.seed,
        )
        return _emit_report(report, as_json=args.json, out_path=args.out)

    if args.command == "checkpoint":
        report = triage_checkpoint(
            ckpt_path=args.ckpt,
            steps=args.steps,
            sampling_steps=args.sampling_steps,
            neighbor=args.neighbor,
            thermostat_on=args.thermostat,
            seed=args.seed,
            device=args.device,
        )
        return _emit_report(report, as_json=args.json, out_path=args.out)

    if args.command == "rollout":
        if args.list_presets:
            print("\n".join(ROLLOUT_PRESETS))
            return 0
        report = run_pocket_rollout(
            preset=args.preset,
            xyz_path=args.xyz,
            pdb_path=args.pdb,
            xml_path=args.xml,
            n_frames=args.frames,
            substeps=args.substeps,
            dt_fs=args.dt_fs,
            temperature_k=args.temp_k,
            friction_per_ps=args.friction,
            seed=args.seed,
            out_npz=args.out_npz,
            out_xyz=args.out_xyz,
        )
        return _emit_report(report, as_json=args.json, out_path=args.out)

    if args.command == "doctor":
        report = run_doctor()
        return _emit_report(report, as_json=args.json, out_path=args.out)

    if args.command == "catalog":
        items = [
            d for d in DATASET_CATALOG
            if args.domain == "all" or d["domain"] == args.domain
        ]
        if args.json:
            print(json.dumps(items, indent=2))
        else:
            for item in items:
                print(f"[{item['domain'].upper():<11s}] {item['id']} — {item['title']}")
                print(f"              URL   : {item['url']}")
                print(f"              Import: {item['import_hint']}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
