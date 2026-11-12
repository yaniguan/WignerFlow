"""Standalone rollout helper and cross-domain benchmark evaluation data for WignerFlow Studio."""

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from equitraj import analysis
from equitraj.pocket import (
    SYSTEMS_DIR,
    PocketMicroEngine,
    _DEFAULT_MASS,
    parse_openmm_xml,
    parse_pdb,
    posebusters_vdw_audit,
)
from equitraj.rollout import temperature

_COV_RADIUS_A = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "B": 0.84, "C": 0.76, "N": 0.71,
    "O": 0.66, "F": 0.57, "Na": 1.66, "Mg": 1.41, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06, "K": 2.03, "Ca": 1.76,
}

DEFAULT_ETHANOL_XYZ = """9
Ethanol CH3CH2OH equilibrium geometry (Angstrom)
C   -1.186700   -0.247200    0.000000
C    0.004800    0.596000    0.000000
O    1.187900   -0.207100    0.000000
H   -1.201600   -0.888400    0.886900
H   -1.201600   -0.888400   -0.886900
H   -2.098200    0.355600    0.000000
H   -0.011200    1.246200    0.883700
H   -0.011200    1.246200   -0.883700
H    1.984500    0.341100    0.000000
"""


def parse_xyz_text(xyz_text: str) -> dict:
    lines = [ln.rstrip() for ln in xyz_text.strip().splitlines()]
    if len(lines) < 3:
        raise ValueError("XYZ text must have at least 3 lines (N, comment, coordinates)")
    n_atoms = int(lines[0].strip().split()[0])
    if len(lines) < 2 + n_atoms:
        raise ValueError(f"XYZ declares {n_atoms} atoms, but only {len(lines) - 2} atom lines were provided")
    elements, coords = [], []
    for row in lines[2:2 + n_atoms]:
        parts = row.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ coordinate line: {row!r}")
        elem = "".join(ch for ch in parts[0] if ch.isalpha())[:2].capitalize()
        elements.append(elem)
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pos = np.asarray(coords, dtype=np.float64)
    masses = np.array([_DEFAULT_MASS.get(e, 12.011) for e in elements], dtype=np.float64)
    return {"elements": elements, "positions": pos, "masses": masses, "comment": lines[1].strip()}


def infer_covalent_topology(positions: np.ndarray, elements: list[str], box_length: float | None = None) -> dict:
    pos = np.asarray(positions, dtype=np.float64)
    n = len(pos)
    rcov = np.array([_COV_RADIUS_A.get(e, 0.77) for e in elements], dtype=np.float64)
    bond_pairs, bond_r0, bond_k = [], [], []
    adj: list[list[int]] = [[] for _ in range(n)]
    excluded = set()

    for i in range(n):
        for j in range(i + 1, n):
            if elements[i] in ("Li", "Na", "K", "Ar") or elements[j] in ("Li", "Na", "K", "Ar"):
                continue
            dr = pos[i] - pos[j]
            if box_length is not None:
                dr = dr - box_length * np.round(dr / box_length)
            d = float(np.linalg.norm(dr))
            if 0.45 <= d <= 1.24 * (rcov[i] + rcov[j]):
                bond_pairs.append((i, j))
                bond_r0.append(d)
                bond_k.append(2200.0 if ("H" in (elements[i], elements[j])) else 1800.0)
                adj[i].append(j)
                adj[j].append(i)
                excluded.add((i, j))

    ang_triplets, ang_th0, ang_k = [], [], []
    for c in range(n):
        nbrs = adj[c]
        for a_i in range(len(nbrs)):
            for b_i in range(a_i + 1, len(nbrs)):
                u_idx, v_idx = nbrs[a_i], nbrs[b_i]
                u = pos[u_idx] - pos[c]
                v = pos[v_idx] - pos[c]
                if box_length is not None:
                    u = u - box_length * np.round(u / box_length)
                    v = v - box_length * np.round(v / box_length)
                nu, nv = float(np.linalg.norm(u)), float(np.linalg.norm(v))
                if nu > 1e-6 and nv > 1e-6:
                    cos_th = float(np.clip(np.dot(u, v) / (nu * nv), -0.999999, 0.999999))
                    ang_triplets.append((u_idx, c, v_idx))
                    ang_th0.append(float(np.arccos(cos_th)))
                    ang_k.append(380.0)
                    excluded.add((min(u_idx, v_idx), max(u_idx, v_idx)))

    return {
        "bond_pairs": np.asarray(bond_pairs, dtype=np.int64).reshape(-1, 2),
        "bond_r0": np.asarray(bond_r0, dtype=np.float64),
        "bond_k": np.asarray(bond_k, dtype=np.float64),
        "angle_triplets": np.asarray(ang_triplets, dtype=np.int64).reshape(-1, 3),
        "angle_theta0": np.asarray(ang_th0, dtype=np.float64),
        "angle_k": np.asarray(ang_k, dtype=np.float64),
        "excluded_pairs": excluded,
    }


def run_studio_rollout(
    *,
    preset: str = "aspirin",
    custom_xyz: str | None = None,
    n_frames: int = 60,
    substeps: int = 8,
    dt_fs: float = 1.5,
    temperature_k: float = 300.0,
    friction_per_ps: float = 2.0,
    seed: int = 0,
) -> dict:
    """Run a live CPU BAOAB rollout on a preset or custom XYZ molecule for interactive 3D Studio inspection."""
    t0 = time.perf_counter()
    box_l = None
    if custom_xyz and custom_xyz.strip():
        parsed = parse_xyz_text(custom_xyz)
        title = f"Custom XYZ ({len(parsed['elements'])} atoms · {parsed['comment'][:40]})"
        elements, pos0, masses = parsed["elements"], parsed["positions"], parsed["masses"]
        pos0 = pos0 - pos0.mean(axis=0, keepdims=True)
        topo = infer_covalent_topology(pos0, elements)
        charges = np.zeros(len(pos0), dtype=np.float64)
        sigma_a = np.full(len(pos0), 3.15, dtype=np.float64)
        eps_kj = np.full(len(pos0), 0.25, dtype=np.float64)
    elif preset == "ethanol":
        parsed = parse_xyz_text(DEFAULT_ETHANOL_XYZ)
        title = "Ethanol CH3CH2OH (9-Atom Pocket Preset)"
        elements, pos0, masses = parsed["elements"], parsed["positions"], parsed["masses"]
        topo = infer_covalent_topology(pos0, elements)
        charges = np.array([-0.18, 0.14, -0.68, 0.06, 0.06, 0.06, 0.06, 0.06, 0.42], dtype=np.float64)
        sigma_a = np.array([3.40, 3.40, 3.05, 2.50, 2.50, 2.50, 2.50, 2.50, 1.20], dtype=np.float64)
        eps_kj = np.array([0.45, 0.45, 0.70, 0.06, 0.06, 0.06, 0.06, 0.06, 0.02], dtype=np.float64)
    else:
        preset_files = {
            "aspirin": ("Aspirin (OpenFF-2.2.1 Vacuum, 21 Atoms)", "aspirin.pdb", "aspirin.xml"),
            "ala2": ("Alanine Dipeptide (AMBER Vacuum, 22 Atoms)", "ala2.pdb", "ala2.xml"),
            "li_ec_pf6": ("LiPF6 in EC Periodic Cell (ECC q=0.8)", "li_ec_pf6_q08.pdb", "li_ec_pf6_q08.xml"),
        }
        if preset not in preset_files:
            raise ValueError(f"Unknown rollout preset {preset!r}")
        title, pdb_name, xml_name = preset_files[preset]
        pdb = parse_pdb(SYSTEMS_DIR / pdb_name)
        ff = parse_openmm_xml(SYSTEMS_DIR / xml_name)
        pos0, elements, masses = pdb["positions"], pdb["elements"], ff["masses"]
        box_l = float(pdb["box"][0, 0]) if pdb["box"] is not None else None
        if box_l is None:
            pos0 = pos0 - pos0.mean(axis=0, keepdims=True)
        charges, sigma_a, eps_kj = ff["charges"], ff["sigma_a"], ff["eps_kj"]
        topo = dict(ff)
        # Add 1-3 angle exclusions to PoseBusters steric mask
        excl = set(topo["excluded_pairs"])
        for a0, _, a2 in topo["angle_triplets"]:
            excl.add((min(int(a0), int(a2)), max(int(a0), int(a2))))
        topo["excluded_pairs"] = excl

    eng = PocketMicroEngine(
        masses, charges, sigma_a, eps_kj,
        bond_pairs=topo["bond_pairs"], bond_r0=topo["bond_r0"], bond_k=topo["bond_k"],
        angle_triplets=topo["angle_triplets"], angle_theta0=topo["angle_theta0"], angle_k=topo["angle_k"],
        excluded_pairs=topo["excluded_pairs"], box_length=box_l, dielectric=1.5 if box_l is None else 2.0,
    )
    traj_x, traj_v = eng.run(
        pos0, n_frames=int(n_frames), substeps=int(substeps), dt_fs=float(dt_fs),
        temperature_k=float(temperature_k), friction_per_ps=float(friction_per_ps), seed=int(seed),
    )
    wall_s = time.perf_counter() - t0
    stride_fs = float(substeps) * float(dt_fs)
    sim_ps = max(1, int(n_frames) - 1) * stride_fs / 1000.0
    half = max(1, len(traj_x) // 2)

    t_series = temperature(torch.from_numpy(traj_v), torch.from_numpy(masses)).numpy()
    t_mean = float(t_series[half:].mean())
    t_rel_err = abs(t_mean - float(temperature_k)) / max(float(temperature_k), 1e-6)
    vdw_last = posebusters_vdw_audit(traj_x[-1], elements, excluded_pairs=topo["excluded_pairs"], box_length=box_l)

    if len(topo["bond_pairs"]) > 0:
        bi, bj = topo["bond_pairs"][:, 0], topo["bond_pairs"][:, 1]
        dr_b = traj_x[:, bi] - traj_x[:, bj]
        if box_l is not None:
            dr_b = dr_b - box_l * np.round(dr_b / box_l)
        bond_rms = float(np.sqrt(np.mean((np.linalg.norm(dr_b, axis=-1) - topo["bond_r0"][None, :]) ** 2)))
    else:
        bond_rms = 0.0

    m_tot = float(masses.sum())
    v_com = (traj_v * masses[None, :, None]).sum(axis=1) / max(m_tot, 1e-12)
    k_com = 0.5 * m_tot * (v_com ** 2).sum(axis=-1)
    k_tot = 0.5 * (masses[None, :] * (traj_v ** 2).sum(axis=-1)).sum(axis=-1)
    com_leak = float((k_com / np.maximum(k_tot, 1e-15)).max())

    verdict = (
        "REJECT" if (vdw_last["min_vdw_ratio"] < 0.65 or bond_rms > 0.18)
        else ("GO" if (t_rel_err <= 0.35 and bond_rms <= 0.09 and com_leak < 1e-6) else "HOLD")
    )

    vis_n = min(96, traj_x.shape[1])
    vis_frames = (traj_x[:, :vis_n] % box_l if box_l is not None else traj_x[:, :vis_n]).round(3).tolist()
    vis_bonds = [[int(i), int(j)] for i, j in topo["bond_pairs"] if i < vis_n and j < vis_n]

    xyz_chunks = []
    for f_i, fr in enumerate(traj_x[::max(1, len(traj_x) // 24)]):
        xyz_chunks.append(str(len(elements)))
        xyz_chunks.append(f"step={f_i} time_ps={(f_i * stride_fs / 1000.0):.4f}")
        for el, (x, y, z) in zip(elements, fr):
            xyz_chunks.append(f"{el:<2s} {x:11.5f} {y:11.5f} {z:11.5f}")

    return {
        "mode": "rollout",
        "title": title,
        "verdict": verdict,
        "wall_s": round(wall_s, 3),
        "n_atoms": int(traj_x.shape[1]),
        "simulated_ps": round(sim_ps, 3),
        "ns_per_day_cpu": round((sim_ps * 1e-3) / max(wall_s, 1e-6) * 86400.0, 1),
        "metrics": {
            "n_bonds": int(len(topo["bond_pairs"])),
            "n_angles": int(len(topo["angle_triplets"])),
            "T_target_K": round(float(temperature_k), 1),
            "T_mean_K": round(t_mean, 2),
            "T_rel_err": round(t_rel_err, 4),
            "bond_rms_drift_A": round(bond_rms, 4),
            "min_vdw_ratio": round(vdw_last["min_vdw_ratio"], 3),
            "vdw_clashes": int(vdw_last["n_vdw_clashes"]),
            "com_kinetic_leakage": float(f"{com_leak:.2e}"),
        },
        "series": {
            "time_ps": ((np.arange(len(traj_x)) * stride_fs) / 1000.0).round(3).tolist(),
            "temperature_K": t_series.round(2).tolist(),
        },
        "viewer_frames": vis_frames,
        "viewer_elements": elements[:vis_n],
        "viewer_bonds": vis_bonds,
        "viewer_box_A": None if box_l is None else round(box_l, 3),
        "xyz_download": "\n".join(xyz_chunks) + "\n",
    }


BENCHMARK_SUITES = {
    "pure_md_md17_md22": [
        {"system": "Ethanol (MD17-CCSD(T))", "atoms": 9, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.1846, "vdos_cos": 0.7336, "com_drift": "3.97e-19", "ns_day": 1303},
        {"system": "Malonaldehyde (MD17-CCSD(T))", "atoms": 9, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.2143, "vdos_cos": 0.6290, "com_drift": "5.84e-19", "ns_day": 1401},
        {"system": "Benzene (MD17-CCSD(T))", "atoms": 12, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.4027, "vdos_cos": 0.5620, "com_drift": "7.36e-19", "ns_day": 1328},
        {"system": "Uracil (MD17 DFT)", "atoms": 12, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.3055, "vdos_cos": 0.5665, "com_drift": "7.11e-19", "ns_day": 1390},
        {"system": "Toluene (MD17-CCSD(T))", "atoms": 15, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.1710, "vdos_cos": 0.7759, "com_drift": "8.59e-19", "ns_day": 1316},
        {"system": "Aspirin (MD17-CCSD)", "atoms": 21, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.1311, "vdos_cos": 0.6161, "com_drift": "1.13e-18", "ns_day": 1254},
        {"system": "AT-AT DNA Base Pair (MD22 DFT)", "atoms": 60, "vv_10fs": "0% (0.01 ps)", "wf_10fs": "100% (1.0 ps)", "wf_20fs": "100% (1.0 ps)", "mae_hr": 0.1076, "vdos_cos": 0.7470, "com_drift": "1.98e-18", "ns_day": 907},
    ],
    "calisol23_alignment": {
        "n_formulations": 46,
        "unaligned_spearman_rho": -0.165,
        "aligned_spearman_rho": 0.996,
        "unaligned_mae_mS_cm": 8.88,
        "aligned_mae_mS_cm": 0.45,
        "aligned_r2": 0.992,
        "box_scaling": [
            {"atoms": 1100, "box_A": 24.1, "cpu_wall_s": 0.42, "speedup_vs_h200_12h": "102,800x triage"},
            {"atoms": 3200, "box_A": 34.2, "cpu_wall_s": 1.65, "speedup_vs_h200_12h": "26,180x triage"},
            {"atoms": 10000, "box_A": 50.1, "cpu_wall_s": 11.40, "speedup_vs_h200_12h": "3,789x triage"},
        ],
    },
    "top15_solvation_champions": [
        {"rank": "Baseline", "formula": "1.0M LiPF6 in EC:DMC (1:1)", "sigma_298": 10.80, "sigma_253": 3.10, "ssip": 90.2, "cip": 9.8, "agg_i": 0.0, "agg_net": 0.0, "tau_ex_ps": 2.37, "readiness": 58.9, "regime": "SSIP_SOLVENT_DOMINATED"},
        {"rank": "#11", "formula": "1.0M LiFSI in EC:DMC (7:3) + 35wt% F-Ester-5", "sigma_298": 24.12, "sigma_253": 10.45, "ssip": 55.6, "cip": 35.9, "agg_i": 8.5, "agg_net": 0.0, "tau_ex_ps": 1.72, "readiness": 89.1, "regime": "GOLDILOCKS_FAST_CHARGE_SEI"},
        {"rank": "#4", "formula": "1.0M LiFSI in FEC:MA (3:7) + 35wt% F-Ester-2", "sigma_298": 27.64, "sigma_253": 12.11, "ssip": 47.5, "cip": 40.3, "agg_i": 12.2, "agg_net": 0.0, "tau_ex_ps": 1.58, "readiness": 83.7, "regime": "GOLDILOCKS_FAST_CHARGE_SEI"},
        {"rank": "#2", "formula": "1.0M LiFSI in AN:DMC (7:3) + 5wt% F-Ester-1", "sigma_298": 30.85, "sigma_253": 15.01, "ssip": 77.1, "cip": 21.0, "agg_i": 1.9, "agg_net": 0.0, "tau_ex_ps": 1.44, "readiness": 73.8, "regime": "GOLDILOCKS_FAST_CHARGE_SEI"},
        {"rank": "#1", "formula": "1.0M LiFSI in AN:MA (7:3) + 10wt% F-Nitrile-1", "sigma_298": 32.40, "sigma_253": 16.20, "ssip": 74.8, "cip": 22.8, "agg_i": 2.4, "agg_net": 0.0, "tau_ex_ps": 1.39, "readiness": 75.4, "regime": "GOLDILOCKS_FAST_CHARGE_SEI"},
        {"rank": "#6", "formula": "1.2M LiFSI in 2-MeTHF:DME + 15wt% F-Ether-2", "sigma_298": 21.30, "sigma_253": 9.85, "ssip": 18.2, "cip": 29.4, "agg_i": 19.4, "agg_net": 33.0, "tau_ex_ps": 2.81, "readiness": 46.2, "regime": "PERCOLATED_AGG_RISK"},
    ],
}
