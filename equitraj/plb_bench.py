"""Ground-truth protein-ligand pocket MD benchmark on OpenForceField / Wang X-ray complexes."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from equitraj import binding
from equitraj.pocket import (
    PocketMicroEngine,
    compute_openbpmd_scores,
    parse_pdb,
    posebusters_vdw_audit,
    tau_ramd_probe,
)


def compute_pose_auroc(natives: list[dict], decoys: list[dict]) -> dict:
    """Compute Mann-Whitney U AUROC for both combined pose triage and pure MD CompScore."""
    wins = 0.0
    wins_comp = 0.0
    pairs = max(1, len(natives) * len(decoys))
    for nr in natives:
        sn = -nr["comp_score"] + 0.6 * nr["tau_ramd_ps"] - 0.25 * nr["vdw_clashes"]
        sc_n = -nr["comp_score"]
        for dr in decoys:
            sd = -dr["comp_score"] + 0.6 * dr["tau_ramd_ps"] - 0.25 * dr["vdw_clashes"]
            sc_d = -dr["comp_score"]
            wins += 1.0 if sn > sd else (0.5 if abs(sn - sd) < 1e-9 else 0.0)
            wins_comp += 1.0 if sc_n > sc_d else (0.5 if abs(sc_n - sc_d) < 1e-9 else 0.0)
    return {
        "pose_discrimination_auroc": round(wins / pairs, 4),
        "comp_score_only_auroc": round(wins_comp / pairs, 4),
    }


def benchmark_binding_pose_and_potency(plb_root: str | Path | None = None, *,
                                       targets: tuple[str, ...] = ("cdk2", "mcl1", "thrombin", "tyk2"),
                                       max_ligands_per_target: int = 5, n_frames: int = 36, seed: int = 0) -> dict:
    """Benchmark fast pocket MD rollout on real X-ray protein-ligand complexes vs. clash/flipped decoys."""
    import yaml
    from rdkit import Chem
    from rdkit.Chem import AllChem

    if plb_root is None or not Path(plb_root).exists():
        raise FileNotFoundError(f"OpenForceField benchmark root not found: {plb_root}")
    plb_root = Path(plb_root)

    records = []
    t0_bench = time.perf_counter()

    for t_name in targets:
        t_dir = plb_root / t_name
        pdb_path = t_dir / "01_protein" / "crd" / "protein.pdb"
        sdf_path = t_dir / "02_ligands" / "ligands.sdf"
        yml_path = t_dir / "00_data" / "ligands.yml"
        if not (pdb_path.exists() and sdf_path.exists() and yml_path.exists()):
            continue

        prot = parse_pdb(pdb_path)
        prot_pos = prot["positions"]
        prot_elem = prot["elements"]
        prot_mass = prot["masses"]

        with yml_path.open("r", encoding="utf-8") as f:
            aff_meta = yaml.safe_load(f)

        mols = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if m is not None][:max_ligands_per_target]
        for m_idx, mol in enumerate(mols):
            AllChem.ComputeGasteigerCharges(mol)
            conf = mol.GetConformer()
            lig_name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{t_name}_{m_idx}"
            n_lig = mol.GetNumAtoms()
            lig_pos = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
                                for i in range(n_lig)], dtype=np.float64)
            lig_elem = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_lig)]
            lig_mass = np.array([mol.GetAtomWithIdx(i).GetMass() for i in range(n_lig)], dtype=np.float64)
            lig_q = np.array([
                float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))
                if mol.GetAtomWithIdx(i).HasProp("_GasteigerCharge") and
                math.isfinite(float(mol.GetAtomWithIdx(i).GetProp("_GasteigerCharge"))) else 0.0
                for i in range(n_lig)
            ], dtype=np.float64)

            heavy_mask = np.array([e != "H" for e in prot_elem], dtype=bool)
            heavy_idx = np.flatnonzero(heavy_mask)
            d_mat = np.linalg.norm(prot_pos[heavy_idx, None, :] - lig_pos[None, :, :], axis=-1)
            near_pock = heavy_idx[d_mat.min(axis=-1) <= 4.8]
            if len(near_pock) > 55:
                order_p = np.argsort(d_mat.min(axis=-1)[d_mat.min(axis=-1) <= 4.8])[:55]
                near_pock = near_pock[order_p]
            if len(near_pock) < 8:
                continue

            pock_pos = prot_pos[near_pock]
            pock_elem = [prot_elem[i] for i in near_pock]
            pock_mass = prot_mass[near_pock]
            pock_q = np.array([-0.38 if e == "O" else (-0.28 if e == "N" else 0.08) for e in pock_elem], dtype=np.float64)

            lig_bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
            dg_exp = None
            if lig_name in aff_meta and "measurement" in aff_meta[lig_name]:
                m_entry = aff_meta[lig_name]["measurement"]
                val = float(m_entry["value"])
                unit = str(m_entry.get("unit", "")).lower()
                if str(m_entry.get("type", "")).lower() == "pic50":
                    kd_m = 10.0 ** (-val)
                elif unit == "nm":
                    kd_m = val * 1e-9
                else:
                    kd_m = val * 1e-6
                dg_exp = 0.5924 * math.log(max(kd_m, 1e-15))

            for pose_kind in ("native_crystal", "perturbed_decoy"):
                t_c = time.perf_counter()
                cur_lig = lig_pos.copy()
                if pose_kind == "perturbed_decoy":
                    com = cur_lig.mean(axis=0)
                    cur_lig = (cur_lig - com) @ np.diag([-1.0, -1.0, 1.0]) + com + np.array([1.05, -0.65, 0.45])

                pos_all = np.vstack([cur_lig, pock_pos])
                masses_all = np.concatenate([lig_mass, pock_mass])
                elems_all = lig_elem + pock_elem
                charges_all = np.concatenate([lig_q, pock_q])
                sig_all = np.array([2.45 if e == "H" else (2.95 if e in ("O", "N", "F") else 3.35) for e in elems_all])
                eps_all = np.array([0.08 if e == "H" else 0.36 for e in elems_all])

                bp = np.asarray(lig_bonds, dtype=np.int64) if lig_bonds else np.zeros((0, 2), dtype=np.int64)
                br0 = np.linalg.norm(cur_lig[bp[:, 0]] - cur_lig[bp[:, 1]], axis=-1) if len(bp) else np.empty(0)
                bk = np.full(len(bp), 1450.0)
                excl = {(min(a, b), max(a, b)) for a, b in lig_bonds}
                adj: dict[int, list[int]] = {}
                for a, b in lig_bonds:
                    adj.setdefault(a, []).append(b)
                    adj.setdefault(b, []).append(a)
                for nbrs in adj.values():
                    for i1 in range(len(nbrs)):
                        for i2 in range(i1 + 1, len(nbrs)):
                            excl.add((min(nbrs[i1], nbrs[i2]), max(nbrs[i1], nbrs[i2])))

                lig_idx = np.arange(n_lig, dtype=np.int64)
                pock_idx = np.arange(n_lig, len(pos_all), dtype=np.int64)
                for pa in range(len(pock_idx)):
                    for pb_i in range(pa + 1, len(pock_idx)):
                        excl.add((int(pock_idx[pa]), int(pock_idx[pb_i])))

                eng = PocketMicroEngine(
                    masses_all, charges_all, sig_all, eps_all,
                    bond_pairs=bp, bond_r0=br0, bond_k=bk, excluded_pairs=excl,
                    restraint_indices=pock_idx, restraint_anchors=pock_pos, restraint_k=85.0,
                    dielectric=2.0,
                )
                pb_check = posebusters_vdw_audit(pos_all, elems_all, group_a=lig_idx, group_b=pock_idx)
                tx, _ = eng.run(pos_all, n_frames=n_frames, substeps=5, dt_fs=1.2,
                                temperature_k=300.0, friction_per_ps=7.0, seed=seed + m_idx)
                half = max(1, n_frames // 2)
                rmsd_curve = binding.ligand_rmsd(tx, lig_idx, pock_idx)
                pers = binding.contact_persistence(np.concatenate([tx[:1], tx[half:]], axis=0), lig_idx, pock_idx, cutoff=4.5)
                bpm = compute_openbpmd_scores([float(rmsd_curve[half:].mean())], [float(pers["mean_persistence"])])
                ramd_force = max(85.0, 0.45 * float(lig_mass.sum()))
                ramd = tau_ramd_probe(eng, pos_all, lig_idx, pock_idx, masses_all,
                                      n_replicas=2, max_blocks=24, force_kj_mol_a=ramd_force,
                                      unbind_threshold_a=3.4, seed=seed + 71 + m_idx)
                wall_case = time.perf_counter() - t_c
                records.append({
                    "target": t_name,
                    "ligand": lig_name,
                    "pose_kind": pose_kind,
                    "is_native": 1 if pose_kind == "native_crystal" else 0,
                    "n_pocket_complex_atoms": int(len(pos_all)),
                    "dg_exp_kcal_mol": None if dg_exp is None else round(dg_exp, 2),
                    "comp_score": round(bpm["comp_score"], 3),
                    "pose_rmsd_A": round(bpm["pose_score_A"], 3),
                    "contact_persistence": round(bpm["pers_score"], 3),
                    "tau_ramd_ps": ramd["tau_ramd_median_ps"],
                    "vdw_clashes": pb_check["n_vdw_clashes"],
                    "wall_s": round(wall_case, 3),
                })

    total_wall = time.perf_counter() - t0_bench
    natives = [r for r in records if r["is_native"] == 1]
    decoys = [r for r in records if r["is_native"] == 0]
    auc = compute_pose_auroc(natives, decoys)
    mean_wall = float(np.mean([r["wall_s"] for r in records])) if records else 0.0

    return {
        "n_complexes_evaluated": len(records),
        "n_native_crystal_poses": len(natives),
        "n_perturbed_decoys": len(decoys),
        "targets": list(targets),
        "pose_discrimination_auroc": auc["pose_discrimination_auroc"],
        "comp_score_only_auroc": auc["comp_score_only_auroc"],
        "native_mean_comp_score": round(float(np.mean([r["comp_score"] for r in natives])), 3) if natives else None,
        "decoy_mean_comp_score": round(float(np.mean([r["comp_score"] for r in decoys])), 3) if decoys else None,
        "native_mean_tau_ramd_ps": round(float(np.mean([r["tau_ramd_ps"] for r in natives])), 3) if natives else None,
        "decoy_mean_tau_ramd_ps": round(float(np.mean([r["tau_ramd_ps"] for r in decoys])), 3) if decoys else None,
        "mean_wall_s_per_pose_cpu": round(mean_wall, 3),
        "total_wall_s": round(total_wall, 3),
        "speedup_vs_explicit_openbpmd_6_5h": round((6.5 * 3600.0) / max(mean_wall, 1e-4), 0),
        "sample_records": records[:12],
    }
