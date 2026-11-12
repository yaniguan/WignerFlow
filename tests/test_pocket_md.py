"""Tests for PocketMD v2 literature-aligned CPU triage calculator (`equitraj.pocket` & `scripts/pocket_md.py`)."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from equitraj import pocket

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parse_pdb_handles_ter_serial_offsets_and_bonds():
    li_ec = pocket.parse_pdb(REPO_ROOT / "systems" / "li_ec_pf6_q08.pdb")
    # 1 Li + 100 EC (1000 atoms) + 1 PF6 (7 atoms) = 1008 atoms across 102 TER-separated molecules
    assert li_ec["positions"].shape == (1008, 3)
    assert li_ec["box"] is not None and np.isclose(li_ec["box"][0, 0], 23.143, atol=1e-2)
    assert int(li_ec["mol_ids"].max()) == 101
    assert len(li_ec["bonds"]) >= 1000
    assert int(li_ec["bonds"].max()) < 1008


def test_parse_openmm_xml_extracts_openff_aspirin_parameters():
    ff = pocket.parse_openmm_xml(REPO_ROOT / "systems" / "aspirin.xml")
    assert len(ff["masses"]) == 21
    assert np.isclose(ff["charges"].sum(), 0.0, atol=1e-4)
    assert len(ff["bond_pairs"]) == 21  # 13 heavy bonds + 8 X-H constraints
    assert len(ff["angle_triplets"]) > 20
    assert (ff["sigma_a"] > 0.5).all()


def test_openbpmd_comp_score_and_posebusters_vdw_audit():
    scores = pocket.compute_openbpmd_scores([1.1, 1.3], [0.80, 0.70])
    assert np.isclose(scores["pose_score_A"], 1.2)
    assert np.isclose(scores["pers_score"], 0.75)
    assert np.isclose(scores["comp_score"], 1.2 - 5.0 * 0.75)

    # PoseBusters check: two C atoms (r_vdW = 1.70 Å each -> sum 3.40 Å)
    clean_pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])  # ratio = 3.0 / 3.4 = 0.882 >= 0.75
    clash_pos = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])  # ratio = 1.8 / 3.4 = 0.529 < 0.75
    assert pocket.posebusters_vdw_audit(clean_pos, ["C", "C"])["n_vdw_clashes"] == 0
    clash_res = pocket.posebusters_vdw_audit(clash_pos, ["C", "C"])
    assert clash_res["n_vdw_clashes"] == 1
    assert clash_res["min_vdw_ratio"] < 0.75


def test_nernst_einstein_and_cluster_nernst_einstein_neutral_pair_mirage():
    # 1. Analytical uncorrelated Nernst-Einstein check
    ne = pocket.nernst_einstein_metrics(
        2.0e-6, 3.0e-6,
        n_cation=2, n_anion=2, z_cation=1.0, z_anion=-1.0,
        volume_a3=1000.0, temperature_k=300.0,
    )
    expected_sigma = pocket.NERNST_EINSTEIN_MS_CM * (2 * 2.0e-6 + 2 * 3.0e-6) / (1000.0 * 300.0)
    assert np.isclose(ne["sigma_ne_mS_cm"], expected_sigma, rtol=1e-6)
    assert np.isclose(ne["transference_plus_ne"], 0.4, atol=1e-6)

    with pytest.raises(ValueError):
        pocket.nernst_einstein_metrics(1e-6, 1e-6, n_cation=1, n_anion=1, z_cation=1, z_anion=-1,
                                       volume_a3=0.0, temperature_k=300.0)

    # 2. France-Lanord Cluster-NE check:
    # A tightly bound neutral [Li+ PF6-]^0 contact ion pair co-diffusing linearly has D_+ = D_- > 0
    # (so naive sigma_NE > 0), MUST yield sigma_cNE == 0 and ionicity == 0.0!
    t_steps = 20
    drift = np.linspace(0.0, 3.0, t_steps)[:, None] * np.array([[1.0, 0.5, -0.5]])
    pos_pair = np.zeros((t_steps, 2, 3), dtype=np.float64)
    pos_pair[:, 0, :] = np.array([2.0, 2.0, 2.0]) + drift       # Li+
    pos_pair[:, 1, :] = np.array([4.0, 2.0, 2.0]) + drift       # Anion at 2.0 Å (< r_pair=4.25 Å)
    cne_pair = pocket.cluster_nernst_einstein(
        pos_pair, 20.0, [0], [1], z_cation=1.0, z_anion=-1.0, dt_frame_fs=30.0, temperature_k=300.0, r_pair=4.25
    )
    assert cne_pair["sigma_ne_mS_cm"] > 1.0
    assert np.isclose(cne_pair["sigma_cne_mS_cm"], 0.0, atol=1e-9)
    assert np.isclose(cne_pair["ionicity_haven_inv"], 0.0, atol=1e-9)

    # Conversely, well-separated dissociated ions (10 Å apart in 30 Å box) yield sigma_cNE == sigma_NE (ionicity = 1.0)
    pos_free = np.zeros((t_steps, 2, 3), dtype=np.float64)
    pos_free[:, 0, :] = np.array([2.0, 2.0, 2.0]) + drift
    pos_free[:, 1, :] = np.array([15.0, 15.0, 15.0]) - drift
    cne_free = pocket.cluster_nernst_einstein(
        pos_free, 30.0, [0], [1], z_cation=1.0, z_anion=-1.0, dt_frame_fs=30.0, temperature_k=300.0, r_pair=4.25
    )
    assert np.isclose(cne_free["sigma_cne_mS_cm"], cne_free["sigma_ne_mS_cm"], rtol=1e-6)
    assert np.isclose(cne_free["ionicity_haven_inv"], 1.0, atol=1e-6)

    # 3. Asymmetric negatively charged triplet [Li(Anion)_2]^- yields negative cation transference t+_cNE < 0!
    pos_trip = np.zeros((t_steps, 3, 3), dtype=np.float64)
    pos_trip[:, 0, :] = np.array([5.0, 5.0, 5.0]) + drift       # 1 Li+
    pos_trip[:, 1, :] = np.array([7.0, 5.0, 5.0]) + drift       # Anion 1 bound
    pos_trip[:, 2, :] = np.array([3.0, 5.0, 5.0]) + drift       # Anion 2 bound
    cne_trip = pocket.cluster_nernst_einstein(
        pos_trip, 30.0, [0], [1, 2], z_cation=1.0, z_anion=-1.0, dt_frame_fs=30.0, temperature_k=300.0, r_pair=4.25
    )
    assert cne_trip["sigma_cne_mS_cm"] > 0.0
    assert np.isclose(cne_trip["transference_plus_cne"], -1.0, atol=1e-6)


def test_competitive_additive_score_and_solvation_speciation():
    sac = pocket.competitive_additive_score(2.2, 1.8, n_solvent_mols=9, n_additive_mols=3, paired_fraction=0.0)
    assert sac["cags_additive_score"] > 0.5
    assert sac["additive_role"] == "sacrificial_sei_former"

    dil = pocket.competitive_additive_score(1.0, 0.0, n_solvent_mols=8, n_additive_mols=4, paired_fraction=0.8)
    assert dil["cags_additive_score"] < -1.0
    assert dil["additive_role"] == "weak_solvation_anion_recruiter"

    base = pocket.competitive_additive_score(4.0, 0.0, n_solvent_mols=12, n_additive_mols=0)
    assert base["cags_additive_score"] is None
    assert base["additive_role"] == "baseline_no_additive"

    pos = np.array([[
        [1.0, 1.0, 1.0],   # Li 0
        [8.0, 8.0, 8.0],   # Li 1
        [1.0, 3.0, 1.0],   # Solvent O near Li 0
        [8.0, 9.9, 8.0],   # Additive O near Li 1
        [9.8, 8.0, 8.0],   # Anion 0 near Li 1
        [8.0, 8.0, 9.9],   # Anion 1 near Li 1
    ]], dtype=np.float64)
    sp = pocket.solvation_speciation(pos, 15.0, [0, 1], [2], [4, 5], [3], r_shell=2.5)
    assert np.isclose(sp["ssip_fraction"], 0.5)
    assert np.isclose(sp["agg_fraction"], 0.5)


def test_triage_binding_separates_stable_induced_and_decoy_poses():
    stable = pocket.triage_binding("aspirin_cox_stable", n_frames=50, n_replicas=2, seed=0)
    induced = pocket.triage_binding("aspirin_induced_fit", n_frames=50, n_replicas=2, seed=0)
    decoy = pocket.triage_binding("aspirin_decoy_clash", n_frames=50, n_replicas=2, seed=0)

    assert stable["verdict"] == "GO"
    assert induced["verdict"] in ("HOLD", "GO")
    assert decoy["verdict"] == "REJECT"
    assert stable["metrics"]["comp_score"] <= -1.0
    assert stable["metrics"]["comp_score"] < induced["metrics"]["comp_score"] < decoy["metrics"]["comp_score"]
    assert stable["metrics"]["posebusters_min_vdw_ratio"] > decoy["metrics"]["posebusters_min_vdw_ratio"]
    assert stable["metrics"]["tau_ramd_blocks_50fs"] > decoy["metrics"]["tau_ramd_blocks_50fs"]
    assert "PocketMD Triage Calculator" in pocket.format_receipt(stable)


def test_triage_binding_accepts_custom_pdb_and_xml():
    rep = pocket.triage_binding(
        pdb_path=REPO_ROOT / "systems" / "aspirin.pdb",
        xml_path=REPO_ROOT / "systems" / "aspirin.xml",
        n_frames=20,
        n_replicas=2,
        seed=1,
    )
    assert rep["mode"] == "binding"
    assert rep["n_atoms"] == 21
    assert rep["verdict"] in ("GO", "HOLD", "REJECT")


def test_triage_electrolyte_catches_agg_mirage_and_ranks_additives():
    fec = pocket.triage_electrolyte("li_ec_fec_additive", n_frames=60, seed=0)
    vc = pocket.triage_electrolyte("li_ec_vc_additive", n_frames=60, seed=0)
    agg = pocket.triage_electrolyte("li_high_salt_agg", n_frames=60, seed=0)

    assert fec["verdict"] == "GO"
    assert vc["verdict"] == "HOLD"
    assert agg["verdict"] == "REJECT"
    assert fec["metrics"]["additive_role"] == "sacrificial_sei_former"
    assert fec["metrics"]["sigma_cNE_mS_cm"] > vc["metrics"]["sigma_cNE_mS_cm"]
    # Neutral AGG cluster exhibits high naive sigma_NE but near-zero true sigma_cNE!
    assert agg["metrics"]["sigma_NE_mS_cm"] > 10.0
    assert agg["metrics"]["sigma_cNE_mS_cm"] < 1.0
    assert agg["metrics"]["ionicity_haven_inv"] < 0.1
    assert 0.0 <= fec["metrics"]["hopping_index_chi"] <= 1.0


def test_triage_checkpoint_cpu_fast_rollout_and_epistemic_gate():
    rep = pocket.triage_checkpoint(steps=12, sampling_steps=2, seed=0)
    assert rep["mode"] == "checkpoint"
    assert rep["verdict"] == "GO"
    m = rep["metrics"]
    assert m["crashed_at_step"] is None
    assert m["coordination_number"] > 8.0
    assert m["min_vdw_ratio"] > 0.70
    assert m["com_kinetic_leakage_ratio"] < 1e-6
    assert m["ode_curvature_delta_A"] >= 0.0
    assert m["flow_epistemic_spread_A"] >= 0.0


def test_pocket_md_cli_presets_and_binding_json(tmp_path):
    out_json = tmp_path / "binding_rep.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "pocket_md.py"),
        "binding", "--preset", "aspirin_cox_stable", "--frames", "25", "--replicas", "2",
        "--json-out", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[GO]" in proc.stdout or "[HOLD]" in proc.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["name"] == "aspirin_cox_stable"
    assert "comp_score" in payload["metrics"]
    assert len(payload["viewer_frames"]) > 0
