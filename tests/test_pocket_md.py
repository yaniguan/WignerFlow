"""Tests for PocketMD fast CPU triage calculator (`equitraj.pocket` & `scripts/pocket_md.py`)."""

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


def test_nernst_einstein_and_solvation_speciation_physics():
    # Analytical check: 2 Li+ and 2 PF6- (q = +/-1) in 1000 Å^3 at 300 K
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

    # Speciation check: 1 frame, 2 cations: cation 0 has 0 anions (SSIP), cation 1 has 2 anions (AGG)
    pos = np.array([[
        [1.0, 1.0, 1.0],   # Li 0
        [8.0, 8.0, 8.0],   # Li 1
        [1.0, 3.0, 1.0],   # Solvent O near Li 0 (d = 2.0)
        [8.0, 9.9, 8.0],   # Additive O near Li 1 (d = 1.9)
        [9.8, 8.0, 8.0],   # Anion 0 near Li 1 (d = 1.8)
        [8.0, 8.0, 9.9],   # Anion 1 near Li 1 (d = 1.9)
    ]], dtype=np.float64)
    sp = pocket.solvation_speciation(pos, 15.0, [0, 1], [2], [4, 5], [3], r_shell=2.5)
    assert np.isclose(sp["solvent_cn"], 0.5)
    assert np.isclose(sp["additive_cn"], 0.5)
    assert np.isclose(sp["anion_cn"], 1.0)
    assert np.isclose(sp["ssip_fraction"], 0.5)
    assert np.isclose(sp["cip_fraction"], 0.0)
    assert np.isclose(sp["agg_fraction"], 0.5)


def test_triage_binding_separates_stable_induced_and_decoy_poses():
    stable = pocket.triage_binding("aspirin_cox_stable", n_frames=50, seed=0)
    induced = pocket.triage_binding("aspirin_induced_fit", n_frames=50, seed=0)
    decoy = pocket.triage_binding("aspirin_decoy_clash", n_frames=50, seed=0)

    assert stable["verdict"] == "GO"
    assert induced["verdict"] in ("HOLD", "GO")
    assert decoy["verdict"] == "REJECT"
    assert stable["metrics"]["ligand_rmsd_plateau_A"] < decoy["metrics"]["ligand_rmsd_plateau_A"]
    assert stable["metrics"]["contact_persistence_R"] > decoy["metrics"]["contact_persistence_R"]
    assert "PocketMD Triage Calculator" in pocket.format_receipt(stable)


def test_triage_binding_accepts_custom_pdb_and_xml():
    rep = pocket.triage_binding(
        pdb_path=REPO_ROOT / "systems" / "aspirin.pdb",
        xml_path=REPO_ROOT / "systems" / "aspirin.xml",
        n_frames=20,
        seed=1,
    )
    assert rep["mode"] == "binding"
    assert rep["n_atoms"] == 21
    assert rep["verdict"] in ("GO", "HOLD", "REJECT")


def test_triage_electrolyte_separates_fast_additive_from_agg_trap():
    fec = pocket.triage_electrolyte("li_ec_fec_additive", n_frames=60, seed=0)
    vc = pocket.triage_electrolyte("li_ec_vc_additive", n_frames=60, seed=0)
    agg = pocket.triage_electrolyte("li_high_salt_agg", n_frames=60, seed=0)

    assert fec["verdict"] == "GO"
    assert vc["verdict"] == "HOLD"
    assert agg["verdict"] == "REJECT"
    assert fec["metrics"]["additive_cn"] > 0.3
    assert fec["metrics"]["sigma_NE_mS_cm"] > vc["metrics"]["sigma_NE_mS_cm"]
    assert agg["metrics"]["agg_fraction"] + agg["metrics"]["cip_fraction"] >= 0.65


def test_triage_checkpoint_cpu_fast_rollout():
    rep = pocket.triage_checkpoint(steps=12, sampling_steps=2, seed=0)
    assert rep["mode"] == "checkpoint"
    assert rep["verdict"] == "GO"
    assert rep["metrics"]["crashed_at_step"] is None
    assert rep["metrics"]["coordination_number"] > 8.0


def test_pocket_md_cli_presets_and_binding_json(tmp_path):
    out_json = tmp_path / "binding_rep.json"
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "pocket_md.py"),
        "binding", "--preset", "aspirin_cox_stable", "--frames", "25", "--json-out", str(out_json),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "[GO]" in proc.stdout or "[HOLD]" in proc.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["name"] == "aspirin_cox_stable"
    assert len(payload["viewer_frames"]) > 0
