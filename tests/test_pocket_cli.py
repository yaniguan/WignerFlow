"""Tests for the unified `wignerflow-pocket` CLI (`equitraj/cli.py` and `python -m equitraj`)."""

import json
import subprocess
import sys

import numpy as np

from equitraj.cli import _ETHANOL_ELEMENTS, _ETHANOL_POS_A, main, parse_xyz


def test_cli_doctor_receipt_and_json(capsys, tmp_path):
    rc = main(["doctor"])
    assert rc == 0
    out_txt = capsys.readouterr().out
    assert "PocketMD Triage Calculator" in out_txt
    assert "[GO]" in out_txt

    json_file = tmp_path / "doctor.json"
    rc_json = main(["doctor", "--json", "--out", str(json_file)])
    assert rc_json == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "doctor"
    assert payload["verdict"] == "GO"
    assert payload["metrics"]["live_bench_steps_per_s"] > 50.0
    saved = json.loads(json_file.read_text(encoding="utf-8"))
    assert saved["verdict"] == "GO"


def test_cli_binding_go_vs_reject(capsys):
    rc_go = main([
        "binding", "--preset", "aspirin_cox_stable",
        "--replicas", "2", "--frames", "40", "--json",
    ])
    assert rc_go == 0
    rep_go = json.loads(capsys.readouterr().out)
    assert rep_go["mode"] == "binding"
    assert rep_go["verdict"] == "GO"
    assert rep_go["metrics"]["comp_score"] <= -1.0

    rc_rej = main([
        "binding", "--preset", "aspirin_decoy_clash",
        "--replicas", "2", "--frames", "40", "--json",
    ])
    assert rc_rej == 0
    rep_rej = json.loads(capsys.readouterr().out)
    assert rep_rej["verdict"] == "REJECT"
    assert rep_rej["metrics"]["comp_score"] > rep_go["metrics"]["comp_score"]


def test_cli_electrolyte_go_vs_agg_mirage(capsys):
    rc_go = main([
        "electrolyte", "--preset", "li_ec_fec_additive",
        "--frames", "60", "--seed", "0", "--json",
    ])
    assert rc_go == 0
    rep_go = json.loads(capsys.readouterr().out)
    assert rep_go["mode"] == "electrolyte"
    assert rep_go["verdict"] in ("GO", "HOLD")
    assert rep_go["metrics"]["ionicity_haven_inv"] > 0.45

    rc_agg = main([
        "electrolyte", "--preset", "li_high_salt_agg",
        "--frames", "60", "--seed", "0", "--json",
    ])
    assert rc_agg == 0
    rep_agg = json.loads(capsys.readouterr().out)
    assert rep_agg["verdict"] == "REJECT"
    assert rep_agg["metrics"]["sigma_cNE_mS_cm"] < rep_agg["metrics"]["sigma_NE_mS_cm"]


def test_cli_checkpoint_audit(capsys):
    rc = main(["checkpoint", "--steps", "12", "--sampling-steps", "2", "--json"])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["mode"] == "checkpoint"
    assert rep["n_atoms"] > 0
    assert rep["metrics"]["com_kinetic_leakage_ratio"] < 1e-3


def test_cli_rollout_xyz_and_exports(capsys, tmp_path):
    in_xyz = tmp_path / "ethanol_in.xyz"
    lines = [str(len(_ETHANOL_ELEMENTS)), "ethanol test geometry"]
    for el, (x, y, z) in zip(_ETHANOL_ELEMENTS, _ETHANOL_POS_A):
        lines.append(f"{el} {x:.6f} {y:.6f} {z:.6f}")
    in_xyz.write_text("\n".join(lines) + "\n", encoding="utf-8")

    out_npz = tmp_path / "ethanol_traj.npz"
    out_xyz = tmp_path / "ethanol_traj.xyz"
    rc = main([
        "rollout",
        "--xyz", str(in_xyz),
        "--frames", "30",
        "--substeps", "6",
        "--dt-fs", "1.2",
        "--out-npz", str(out_npz),
        "--out-xyz", str(out_xyz),
        "--json",
    ])
    assert rc == 0
    rep = json.loads(capsys.readouterr().out)
    assert rep["mode"] == "rollout"
    assert rep["n_atoms"] == 9
    assert rep["metrics"]["n_bonds"] == 8
    assert rep["metrics"]["com_kinetic_leakage_ratio"] < 1e-10
    assert out_npz.exists()
    assert out_xyz.exists()

    arr = np.load(out_npz)
    assert arr["positions_A"].shape == (30, 9, 3)
    parsed_back = parse_xyz(out_xyz)
    assert len(parsed_back["elements"]) == 9


def test_python_m_equitraj_entrypoint():
    proc = subprocess.run(
        [sys.executable, "-m", "equitraj", "catalog", "--domain", "electrolyte", "--json"],
        check=True,
        capture_output=True,
        text=True,
    )
    items = json.loads(proc.stdout)
    assert len(items) >= 2
    assert all(it["domain"] == "electrolyte" for it in items)
