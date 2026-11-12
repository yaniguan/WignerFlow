"""Tests for the 10-Gate Equivariant Physics Compliance Auditor and WignerFlow Studio HTTP API."""

import json
import threading
import urllib.request

from equitraj.compliance import run_compliance_audit
from equitraj.studio import create_studio_server
from equitraj.studio_data import DEFAULT_ETHANOL_XYZ


def _gate_map(report: dict) -> dict[str, dict]:
    return {g["id"]: g for g in report["gates"]}


def test_clean_hetero_o3_passes_all_10_gates_float64():
    rep = run_compliance_audit(target_id="aspirin_hetero_o3", fault="none", precision="float64", rollout_steps=12)
    assert rep["verdict"] == "GO"
    assert rep["pass_count"] == 10
    gm = _gate_map(rep)
    assert gm["G1"]["value"] < 5e-7
    assert gm["G2"]["value"] < 5e-7
    assert gm["G3"]["value"] < 1e-12
    assert gm["G4"]["value"] < 1e-12
    assert gm["G5"]["value"] < 1e-11


def test_chiral_parity_false_preserves_so3_and_trips_o3():
    rep = run_compliance_audit(target_id="aspirin_chiral_so3", fault="none", precision="float64", rollout_steps=12)
    gm = _gate_map(rep)
    assert gm["G1"]["status"] == "PASS"
    assert gm["G2"]["status"] == "FAIL"
    assert rep["verdict"] == "REJECT"


def test_surgical_fault_injections_trip_expected_gates():
    cases = [
        ("F1_break_so3", "G1"),
        ("F2_pole_glitch", "G1"),
        ("F3_chiral_parity_leak", "G2"),
        ("F4_unweighted_com_leak", "G3"),
        ("F5_fixed_box_pbc_glitch", "G5"),
        ("F6_nonconservative_curl", "G6"),
        ("F7_flying_ice_cube", "G7"),
        ("F8_steric_clash_no_prior", "G8"),
        ("F9_deterministic_overstride", "G9"),
        ("F10_ood_bond_stretch", "G10"),
    ]
    for fault_id, expected_gate in cases:
        rep = run_compliance_audit(
            target_id="aspirin_hetero_o3", fault=fault_id, precision="float32", rollout_steps=12
        )
        gm = _gate_map(rep)
        assert rep["verdict"] == "REJECT", f"{fault_id} did not trigger REJECT"
        assert gm[expected_gate]["status"] == "FAIL", f"{fault_id} failed to trip {expected_gate}"


def test_trained_argon_checkpoint_passes_compliance():
    rep = run_compliance_audit(target_id="argon_flow_v1", fault="none", precision="float32", rollout_steps=12)
    assert rep["verdict"] == "GO"
    assert rep["fail_count"] == 0
    assert rep["pass_count"] >= 9


def test_studio_http_endpoints_end_to_end():
    srv = create_studio_server("127.0.0.1", 0)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(f"{base}/") as r_html:
            html = r_html.read().decode("utf-8")
            assert "WignerFlow Studio" in html

        with urllib.request.urlopen(f"{base}/api/overview") as r_ov:
            ov = json.loads(r_ov.read().decode("utf-8"))
            assert len(ov["fault_catalog"]) == 11
            assert "pure_md_md17_md22" in ov["benchmarks"]

        req_comp = urllib.request.Request(
            f"{base}/api/compliance",
            data=json.dumps({
                "target_id": "aspirin_hetero_o3",
                "fault": "F3_chiral_parity_leak",
                "precision": "float32",
                "euler_deg": [25, 70, 140],
                "rollout_steps": 12,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_comp) as r_comp:
            comp = json.loads(r_comp.read().decode("utf-8"))
            gm = _gate_map(comp)
            assert gm["G1"]["status"] == "PASS"
            assert gm["G2"]["status"] == "FAIL"

        req_roll = urllib.request.Request(
            f"{base}/api/rollout",
            data=json.dumps({
                "preset": "ethanol",
                "custom_xyz": DEFAULT_ETHANOL_XYZ,
                "n_frames": 24,
                "dt_fs": 1.2,
                "temperature_k": 300.0,
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req_roll) as r_roll:
            roll = json.loads(r_roll.read().decode("utf-8"))
            assert roll["n_atoms"] == 9
            assert roll["metrics"]["n_bonds"] == 8
            assert len(roll["viewer_frames"]) == 24
    finally:
        srv.shutdown()
        srv.server_close()


def test_benchmark_tables_come_from_result_files():
    from equitraj.studio_data import _read_json, load_benchmark_suites

    suites = load_benchmark_suites()
    md = _read_json("pure_md_md17/md17_md22_benchmark.json")
    if md is None:
        assert suites["pure_md_md17_md22"] == []
    else:
        assert [r["mae_hr"] for r in suites["pure_md_md17_md22"]] == [m["wf_10fs_hr_mae"] for m in md["molecules"]]
    cal = _read_json("calisol/align_eval_seed42.json")
    assert suites["calisol23_alignment"]["available"] is (cal is not None)
    if cal is not None:
        assert suites["calisol23_alignment"]["aligned_r2"] == cal["calisol_alignment"]["full_heldout_test_metrics"]["r2"]
