"""Regression tests for Studio validation, LiPF6 relaxation, compliance gates, and PocketMD (#93)."""

import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from equitraj import cli, compliance, pocket
from equitraj.compliance import run_compliance_audit
from equitraj.pocket import SYSTEMS_DIR, parse_openmm_xml, parse_pdb
from equitraj.studio import create_studio_server
from equitraj.studio_data import run_studio_rollout


def _http_request(url: str, *, method: str = "GET", payload: dict | None = None, headers: dict | None = None):
    hdrs = dict(headers or {})
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        hdrs.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
            return resp.status, dict(resp.headers.items()), (json.loads(raw) if raw else None)
    except urllib.error.HTTPError as err:
        raw = err.read().decode("utf-8")
        return err.code, dict(err.headers.items()), (json.loads(raw) if raw else None)


def test_studio_http_validation_cors_head_and_forbidden_paths():
    srv = create_studio_server("127.0.0.1", 0)
    port = srv.server_address[1]
    base = f"http://127.0.0.1:{port}"
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        # HEAD works cleanly with empty body
        code, hdrs, body = _http_request(f"{base}/api/health", method="HEAD", headers={"Origin": "https://evil.example"})
        assert code == 200 and body is None
        cors = hdrs.get("Access-Control-Allow-Origin") or hdrs.get("access-control-allow-origin")
        assert cors != "*" and "evil.example" not in (cors or "")

        # Local file paths in HTTP /api/triage are rejected with 400
        for bad_key in ("pdb_path", "xml_path", "ckpt_path"):
            code, _, err = _http_request(f"{base}/api/triage", method="POST", payload={"mode": "binding", bad_key: "secret.pdb"})
            assert code == 400 and bad_key in err["error"]

        # Python 0 == False / True == 1 thermostat_K traps return 400
        for bad_therm in (0, True, -5):
            code, _, _ = _http_request(f"{base}/api/cpu_predict", method="POST", payload={"thermostat_K": bad_therm})
            assert code == 400

        # Invalid compliance precision / euler_deg / rollout inputs return 400 instead of 500 or NaN JSON
        code, _, _ = _http_request(f"{base}/api/compliance", method="POST", payload={"precision": "fp16"})
        assert code == 400
        code, _, _ = _http_request(f"{base}/api/compliance", method="POST", payload={"euler_deg": [10.0]})
        assert code == 400
        code, _, _ = _http_request(f"{base}/api/rollout", method="POST", payload={"custom_xyz": "1\nc\nAr 0 0 0"})
        assert code == 400
        code, _, _ = _http_request(f"{base}/api/rollout", method="POST", payload={"temperature_k": -20.0})
        assert code == 400
    finally:
        srv.shutdown()
        srv.server_close()


def test_li_ec_pf6_rollout_relaxes_cleanly_and_xyz_timestamps_are_exact():
    rep = run_studio_rollout(preset="li_ec_pf6", n_frames=36, substeps=6, dt_fs=1.5, temperature_k=300.0, seed=0)
    assert rep["verdict"] == "GO"
    assert 200.0 <= rep["metrics"]["T_mean_K"] <= 390.0
    assert rep["metrics"]["min_vdw_ratio"] >= 0.70

    # Viewer includes Li+ and PF6- (P and 6 F) and only whole 10-atom EC molecules
    elems = rep["viewer_elements"]
    assert elems.count("Li") == 1
    assert elems.count("P") == 1 and elems.count("F") == 6
    assert (len(elems) - 8) % 10 == 0  # every included EC molecule has all 10 atoms

    # XYZ export step and time_ps match actual strided frame index
    r_asp = run_studio_rollout(preset="aspirin", n_frames=54, substeps=8, dt_fs=1.5, temperature_k=300.0)
    step_lines = [ln for ln in r_asp["xyz_download"].splitlines() if ln.startswith("step=")]
    assert step_lines[-1] == "step=52 time_ps=0.6240"


def test_compliance_fp32_clean_aspirin_f8_clash_and_late_nan_crash(monkeypatch):
    # Clean aspirin_hetero_o3 passes 10/10 in both float32 and float64
    for prec in ("float32", "float64"):
        rep = run_compliance_audit(target_id="aspirin_hetero_o3", fault="none", precision=prec, rollout_steps=12)
        assert rep["verdict"] == "GO" and rep["pass_count"] == 10

    # F8_steric_clash_no_prior displaces a non-bonded pair so min_vdw drops below 0.72
    rep_f8 = run_compliance_audit(target_id="aspirin_hetero_o3", fault="F8_steric_clash_no_prior", precision="float32", rollout_steps=12)
    g8 = next(g for g in rep_f8["gates"] if g["id"] == "G8")
    assert g8["status"] == "FAIL" and g8["value"] < 0.72

    # A rollout blowing up to NaN at step 12 of 14 must FAIL G8 and G9 instead of reporting 100% survival
    orig_load = compliance._load_audit_system

    def nan_at_step_12(target_id, precision="float32", seed=0):
        ctx = orig_load(target_id, precision=precision, seed=seed)
        orig_pred = ctx["model"].predict
        calls = {"n": 0}

        def boom_predict(*args, **kwargs):
            calls["n"] += 1
            dx, vn = orig_pred(*args, **kwargs)
            # Algebraic gates run ~10 calls, then rollout starts; blow up near the end of rollout
            if calls["n"] >= 20:
                return torch.full_like(dx, float("nan")), torch.full_like(vn, float("nan"))
            return dx, vn

        ctx["model"].predict = boom_predict
        return ctx

    monkeypatch.setattr(compliance, "_load_audit_system", nan_at_step_12)
    rep_nan = run_compliance_audit(target_id="aspirin_hetero_o3", fault="none", precision="float32", rollout_steps=14)
    g8_nan = next(g for g in rep_nan["gates"] if g["id"] == "G8")
    g9_nan = next(g for g in rep_nan["gates"] if g["id"] == "G9")
    assert g8_nan["status"] == "FAIL" and g9_nan["status"] == "FAIL" and rep_nan["verdict"] == "REJECT"


def test_pocket_pdb_elements_14_exceptions_ligand_xml_and_cli(tmp_path):
    # PDB without columns 77-78 infers biomolecular elements instead of Ca/He
    pdb_txt = (
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00\n"
        "ATOM      2  HE1 TRP A   2       1.500   0.000   0.000  1.00  0.00\n"
        "ATOM      3  NH1 ARG A   3       3.000   0.000   0.000  1.00  0.00\n"
        "HETATM    4  LI  LI  B   1       5.000   0.000   0.000  1.00  0.00\n"
    )
    p_file = tmp_path / "no_elem_cols.pdb"
    p_file.write_text(pdb_txt, encoding="utf-8")
    parsed_pdb = parse_pdb(p_file)
    assert parsed_pdb["elements"] == ["C", "H", "N", "Li"]
    assert parsed_pdb["masses"][0] == pytest.approx(12.011)

    # OpenMM XML keeps 1-4 scaled pairs in exceptions_14 instead of excluded_pairs
    ff_asp = parse_openmm_xml(SYSTEMS_DIR / "aspirin.xml")
    assert len(ff_asp["exceptions_14"]) > 0
    ex14_pairs = {(i, j) for i, j, *_ in ff_asp["exceptions_14"]}
    assert not (ex14_pairs & ff_asp["excluded_pairs"])

    # triage_binding supports ligand-only XML and raises FileNotFoundError on missing XML
    asp_pdb = parse_pdb(SYSTEMS_DIR / "aspirin.pdb")
    complex_lines = []
    for i, (el, (x, y, z)) in enumerate(zip(asp_pdb["elements"], asp_pdb["positions"]), start=1):
        complex_lines.append(f"HETATM{i:5d}  {el:<3s} LIG A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}")
    complex_lines.append("TER")
    for k in range(6):
        s = len(asp_pdb["elements"]) + 1 + k
        complex_lines.append(f"ATOM  {s:5d}  CA  ALA B  {k+1:2d}    {5.0 + 1.5*k:8.3f}   4.000   0.000  1.00  0.00           C")
    comp_path = tmp_path / "complex.pdb"
    comp_path.write_text("\n".join(complex_lines) + "\n", encoding="utf-8")

    rep_bind = pocket.triage_binding(
        pdb_path=str(comp_path), xml_path=str(SYSTEMS_DIR / "aspirin.xml"), n_frames=8, n_replicas=1, seed=0
    )
    assert rep_bind["verdict"] in ("GO", "HOLD", "REJECT")

    with pytest.raises(FileNotFoundError):
        pocket.triage_binding(pdb_path=str(comp_path), xml_path=str(tmp_path / "missing.xml"), n_frames=4, n_replicas=1)

    # CLI: leading blank line in XYZ works, 1-atom XYZ fails, catalog --out writes file
    xyz_ok = tmp_path / "padded.xyz"
    xyz_ok.write_text("\n\n2\ndimer\nC 0.0 0.0 0.0\nO 1.22 0.0 0.0\n", encoding="utf-8")
    assert len(cli.parse_xyz(xyz_ok)["elements"]) == 2

    xyz_one = tmp_path / "one.xyz"
    xyz_one.write_text("1\nmonomer\nAr 0.0 0.0 0.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least 2 atoms"):
        cli.parse_xyz(xyz_one)

    cat_out = tmp_path / "catalog.json"
    assert cli.main(["catalog", "--json", "--out", str(cat_out)]) == 0
    assert len(json.loads(cat_out.read_text(encoding="utf-8"))) >= 4
