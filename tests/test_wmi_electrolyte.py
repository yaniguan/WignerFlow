"""Tests for `02_Na_MD` (`MU-PFF` / WMI) polarizable electrolyte ingestion, multi-resolution reduction,
NPT variable-box windows, Onsager conductivity evaluation, and differentiable transport calibration.

Includes both a self-contained synthetic WMI fixture (runs on Linux CI) and live verification against
`best-napf6-ec-dmc-m4a__{r1-npt,r1,r2}` when present on disk.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from equitraj import (
    evaluate_electrolyte_transport,
    import_wmi_replica,
    load_wmi_build_dir,
    make_model,
    transport_calibration_loss,
)
import h5py
from equitraj.__main__ import main as cli_main
from equitraj.adapters import import_trajectory
from equitraj.dataset import TrajWindows, compute_stats
from equitraj.electrolyte import reduce_wmi_frames

REAL_BUILD_ROOT = Path(r"c:\Users\yanig\ws\02_ses\02_Na_MD\build_outputs")
REAL_DIRS = [
    REAL_BUILD_ROOT / "best-napf6-ec-dmc-m4a__r1-npt",
    REAL_BUILD_ROOT / "best-napf6-ec-dmc-m4a__r1",
    REAL_BUILD_ROOT / "best-napf6-ec-dmc-m4a__r2",
]
REAL_RESULTS_MU = Path(
    r"c:\Users\yanig\ws\02_ses\02_Na_MD\_gate_dt2\best-napf6-ec-dmc-m4a__r1-npt\md-analysis\output\results_mu.json"
)


def _write_mini_wmi_build(root: Path) -> Path:
    """Create a miniature `02_Na_MD/build_outputs/<sys>` fixture with `He` `LP*` virtual sites (-0.22 e each)."""
    omm = root / "system" / "openmm"
    omm.mkdir(parents=True, exist_ok=True)
    (root / "config_augmented.json").write_text(json.dumps({"temperature": 298.15}), encoding="utf-8")

    # Residue templates:
    # - MOL1 (dipolar solvent like EC): C0 (+0.60), O0 (-0.60 carbonyl), O1 (+0.22 ether), O2 (+0.22 ether), LP1 (-0.22), LP2 (-0.22) -> net 0.0 after LP fold!
    # - MOL3 (Na+): Na1 (+1.0)
    # - MOL4 (PF6-like): P1 (+0.80), F1 (-0.90), F2 (-0.90) -> net -1.0
    pol_xml = """<ForceField>
  <AtomTypes>
    <Type name="1" class="c" element="C" mass="12.011"/>
    <Type name="2" class="o" element="O" mass="15.999"/>
    <Type name="3" class="os" element="O" mass="15.999"/>
    <Type name="8" class="lp" element="He" mass="0.0"/>
    <Type name="9" class="na" element="Na" mass="22.990"/>
    <Type name="10" class="p" element="P" mass="30.974"/>
    <Type name="11" class="f" element="F" mass="18.998"/>
  </AtomTypes>
  <Residues>
    <Residue name="MOL1">
      <Atom name="C0" type="1"/>
      <Atom name="O0" type="2"/>
      <Atom name="O1" type="3"/>
      <Atom name="O2" type="3"/>
      <Atom name="LP1" type="8"/>
      <Atom name="LP2" type="8"/>
      <Bond atomName1="C0" atomName2="O0"/>
      <Bond atomName1="C0" atomName2="O1"/>
      <Bond atomName1="C0" atomName2="O2"/>
      <Bond atomName1="O1" atomName2="LP1"/>
      <Bond atomName1="O2" atomName2="LP2"/>
    </Residue>
    <Residue name="MOL3">
      <Atom name="Na1" type="9"/>
    </Residue>
    <Residue name="MOL4">
      <Atom name="P1" type="10"/>
      <Atom name="F1" type="11"/>
      <Atom name="F2" type="11"/>
      <Bond atomName1="P1" atomName2="F1"/>
      <Bond atomName1="P1" atomName2="F2"/>
    </Residue>
  </Residues>
  <AmoebaMultipoleForce>
    <Multipole type="1" c0="0.60"/>
    <Multipole type="2" c0="-0.60"/>
    <Multipole type="3" c0="0.22"/>
    <Multipole type="8" c0="-0.22"/>
    <Multipole type="9" c0="1.00"/>
    <Multipole type="10" c0="0.80"/>
    <Multipole type="11" c0="-0.90"/>
    <Polarize type="1" polarizability="0.0010"/>
    <Polarize type="2" polarizability="0.0008"/>
    <Polarize type="3" polarizability="0.0007"/>
    <Polarize type="8" polarizability="0.0002"/>
    <Polarize type="9" polarizability="0.0001"/>
    <Polarize type="10" polarizability="0.0012"/>
    <Polarize type="11" polarizability="0.0005"/>
  </AmoebaMultipoleForce>
</ForceField>
"""
    (omm / "pol_ff.xml").write_text(pol_xml, encoding="utf-8")

    # Build 4 MOL1 (6 sites each = 24), 2 MOL3 (2), 2 MOL4 (3 sites each = 6) -> 32 sites total, 8 LP sites, 24 phys atoms, 8 molecules
    pdb_lines = ["CRYST1   18.000   18.000   18.000  90.00  90.00  90.00 P 1           1"]
    vs_sites = []
    serial = 1
    res_seq = 1
    centers = [
        [3.0, 3.0, 3.0],
        [9.0, 3.0, 3.0],
        [3.0, 9.0, 3.0],
        [9.0, 9.0, 3.0],
    ]
    offsets_mol1 = [
        ("C0", "C", [0.0, 0.0, 0.0]),
        ("O0", "O", [1.2, 0.0, 0.0]),
        ("O1", "O", [-0.8, 0.9, 0.0]),
        ("O2", "O", [-0.8, -0.9, 0.0]),
        ("LP1", "He", [-1.1, 1.2, 0.0]),
        ("LP2", "He", [-1.1, -1.2, 0.0]),
    ]
    for c in centers:
        base_1based = serial
        for nm, el, off in offsets_mol1:
            x, y, z = c[0] + off[0], c[1] + off[1], c[2] + off[2]
            pdb_lines.append(
                f"HETATM{serial:5d} {nm:<4s} MOL1 {res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
            )
            serial += 1
        vs_sites.append({"lp_index_1based": base_1based + 4, "parent_index_1based": base_1based + 2})
        vs_sites.append({"lp_index_1based": base_1based + 5, "parent_index_1based": base_1based + 3})
        res_seq += 1

    for c in ([5.0, 5.0, 10.0], [12.0, 12.0, 10.0]):
        pdb_lines.append(
            f"HETATM{serial:5d} {'Na1':<4s} MOL3 {res_seq:4d}    {c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00  0.00          Na"
        )
        serial += 1
        res_seq += 1

    for c in ([6.2, 5.0, 10.0], [15.5, 12.0, 10.0]):
        for nm, el, off in (("P1", "P", [0.0, 0.0, 0.0]), ("F1", "F", [1.5, 0.0, 0.0]), ("F2", "F", [-1.5, 0.0, 0.0])):
            x, y, z = c[0] + off[0], c[1] + off[1], c[2] + off[2]
            pdb_lines.append(
                f"HETATM{serial:5d} {nm:<4s} MOL4 {res_seq:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
            )
            serial += 1
        res_seq += 1
    pdb_lines.append("END")
    (omm / "system.pdb").write_text("\n".join(pdb_lines) + "\n", encoding="utf-8")
    (root / "system" / "virtual_sites.json").write_text(json.dumps({"sites": vs_sites}, indent=2), encoding="utf-8")
    return root


def _make_synthetic_traj_npz(path: Path, wmi: dict, n_frames: int = 24, seed: int = 0, variable_box: bool = False) -> Path:
    rng = np.random.default_rng(seed)
    pos0 = wmi["pdb_positions_A"]
    t_grid = np.arange(n_frames, dtype=np.float64)[:, None, None]
    drift = rng.normal(0.0, 0.04, size=(1, wmi["n_total_sites"], 3))
    jitter = rng.normal(0.0, 0.02, size=(n_frames, wmi["n_total_sites"], 3))
    pos = pos0[None] + t_grid * drift + jitter
    if variable_box:
        l_t = 18.0 + 0.25 * np.sin(np.linspace(0.0, 2.0 * np.pi, n_frames))
        boxes = np.zeros((n_frames, 3, 3), dtype=np.float64)
        for d in range(3):
            boxes[:, d, d] = l_t
    else:
        boxes = np.broadcast_to(np.eye(3) * 18.0, (n_frames, 3, 3)).copy()
    np.savez(path, positions_A=pos, boxes_A=boxes)
    return path


def test_drop_drude_filters_he_lone_pairs(tmp_path: Path):
    src = tmp_path / "with_lp.npz"
    np.savez(
        src,
        positions=np.zeros((4, 4, 3), dtype=np.float64),
        elements=np.array(["C", "O", "He", "He"]),
        masses=np.array([12.011, 15.999, 4.003, 4.003]),
        names=np.array(["C1", "O1", "LP1", "EP1"]),
        box=np.eye(3) * 15.0,
    )
    h5_out = import_trajectory(src, tmp_path / "imported", unit="angstrom", drop_drude=True)
    with h5py.File(h5_out, "r") as f:
        assert [e.decode() for e in f["elements"][:]] == ["C", "O"]


def test_synthetic_wmi_resolutions_npt_and_calibration(tmp_path: Path):
    build_dir = _write_mini_wmi_build(tmp_path / "mini_wmi")
    wmi = load_wmi_build_dir(build_dir)
    assert wmi["n_total_sites"] == 32
    assert wmi["n_virtual_sites"] == 8
    assert wmi["n_phys_atoms"] == 24
    assert wmi["n_molecules"] == 8
    # Verify LP charge folding: MOL1 solvent must be strictly neutral (0.0 e), Na+ = +1, PF6- = -1
    assert abs(wmi["species_table"]["MOL1"]["charge"]) < 1e-8
    assert np.isclose(wmi["species_table"]["MOL3"]["charge"], 1.0)
    assert np.isclose(wmi["species_table"]["MOL4"]["charge"], -1.0)
    assert abs(float(wmi["phys_charges"].sum())) < 1e-8

    t0 = _make_synthetic_traj_npz(tmp_path / "rep0_npt.npz", wmi, n_frames=24, seed=1, variable_box=True)
    t1 = _make_synthetic_traj_npz(tmp_path / "rep1_nvt.npz", wmi, n_frames=24, seed=2, variable_box=False)

    # 1. Test polar_beads (4 solvents * 2 + 4 ions = 12 beads) & all_atom (24 atoms) imports
    for res, expected_n in (("polar_beads", 12), ("all_atom", 24)):
        out_r = tmp_path / f"ds_{res}"
        h5_p = import_wmi_replica(build_dir, t0, out_r, traj_index=0, resolution=res, dt_frame_fs=100.0)
        assert h5_p.is_file()
        (out_r / "config.yaml").write_text(
            yaml.safe_dump({"system": res, "split": {"train": [0], "val": [0], "test": [0]}}), encoding="utf-8"
        )
        ds_r = TrajWindows(str(out_r), "train", stride_steps=1, k=3, m=1)
        assert len(ds_r.types) == expected_n

    # 2. Test raw fluctuating NPT box (`--no_canonicalize_npt_box` -> `allow_variable_box: true`)
    npt_dir = tmp_path / "ds_mol_raw_npt"
    rc = cli_main([
        "import-wmi",
        "--build_dirs", str(build_dir),
        "--trajs", str(t0), str(t1),
        "--resolution", "molecular",
        "--dt_frame_fs", "200.0",
        "--no_canonicalize_npt_box",
        "--out_dir", str(npt_dir),
    ])
    assert rc == 0
    ds_npt = TrajWindows(str(npt_dir), "train", stride_steps=1, k=3, m=1)
    _, _, box_f0 = ds_npt[0]
    _, _, box_f5 = ds_npt[5]
    assert not torch.allclose(box_f0, box_f5)

    # 3. Canonicalized molecular dataset -> train mini EquiTraj -> evaluate-electrolyte -> calibrate-electrolyte
    mol_dir = tmp_path / "ds_mol_canon"
    assert cli_main([
        "import-wmi",
        "--build_dirs", str(build_dir),
        "--trajs", str(t0), str(t1),
        "--resolution", "molecular",
        "--dt_frame_fs", "200.0",
        "--out_dir", str(mol_dir),
    ]) == 0

    run_dir = tmp_path / "run_mol"
    cfg = {
        "data_dir": str(mol_dir),
        "stride_steps": 1,
        "k": 3,
        "seed": 1,
        "model": {
            "arch": "equi",
            "head": "flow",
            "lmax": 1,
            "channels": 16,
            "heads": 2,
            "n_rbf": 8,
            "cutoff": 7.5,
            "n_spatial": 1,
            "n_fusion": 1,
            "n_denoiser": 1,
            "target": "residual",
            "target_scaling": "mass",
            "n_ff_types": "auto",
            "use_charges": True,
        },
        "train": {
            "batch_size": 4,
            "lr": 1e-3,
            "steps": 2,
            "warmup": 1,
            "unroll": [[0, 1]],
            "backprop": "truncated",
            "workers": 0,
        },
        "eval": {"every": 10, "val_batches": 1, "rollout_steps": 4, "rollout_starts": 1, "sampling_steps": 2},
        "out_dir": str(run_dir),
    }
    cfg_file = tmp_path / "train_mol.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert cli_main(["train", "--config", str(cfg_file), "--device", "cpu", "--no_tensorboard"]) == 0

    eval_out = tmp_path / "eval_mol"
    assert cli_main([
        "evaluate-electrolyte",
        "--run", str(run_dir),
        "--ckpt", "ckpt.pt",
        "--steps", "8",
        "--starts", "2",
        "--seeds", "2",
        "--sampling_steps", "2",
        "--exp_sigma", "8.60",
        "--out", str(eval_out),
        "--device", "cpu",
    ]) == 0
    res_json = json.loads((eval_out / "results_wignerflow.json").read_text(encoding="utf-8"))
    assert "conductivity_total [mS/cm]" in res_json["system_properties"]
    assert "conductivity_self [mS/cm]" in res_json["system_properties"]
    assert {c["category"] for c in res_json["cluster"]} == {"SSIP", "CIP", "AGG"}
    assert "vs_experiment" in res_json

    cal_out = tmp_path / "calibrated_mol"
    assert cli_main([
        "calibrate-electrolyte",
        "--run", str(run_dir),
        "--ckpt", "ckpt.pt",
        "--out_dir", str(cal_out),
        "--target_sigma", "8.60",
        "--steps", "3",
        "--batch_size", "4",
        "--unroll_steps", "2",
        "--sampling_steps", "2",
        "--device", "cpu",
    ]) == 0
    assert (cal_out / "best_stable.pt").is_file()


def test_napf6_ec_dmc_h100_configs_instantiate():
    cfg_root = Path(__file__).resolve().parent.parent / "configs" / "napf6_ec_dmc"
    yamls = sorted(cfg_root.glob("*.yaml"))
    assert len(yamls) == 4
    for ypath in yamls:
        cfg = yaml.safe_load(ypath.read_text(encoding="utf-8"))
        mcfg = dict(cfg["model"])
        if mcfg.get("n_ff_types") == "auto":
            mcfg["n_ff_types"] = 6
        model = make_model(n_types=4, k=cfg["k"], stride_fs=200.0, **mcfg)
        assert sum(p.numel() for p in model.parameters()) > 0


@pytest.mark.skipif(not all(d.is_dir() for d in REAL_DIRS), reason="Local 02_Na_MD build_outputs not present")
def test_real_best_napf6_ec_dmc_m4a_builds():
    for bdir in REAL_DIRS:
        wmi = load_wmi_build_dir(bdir)
        assert wmi["n_total_sites"] == 8264
        assert wmi["n_virtual_sites"] == 2120
        assert wmi["n_phys_atoms"] == 6144
        assert wmi["n_molecules"] == 610
        assert np.isclose(wmi["temperature_K"], 298.15)

        sp = wmi["species_table"]
        assert sp["MOL1"]["species"] == "S00" and sp["MOL1"]["count"] == 268
        assert sp["MOL2"]["species"] == "S01" and sp["MOL2"]["count"] == 262
        assert sp["MOL3"]["species"] == "cation" and sp["MOL3"]["count"] == 40
        assert sp["MOL4"]["species"] == "anion" and sp["MOL4"]["count"] == 40
        assert abs(sp["MOL1"]["charge"]) < 1e-6
        assert abs(sp["MOL2"]["charge"]) < 1e-6
        assert np.isclose(sp["MOL3"]["charge"], 1.0)
        assert np.isclose(sp["MOL4"]["charge"], -1.0)
        assert abs(float(wmi["phys_charges"].sum())) < 1e-5

    # Check reduction shapes & equilibrium box match against `02_Na_MD/.../results_mu.json`
    w0 = load_wmi_build_dir(REAL_DIRS[0])
    pos2 = np.stack([w0["pdb_positions_A"], w0["pdb_positions_A"] + 0.01], axis=0)
    boxes2 = np.broadcast_to(w0["pdb_box_A"], (2, 3, 3)).copy()
    assert reduce_wmi_frames(pos2, None, boxes2, w0, resolution="molecular")["positions_A"].shape == (2, 610, 3)
    assert reduce_wmi_frames(pos2, None, boxes2, w0, resolution="polar_beads")["positions_A"].shape == (2, 1140, 3)
    assert reduce_wmi_frames(pos2, None, boxes2, w0, resolution="all_atom")["positions_A"].shape == (2, 6144, 3)

    if REAL_RESULTS_MU.is_file():
        ref_mu = json.loads(REAL_RESULTS_MU.read_text(encoding="utf-8"))
        assert abs(w0["equilibrium_box_L_A"] - float(ref_mu["cubic_box_length [angstrom]"])) < 0.02
