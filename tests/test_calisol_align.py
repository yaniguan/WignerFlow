"""Unit tests for CALiSol-23 periodic box scaling and WignerFlow-Align (`equitraj/align.py`)."""
import csv
from pathlib import Path

import numpy as np
import pytest
import torch

from equitraj.align import (
    analyze_electrolyte_box,
    build_periodic_electrolyte_box,
    compute_formulation_descriptors,
    extract_rollout_transport_observables,
    fit_calisol_aligner,
    load_calisol23,
    posttrain_equitraj_transport,
    regression_metrics,
    run_periodic_langevin_window,
    split_calisol_grouped,
    yeh_hummer_correction_cm2_s,
)
from equitraj.model import EquiTraj


def test_yeh_hummer_scaling_and_box_analysis():
    d20 = yeh_hummer_correction_cm2_s(20.0, 298.15, 2.0)
    d40 = yeh_hummer_correction_cm2_s(40.0, 298.15, 2.0)
    assert np.isclose(d20, 2.0 * d40, rtol=1e-6)
    with pytest.raises(ValueError):
        yeh_hummer_correction_cm2_s(0.0, 298.15, 2.0)

    desc = compute_formulation_descriptors("LiPF6", {"EC": 0.5, "DMC": 0.5}, "w", 1.0, "mol/l", 298.15)
    b_small = analyze_electrolyte_box(desc, n_ion_pairs=4)
    b_sweet = analyze_electrolyte_box(desc, n_ion_pairs=32)
    assert b_sweet["box_length_A"] > b_small["box_length_A"]
    assert b_sweet["uncorrected_pbc_drag_percent"] < b_small["uncorrected_pbc_drag_percent"]
    assert b_sweet["min_image_valid"] is True
    assert b_sweet["speedup_vs_openmm_pff_12h"] > 1000


def test_periodic_box_langevin_and_stage1_transport_posttraining():
    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        desc = compute_formulation_descriptors("LiFSI", {"EC": 0.3, "EMC": 0.7}, "w", 1.2, "mol/kg", 298.15)
        bd = build_periodic_electrolyte_box(desc, n_ion_pairs=3, ion_charge_scale=0.8, seed=7)
        n = len(bd["masses"])
        assert bd["positions"].shape == (n, 3)
        assert abs(bd["charges"].sum()) < 1e-6

        traj_x, traj_v = run_periodic_langevin_window(bd, n_frames=8, substeps=4, dt_fs=2.0, seed=7)
        assert traj_x.shape == (8, n, 3)
        assert np.isfinite(traj_x).all()

        obs = extract_rollout_transport_observables(
            traj_x[1:], traj_v[1:], bd["box_length"], bd["masses"], bd["charges"],
            bd["cation_indices"], bd["anion_indices"], bd["solvent_o_indices"], dt_fs=8.0,
        )
        assert obs["sigma_ne_raw_mS_cm"] >= 0.0
        assert 0.0 < obs["ionicity_cne"] <= 1.0

        model = EquiTraj(
            n_types=5, channels=16, lmax=1, heads=2, n_rbf=12, cutoff=4.5, k=3,
            n_spatial=1, n_fusion=1, n_denoiser=1, head="regression", stride_fs=25.0, use_charges=True,
        ).float()
        model.dx_std.fill_(0.05)
        model.v_std.fill_(0.004)
        model.out_dx_std.fill_(0.02)
        model.out_v_std.fill_(0.002)

        win = {
            "positions": traj_x,
            "velocities": traj_v,
            "types": bd["types"],
            "masses": bd["masses"],
            "charges": bd["charges"],
            "box": np.eye(3, dtype=np.float32) * bd["box_length"],
            "ion_indices": np.concatenate([bd["cation_indices"], bd["anion_indices"]]),
        }
        pt = posttrain_equitraj_transport(model, [win], steps=8, unroll_m=2, lr=2e-3, seed=3)
        assert len(pt["history"]) == 8
        assert np.isfinite(pt["history"][-1]["total_loss"])
        assert pt["history"][-1]["total_loss"] < pt["history"][0]["total_loss"]
    finally:
        torch.set_default_dtype(prev_dtype)


def test_calisol_loader_grouped_split_aligner_and_cli(tmp_path: Path):
    csv_file = tmp_path / "mini_calisol.csv"
    fieldnames = ["doi", "k", "T", "c", "salt", "c units", "solvent ratio type", "EC", "DMC", "EMC", "PC", "AN"]
    rows = []
    for g_idx, (salt, s_map, scale) in enumerate([
        ("LiPF6", {"EC": 0.5, "DMC": 0.5}, 1.00),
        ("LiPF6", {"EC": 0.3, "EMC": 0.7}, 0.88),
        ("LiBF4", {"PC": 0.5, "DMC": 0.5}, 0.58),
        ("LiFSI", {"AN": 0.4, "DMC": 0.6}, 1.35),
    ]):
        for c_val in (0.2, 0.6, 1.0, 1.5, 2.4):
            for temp in (263.15, 298.15, 323.15):
                sig = scale * 16.5 * c_val * np.exp(-0.78 * c_val) * np.exp(-1450.0 * (1.0 / temp - 1.0 / 298.15))
                r = {k: "" for k in fieldnames}
                r.update({
                    "doi": f"10.1000/test.{g_idx}", "k": f"{sig:.4f}", "T": f"{temp:.2f}",
                    "c": f"{c_val:.2f}", "salt": salt, "c units": "mol/l", "solvent ratio type": "w",
                })
                for sk, sv in s_map.items():
                    r[sk] = str(sv)
                rows.append(r)
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    recs = load_calisol23(csv_file)
    assert len(recs) == len(rows)
    tr, te = split_calisol_grouped(recs, test_fraction=0.25, seed=1)
    assert set(r["group_key"] for r in tr).isdisjoint(set(r["group_key"] for r in te))

    aligner, losses = fit_calisol_aligner(tr, n_heads=3, hidden=16, epochs=45, batch_size=32, seed=1)
    assert losses[-1] < losses[0]
    pred_te = aligner.predict_records(te)
    m = regression_metrics([r["sigma_exp_mS_cm"] for r in te], pred_te["sigma_pred_mS_cm"])
    assert m["spearman_rho"] > 0.75

    from scripts.calisol_align import main as cli_main
    out_json = tmp_path / "box_report.json"
    cli_main(["--mode", "box-study", "--json-out", str(out_json)])
    assert out_json.exists()
