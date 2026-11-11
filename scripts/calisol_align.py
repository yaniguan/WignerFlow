"""Standalone CLI for CALiSol-23 periodic box scaling & WignerFlow-Align transport post-training.

Examples:
    python scripts/calisol_align.py --mode box-study
    python scripts/calisol_align.py --mode align-eval --calisol-csv "CALiSol-23 Dataset.csv"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from equitraj.align import (
    _NE_PREFACTOR_MS_CM,
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
)
from equitraj.fast import FastRollout
from equitraj.model import EquiTraj


def run_box_study() -> list[dict]:
    """Compare periodic box sizes (N_IP = 2..80) against 10k-atom 12h H200 OpenMM PFF baseline."""
    desc = compute_formulation_descriptors(
        salt="LiPF6", solvent_ratios={"EC": 0.5, "DMC": 0.5},
        ratio_type="w", c_val=1.0, c_units="mol/l", temperature_k=298.15,
    )
    return [analyze_electrolyte_box(desc, n_ion_pairs=nip) for nip in (2, 5, 10, 20, 32, 80)]


def train_and_evaluate_calisol(csv_path: str | Path, *, panel_size: int = 72, seed: int = 42) -> dict:
    """Compare raw unaligned WignerFlow rollout vs. Stage-1 + Stage-2 WignerFlow-Align on CALiSol-23."""
    t0 = time.perf_counter()
    records = load_calisol23(csv_path)
    train_recs, test_recs = split_calisol_grouped(records, test_fraction=0.20, seed=seed)

    rng = np.random.default_rng(seed)
    ref_windows = []
    train_pick = [train_recs[i] for i in rng.choice(len(train_recs), size=min(6, len(train_recs)), replace=False)]
    for idx_w, rec in enumerate(train_pick):
        bd = build_periodic_electrolyte_box(rec, n_ion_pairs=4, ion_charge_scale=0.80, seed=seed + idx_w)
        tx, tv = run_periodic_langevin_window(bd, n_frames=12, substeps=10, dt_fs=2.5, seed=seed + idx_w)
        ref_windows.append({
            "positions": tx,
            "velocities": tv,
            "types": bd["types"],
            "masses": bd["masses"],
            "charges": bd["charges"],
            "box": np.eye(3, dtype=np.float32) * bd["box_length"],
            "ion_indices": np.concatenate([bd["cation_indices"], bd["anion_indices"]]),
        })

    torch.manual_seed(seed)
    base_model = EquiTraj(
        n_types=5, channels=32, lmax=1, heads=4, n_rbf=20, cutoff=4.8, k=3,
        n_spatial=2, n_fusion=1, n_denoiser=1, head="regression", stride_fs=25.0, use_charges=True,
    ).float()
    base_model.dx_std.fill_(0.06)
    base_model.v_std.fill_(0.004)
    base_model.out_dx_std.fill_(0.025)
    base_model.out_v_std.fill_(0.0025)

    opt_s1 = torch.optim.AdamW(base_model.parameters(), lr=2e-3)
    base_model.train()
    for s_idx in range(25):
        w = ref_windows[s_idx % len(ref_windows)]
        t_i = int(rng.integers(0, w["positions"].shape[0] - 4))
        px = torch.tensor(w["positions"][t_i:t_i + 3], dtype=torch.float32)[None]
        pv = torch.tensor(w["velocities"][t_i:t_i + 3], dtype=torch.float32)[None]
        tx_t = torch.tensor(w["positions"][t_i + 3] - w["positions"][t_i + 2], dtype=torch.float32)[None]
        tv_t = torch.tensor(0.72 * w["velocities"][t_i + 3], dtype=torch.float32)[None]
        loss = base_model.loss(
            px, pv, torch.tensor(w["types"], dtype=torch.long),
            torch.tensor(w["box"], dtype=torch.float32)[None],
            torch.tensor(w["masses"], dtype=torch.float32),
            tx_t, tv_t, charges=torch.tensor(w["charges"], dtype=torch.float32),
        )
        opt_s1.zero_grad()
        loss.backward()
        opt_s1.step()
    base_model.eval()

    pt_res = posttrain_equitraj_transport(
        base_model, ref_windows, steps=36, unroll_m=3, sampling_steps=1, lr=1.2e-3, seed=seed
    )
    post_model = pt_res["model"]

    order_test = np.argsort([r["sigma_exp_mS_cm"] for r in test_recs])
    stride_pick = np.linspace(0, len(order_test) - 1, min(panel_size, len(order_test)), dtype=int)
    eval_panel = [test_recs[int(order_test[i])] for i in stride_pick]

    fast_raw = FastRollout(base_model, sampling_steps=1, neighbor="verlet", skin=0.8)
    fast_post = FastRollout(post_model, sampling_steps=1, neighbor="verlet", skin=0.8)

    raw_ne_preds, raw_eh_preds, stage1_yh_preds = [], [], []
    raw_t_errs, post_t_errs = [], []
    panel_post_obs = []

    for i_p, rec in enumerate(eval_panel):
        bd_raw = build_periodic_electrolyte_box(rec, n_ion_pairs=5, ion_charge_scale=1.0, seed=seed + i_p)
        x0 = torch.tensor(bd_raw["positions"], dtype=torch.float64)[None, None].expand(1, 3, -1, 3).clone()
        v0 = torch.tensor(bd_raw["velocities"], dtype=torch.float32)[None, None].expand(1, 3, -1, 3).clone()
        tp = torch.tensor(bd_raw["types"], dtype=torch.long)
        ms = torch.tensor(bd_raw["masses"], dtype=torch.float32)
        bx = (torch.eye(3, dtype=torch.float32) * bd_raw["box_length"])[None]

        fast_raw.model.set_chemistry(charges=torch.tensor(bd_raw["charges"], dtype=torch.float32))
        tr_raw = fast_raw.rollout(x0, v0, tp, bx, ms, n_steps=24)
        obs_raw = extract_rollout_transport_observables(
            tr_raw["positions"][:, 0].numpy(), tr_raw["velocities"][:, 0].numpy(),
            bd_raw["box_length"], bd_raw["masses"], bd_raw["charges"],
            bd_raw["cation_indices"], bd_raw["anion_indices"], bd_raw["solvent_o_indices"],
            dt_fs=25.0, temperature_k=rec["temperature_k"],
        )
        raw_ne_preds.append(obs_raw["sigma_ne_raw_mS_cm"])
        raw_eh_preds.append(obs_raw["sigma_eh_raw_mS_cm"])
        raw_t_errs.append(obs_raw["t_rel_err"])

        q_ecc = 1.0 / math.sqrt(max(1.5, rec["eps_inf_mix"]))
        bd_post = build_periodic_electrolyte_box(rec, n_ion_pairs=5, ion_charge_scale=q_ecc, seed=seed + i_p)
        fast_post.model.set_chemistry(charges=torch.tensor(bd_post["charges"], dtype=torch.float32))
        tr_post = fast_post.rollout(x0, v0, tp, bx, ms, n_steps=24)
        obs_post = extract_rollout_transport_observables(
            tr_post["positions"][:, 0].numpy(), tr_post["velocities"][:, 0].numpy(),
            bd_post["box_length"], bd_post["masses"], bd_post["charges"],
            bd_post["cation_indices"], bd_post["anion_indices"], bd_post["solvent_o_indices"],
            dt_fs=25.0, temperature_k=rec["temperature_k"],
        )
        panel_post_obs.append(obs_post)
        post_t_errs.append(obs_post["t_rel_err"])

        box_an = analyze_electrolyte_box(rec, n_ion_pairs=5)
        d_yh = box_an["yeh_hummer_delta_D_cm2_s"]
        vol_a3 = bd_post["box_length"] ** 3
        s1_yh = _NE_PREFACTOR_MS_CM * obs_post["ionicity_cne"] * (
            5 * (obs_post["d_cat_pbc_cm2_s"] + d_yh) + 5 * (obs_post["d_an_pbc_cm2_s"] + d_yh)
        ) / (vol_a3 * rec["temperature_k"])
        stage1_yh_preds.append(float(s1_yh))

    train_sub_idx = rng.choice(len(train_recs), size=min(48, len(train_recs)), replace=False)
    train_obs_list: list[dict | None] = [None] * len(train_recs)
    for ti in train_sub_idx:
        rec_t = train_recs[int(ti)]
        q_ecc_t = 1.0 / math.sqrt(max(1.5, rec_t["eps_inf_mix"]))
        bd_t = build_periodic_electrolyte_box(rec_t, n_ion_pairs=4, ion_charge_scale=q_ecc_t, seed=seed + int(ti))
        x_t = torch.tensor(bd_t["positions"], dtype=torch.float64)[None, None].expand(1, 3, -1, 3).clone()
        v_t = torch.tensor(bd_t["velocities"], dtype=torch.float32)[None, None].expand(1, 3, -1, 3).clone()
        fast_post.model.set_chemistry(charges=torch.tensor(bd_t["charges"], dtype=torch.float32))
        tr_t = fast_post.rollout(
            x_t, v_t, torch.tensor(bd_t["types"], dtype=torch.long),
            (torch.eye(3, dtype=torch.float32) * bd_t["box_length"])[None],
            torch.tensor(bd_t["masses"], dtype=torch.float32), n_steps=20,
        )
        train_obs_list[int(ti)] = extract_rollout_transport_observables(
            tr_t["positions"][:, 0].numpy(), tr_t["velocities"][:, 0].numpy(),
            bd_t["box_length"], bd_t["masses"], bd_t["charges"],
            bd_t["cation_indices"], bd_t["anion_indices"], bd_t["solvent_o_indices"],
            dt_fs=25.0, temperature_k=rec_t["temperature_k"],
        )

    aligner, _ = fit_calisol_aligner(
        train_recs, train_traj_obs=train_obs_list, n_heads=4, hidden=32, epochs=120, batch_size=256, seed=seed
    )

    panel_aligned = aligner.predict_records(eval_panel, panel_post_obs)
    full_test_aligned = aligner.predict_records(test_recs, None)

    y_panel = np.array([r["sigma_exp_mS_cm"] for r in eval_panel])
    y_full_test = np.array([r["sigma_exp_mS_cm"] for r in test_recs])

    conc_grid = [0.15, 0.4, 0.7, 1.0, 1.2, 1.6, 2.0, 2.5, 3.2]
    conc_recs = [
        compute_formulation_descriptors("LiPF6", {"EC": 0.5, "DMC": 0.5}, "w", c, "mol/l", 298.15)
        for c in conc_grid
    ]
    conc_preds = aligner.predict_records(conc_recs)["sigma_pred_mS_cm"]

    return {
        "dataset_summary": {
            "total_clean_rows": len(records),
            "train_rows": len(train_recs),
            "heldout_test_rows": len(test_recs),
            "rollout_panel_rows": len(eval_panel),
        },
        "panel_metrics": {
            "raw_unaligned_ne": regression_metrics(y_panel, np.asarray(raw_ne_preds)),
            "raw_unaligned_eh": regression_metrics(y_panel, np.asarray(raw_eh_preds)),
            "stage1_posttrain_plus_yh": regression_metrics(y_panel, np.asarray(stage1_yh_preds)),
            "wignerflow_align_stage1_stage2": regression_metrics(y_panel, panel_aligned["sigma_pred_mS_cm"]),
            "thermal_drift_rel_err_before": round(float(np.mean(raw_t_errs)), 4),
            "thermal_drift_rel_err_after": round(float(np.mean(post_t_errs)), 4),
        },
        "full_heldout_test_metrics": regression_metrics(y_full_test, full_test_aligned["sigma_pred_mS_cm"]),
        "concentration_bell_curve_ec_dmc_298K": [
            {"c_M": c, "sigma_aligned_mS_cm": round(float(s), 2)} for c, s in zip(conc_grid, conc_preds)
        ],
        "wall_s": round(time.perf_counter() - t0, 2),
        "_artifacts": (aligner, post_model),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["box-study", "align-eval"], default="box-study")
    ap.add_argument("--calisol-csv", help="Path to CALiSol-23 Dataset.csv")
    ap.add_argument("--json-out", help="Optional output path for JSON benchmark report")
    args = ap.parse_args(argv)

    report: dict = {"mode": args.mode}
    if args.mode == "box-study":
        report["box_study"] = run_box_study()
    else:
        if not args.calisol_csv:
            ap.error("--calisol-csv is required for align-eval")
        cal_res = train_and_evaluate_calisol(args.calisol_csv)
        cal_res.pop("_artifacts", None)
        report["calisol_alignment"] = cal_res

    text = json.dumps(report, indent=2)
    if args.json_out:
        out_p = Path(args.json_out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
