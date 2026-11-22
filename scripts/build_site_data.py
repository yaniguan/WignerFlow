"""Build site/assets/homepage-data.json and wignerflow-poster.png from committed results/ and checkpoints/.

Precomputes real snapshots for every interactive Studio preset, CPU inference API setting,
Lab surrogate query, and DistanceProjector constraint run so https://yaniguan.github.io/WignerFlow/
works 100% interactively with zero backend server, while local `scripts/wignerflow_studio.py`
also executes live PyTorch requests.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from equitraj import analysis, bench, pocket  # noqa: E402
from equitraj.api import ArgonPredictor  # noqa: E402
from equitraj.compliance import AUDIT_TARGETS, FAULT_CATALOG, run_compliance_audit  # noqa: E402
from equitraj.constraints import DistanceProjector  # noqa: E402
from equitraj.model import EquiTraj  # noqa: E402
from equitraj.rollout import rollout, temperature  # noqa: E402
from equitraj.studio_data import DEFAULT_ETHANOL_XYZ, load_benchmark_suites, parse_xyz_text, run_studio_rollout  # noqa: E402
from lab import surrogate_query as sq  # noqa: E402


def _read_json(rel: str):
    p = ROOT / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def build_argon_seeds_panel() -> dict:
    gt = _read_json("results/argon/ground_truth/metrics.json")
    t_gt = float(gt["T_mean_K"])
    seeds = []
    colors = ["#0d9488", "#e11d48", "#2563eb", "#7c3aed", "#d97706"]
    for idx in range(1, 6):
        run_dir = ROOT / "results" / "argon" / f"B_S3a_seed{idx}"
        log_path = run_dir / "log.jsonl"
        eval_data = _read_json(f"results/argon/B_S3a_seed{idx}/eval/metrics.json") or {}
        pts = []
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if "val_T_end" in row and row["val_T_end"] is not None:
                pts.append([int(row["step"]), round(float(row["val_T_end"]), 2)])
            else:
                pts.append([int(row["step"]), None])
        t_test = (eval_data.get("temperature") or {}).get("T_model_second_half_K")
        crashed = int((eval_data.get("stability") or {}).get("crashed", 0))
        n_roll = int(eval_data.get("n_rollouts", 15))
        seeds.append({
            "name": f"Seed {idx}" + (" (all v0 pass)" if idx in (3, 4) else (" (test crash ~4k)" if idx == 2 else "")),
            "key": f"seed{idx}",
            "color": colors[idx - 1],
            "dash": [6, 5] if idx == 2 else [],
            "highlight": idx in (3, 4),
            "points": pts,
            "test_step": eval_data.get("train_step"),
            "test_T_K": None if t_test is None else round(float(t_test), 2),
            "survived": f"{n_roll - crashed}/{n_roll}",
            "all_pass": bool((eval_data.get("v0_verdict") or {}).get("all_pass", False)),
        })
    rel_eval = _read_json("results/argon/B_S3a_seed1/eval_step10000/metrics.json") or {}
    return {
        "t_gt_K": round(t_gt, 2),
        "band_low_K": round(t_gt * 0.95, 2),
        "band_high_K": round(t_gt * 1.05, 2),
        "released_step10000_T_K": round(float(rel_eval["temperature"]["T_model_second_half_K"]), 2),
        "series": seeds,
    }


def build_argon_rollout_and_curves() -> dict:
    ckpt_path = ROOT / "checkpoints" / "argon_flow_v0.pt"
    torch.set_default_dtype(torch.float32)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = EquiTraj(
        n_types=len(ck["elements"]),
        k=ck["k"],
        stride_fs=ck["stride_fs"],
        **ck["model_cfg"],
    ).eval()
    model.load_state_dict(ck["model"])

    demo = ck["demo"]
    ref = ck["reference"]
    box = demo["box"][None]
    box_l = float(demo["box"][0, 0].item())

    torch.manual_seed(0)
    n_steps = 28
    out = rollout(
        model,
        demo["pos"][None],
        demo["vel"][None],
        ck["types"],
        box,
        ck["masses"],
        n_steps,
        sampling_steps=6,
        t_max=3.0 * float(ref["T_K"]),
    )
    ctx_pos = demo["pos"].numpy() % box_l
    fut_pos = out["positions"][:, 0].numpy() % box_l
    fut_vel = out["velocities"][:, 0].numpy()

    r_ref = np.asarray(ref["rdf_r"], dtype=np.float64)
    g_ref = np.asarray(ref["rdf_g"], dtype=np.float64)
    idx = np.arange(fut_pos.shape[1])
    frames_rdf = fut_pos[8:]
    _, g_mod, _ = analysis.rdf(
        frames_rdf,
        np.tile(demo["box"].numpy(), (len(frames_rdf), 1, 1)),
        idx,
        idx,
        float(r_ref[-1]) + 1e-6,
        len(r_ref),
    )

    lag_fs = np.asarray(ref["vacf_lag_fs"], dtype=np.float64)
    vacf_ref = np.asarray(ref["vacf"], dtype=np.float64)
    all_vel = np.concatenate([demo["vel"].numpy(), fut_vel], axis=0)
    max_lag = min(len(lag_fs), len(all_vel) - 4)
    c_mod = analysis.vacf(all_vel, max_lag)
    c_mod = (c_mod / max(abs(c_mod[0]), 1e-12)).tolist()

    temp_trace = temperature(torch.from_numpy(fut_vel), ck["masses"]).numpy()

    center = np.array([box_l * 0.5, box_l * 0.5, box_l * 0.5])
    d_center = np.linalg.norm(ctx_pos[-1] - center[None, :], axis=-1)
    keep = np.argsort(d_center)[:96]
    all_frames_sub = np.concatenate([ctx_pos[:, keep], fut_pos[:, keep]], axis=0)

    step_r = max(1, len(r_ref) // 100)
    return {
        "box_A": round(box_l, 3),
        "stride_fs": float(ck["stride_fs"]),
        "n_context": 3,
        "n_rollout": int(n_steps),
        "n_atoms_total": 256,
        "n_atoms_cinema": int(len(keep)),
        "t_ref_K": round(float(ref["T_K"]), 2),
        "frames": np.round(all_frames_sub, 3).tolist(),
        "temp_K": np.round(temp_trace, 2).tolist(),
        "rdf": {
            "r_A": np.round(r_ref[::step_r], 3).tolist(),
            "gt": np.round(g_ref[::step_r], 4).tolist(),
            "model": np.round(g_mod[::step_r], 4).tolist(),
        },
        "vacf": {
            "t_fs": np.round(lag_fs[:40], 1).tolist(),
            "gt": np.round(vacf_ref[:40], 4).tolist(),
            "model": [round(float(v), 4) for v in c_mod[: min(40, len(c_mod))]],
        },
    }


def build_cutoff_panel() -> dict:
    old = _read_json("results/bench/cutoff_continuity.json") or {"rows": []}
    spacings = [1e-2, 1e-3, 1e-4]
    v0_jump, ctrl_step = [], []
    for h in spacings:
        cut_rows = [
            r["max_crossing_step"]
            for r in old["rows"]
            if r.get("scan") == "cutoff" and r.get("head") == "regression" and np.isclose(r.get("h", 0.0), h)
        ]
        ctl_rows = [
            r["max_smooth_step"]
            for r in old["rows"]
            if r.get("scan") == "control" and r.get("head") == "regression" and np.isclose(r.get("h", 0.0), h)
        ]
        v0_jump.append(float(np.mean(cut_rows)) if cut_rows else None)
        ctrl_step.append(float(np.mean(ctl_rows)) if ctl_rows else None)

    torch.set_default_dtype(torch.float64)
    v1_jump = []
    model_f64 = bench.make_model(head="regression", dtype=torch.float64, channels=16, heads=4)
    base_probe = bench.make_cutoff_probe(dtype=torch.float64)
    for h in spacings:
        radii = torch.linspace(6.0 - 20.5 * h, 6.0 + 18.5 * h, 40, dtype=torch.float64)
        outs, n_edges = bench.scan_outputs(model_f64, base_probe, radii, n_steps=2)
        res = bench.jump_stats(outs, n_edges)
        v1_jump.append(float(res["max_crossing_step"]))
    torch.set_default_dtype(torch.float32)

    return {
        "spacings_A": spacings,
        "v0_post_softmax_jump": [None if x is None else float(f"{x:.4e}") for x in v0_jump],
        "v1_log_envelope_sink_jump": [float(f"{x:.4e}") for x in v1_jump],
        "interior_control_step": [None if x is None else float(f"{x:.4e}") for x in ctrl_step],
    }


def build_ablation_summary() -> list[dict]:
    entries = [
        ("Flow · Residual (Seed 4)", "results/argon/B_S3a_seed4/eval/metrics.json", "flow", "residual"),
        ("Flow · Residual (Seed 3)", "results/argon/B_S3a_seed3/eval/metrics.json", "flow", "residual"),
        ("Flow · Residual (v0 release, s10k)", "results/argon/B_S3a_seed1/eval_step10000/metrics.json", "flow", "residual"),
        ("Flow · Direct target", "results/argon/B_S3a_direct/eval/metrics.json", "flow", "direct"),
        ("Regression · Residual (Seed 1)", "results/argon/A_S3a_seed1/eval/metrics.json", "regression", "residual"),
        ("Regression · Direct target", "results/argon/A_S3a_direct/eval/metrics.json", "regression", "direct"),
    ]
    rows = []
    for label, rel, head, target in entries:
        m = _read_json(rel)
        if not m:
            continue
        t_m = m["temperature"]["T_model_second_half_K"]
        t_gt = m["T_ground_truth_K"]
        rows.append({
            "label": label,
            "head": head,
            "target": target,
            "train_step": m["train_step"],
            "survived": f"{m['n_rollouts'] - m['stability']['crashed']}/{m['n_rollouts']}",
            "T_model_K": round(float(t_m), 1),
            "T_gt_K": round(float(t_gt), 1),
            "T_signed_pct": round(100.0 * (float(t_m) - float(t_gt)) / float(t_gt), 1),
            "rdf_pos_err_A": round(float(m["rdf"]["peak_pos_err_A"]), 3),
            "vacf_err_pct": round(100.0 * float(m["vacf"]["first_zero_rel_err"]), 2),
            "D_err_pct": round(100.0 * float(m["diffusion"]["rel_err"]), 1),
            "all_pass": bool(m["v0_verdict"]["all_pass"]),
            "source": rel,
        })
    return rows


def build_cpu_api_panel() -> dict:
    raw = _read_json("results/cpu_api/bench.json") or {}
    csvr_s1 = _read_json("results/cpu_api/bench_csvr95_s1.json") or {}
    csvr_s2 = _read_json("results/cpu_api/bench_csvr95_s2.json") or {}
    lat = raw.get("latency_ms", {})
    one = raw.get("one_step", {})
    roll = raw.get("rollouts", {})

    def _mean_rollout(runs_dict):
        runs = (runs_dict or {}).get("runs") or []
        if not runs:
            return None
        return {
            "T_second_half_K": round(float(np.mean([r["T_second_half_K"] for r in runs])), 1),
            "rdf_peak_pos_err_A": round(float(np.mean([r["rdf_peak_pos_err_A"] for r in runs])), 3),
            "rdf_peak_height_pct": round(100.0 * float(np.mean([r["rdf_peak_height_rel_err"] for r in runs])), 1),
            "survived": f"{sum(1 for r in runs if r['crash_step'] < 0)}/{len(runs)}",
        }

    grid = {}
    for n_atoms in (108, 256):
        for s in (1, 2, 4, 10):
            key = f"N={n_atoms},S={s}"
            entry = lat.get(key, {})
            rms = one.get(f"S={s}_vs_S=10_rel_rms", {"dx": 0.0, "v": 0.0})
            r_off = _mean_rollout(roll.get(f"S={s}")) if n_atoms == 256 else None
            r_csvr = None
            if n_atoms == 256:
                if s == 1:
                    r_csvr = _mean_rollout(
                        (csvr_s1.get("rollouts") or {}).get("S=1,CSVR 95 K")
                        or (csvr_s1.get("rollouts") or {}).get("S=1")
                    )
                elif s == 2:
                    r_csvr = _mean_rollout(
                        (csvr_s2.get("rollouts") or {}).get("S=2,CSVR 95 K")
                        or (csvr_s2.get("rollouts") or {}).get("S=2")
                    )
            grid[f"{n_atoms}_{s}"] = {
                "n_atoms": n_atoms,
                "sampling_steps": s,
                "wall_p50_ms": round(float(entry.get("wall", {}).get("p50", 0.0)), 1),
                "wall_p95_ms": round(float(entry.get("wall", {}).get("p95", 0.0)), 1),
                "cpu_p50_ms": round(float(entry.get("cpu", {}).get("p50", 0.0)), 1),
                "cpu_p95_ms": round(float(entry.get("cpu", {}).get("p95", 0.0)), 1),
                "dx_rel_rms_pct": round(100.0 * float(rms.get("dx", 0.0)), 2),
                "v_rel_rms_pct": round(100.0 * float(rms.get("v", 0.0)), 2),
                "rollout_off": r_off,
                "rollout_csvr95": r_csvr,
            }

    ref_old = lat.get("N=256,S=10,EquiTraj.predict (reference path)", {})
    pred = ArgonPredictor(threads=1, sampling_steps=2)
    demo = pred.ck["demo"]
    sample = pred.predict(demo["pos"], demo["vel"], demo["box"], n_steps=1, sampling_steps=2, seed=0)
    sample_preview = {
        "n_atoms": 256,
        "time_fs": sample["time_fs"],
        "temperature_K": [round(float(x), 2) for x in sample["temperature_K"]],
        "input_temperature_K": round(float(sample["input_temperature_K"]), 2),
        "density_g_cm3": round(float(sample["density_g_cm3"]), 4),
        "crash_step": sample["crash_step"],
        "warnings": sample["warnings"],
        "latency_ms": round(float(sample["latency_ms"]), 1),
        "model": sample["model"],
        "positions_sample_A": [[round(float(c), 3) for c in atom] for atom in sample["positions"][0][:2]],
    }

    return {
        "cpu_model": raw.get("cpu", "Intel(R) Xeon(R) Processor"),
        "threads": raw.get("threads", 1),
        "budget_ms": 1000,
        "frame_split_max_abs_diff": float(one.get("frame_split_vs_EquiTraj_predict_max_abs_diff", 0.0)),
        "reference_batched_256_s10": {
            "wall_p50_ms": round(float(ref_old.get("wall", {}).get("p50", 2722.3)), 1),
            "cpu_p50_ms": round(float(ref_old.get("cpu", {}).get("p50", 2641.3)), 1),
        },
        "stage_breakdown_256_s2": [
            {"stage": "Spatial SO(2) (3 frames)", "ms": 551, "pct": 45, "color": "#0d9488"},
            {"stage": "Flow Denoiser (S = 2)", "ms": 399, "pct": 33, "color": "#7c3aed"},
            {"stage": "Fusion Attention", "ms": 189, "pct": 15, "color": "#2563eb"},
            {"stage": "Embed + Neighbor Graph", "ms": 79, "pct": 6, "color": "#64748b"},
        ],
        "grid": grid,
        "sample_response": sample_preview,
    }


def build_surrogate_lab_panel() -> dict:
    release = sq.load_release(ROOT / sq.REGISTRY["argon"]["checkpoint"], "cpu")
    cases = [
        (
            "argon_cryo_2step",
            "Argon 2-Step Sanity Check (In-Domain)",
            {"system": "argon", "n_steps": 2, "sampling_steps": 2, "seed": 0,
             "question": "Does liquid argon stay near 95 K with no atom overlap over two 50 fs steps?"},
        ),
        (
            "argon_half_ps",
            "Argon 10-Step Rollout (0.5 ps Check)",
            {"system": "argon", "n_steps": 10, "sampling_steps": 4, "seed": 0,
             "question": "Check mean temperature and closest pair distance over 0.5 ps."},
        ),
        (
            "ethanol_refusal",
            "Ethanol Query (Out-of-Domain Refusal)",
            {"system": "ethanol", "n_steps": 20,
             "question": "Not trained yet; must return unsupported instead of guessing."},
        ),
        (
            "aspirin_refusal",
            "Aspirin Query (Safety Gate Refusal)",
            {"system": "aspirin", "n_steps": 10,
             "question": "Refuse small molecules until a molecule checkpoint passes evaluation."},
        ),
    ]
    out = {}
    for key, label, q in cases:
        rep = sq.run_query(q, model_and_ckpt=release)
        out[key] = {
            "id": key,
            "label": label,
            "query": q,
            "report": rep,
            "context_lines": sq.to_reasoner_context(rep),
        }
    return out


def build_constraints_and_hybrid_panel() -> dict:
    """Run real DistanceProjector bond projections across stretch levels on Ethanol X-H bonds."""
    parsed = parse_xyz_text(DEFAULT_ETHANOL_XYZ)
    pos0 = torch.from_numpy(parsed["positions"]).double()
    masses = torch.from_numpy(parsed["masses"]).double()
    # Constrain the 6 X-H bonds of ethanol: C0-H3, C0-H4, C0-H5, C1-H6, C1-H7, O2-H8
    xh_pairs = [(0, 3), (0, 4), (0, 5), (1, 6), (1, 7), (2, 8)]
    constraints = []
    for i, j in xh_pairs:
        d0 = float(torch.linalg.norm(pos0[i] - pos0[j]).item())
        constraints.append([float(i), float(j), d0])
    proj = DistanceProjector(constraints, tolerance=1e-10, velocity_tolerance=1e-12)

    levels = []
    gen = torch.Generator().manual_seed(7)
    noise_dir = torch.randn(pos0.shape, generator=gen, dtype=torch.float64)
    noise_dir = noise_dir / noise_dir.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    vel0 = torch.randn(1, pos0.shape[0], 3, generator=gen, dtype=torch.float64) * 0.012

    for amp in (0.04, 0.10, 0.18, 0.26):
        x_pert = (pos0 + amp * noise_dir)[None]
        before_err = max(
            abs(float(torch.linalg.norm(x_pert[0, int(i)] - x_pert[0, int(j)]).item()) - r0)
            for i, j, r0 in constraints
        )
        com_before = (x_pert[0] * masses[:, None]).sum(0) / masses.sum()
        x_corr, v_corr, info = proj(x_pert, vel0, masses)
        com_after = (x_corr[0] * masses[:, None]).sum(0) / masses.sum()
        com_shift = float(torch.linalg.norm(com_after - com_before).item())
        bonds_detail = []
        for idx_b, (i, j, r0) in enumerate(constraints):
            lbl = f"{parsed['elements'][int(i)]}{int(i)}-{parsed['elements'][int(j)]}{int(j)}"
            d_pre = float(torch.linalg.norm(x_pert[0, int(i)] - x_pert[0, int(j)]).item())
            d_post = float(torch.linalg.norm(x_corr[0, int(i)] - x_corr[0, int(j)]).item())
            bonds_detail.append({
                "bond": lbl,
                "target_A": round(r0, 4),
                "perturbed_A": round(d_pre, 4),
                "projected_A": round(d_post, 4),
            })
        levels.append({
            "perturbation_A": amp,
            "max_bond_err_before_A": round(before_err, 4),
            "max_bond_err_after_A": float(f"{info['distance_residual_A']:.2e}"),
            "max_vel_err_after_A_fs": float(f"{info['velocity_residual_A_per_fs']:.2e}"),
            "iterations": int(info["iterations"]),
            "com_shift_A": float(f"{com_shift:.2e}"),
            "bonds": bonds_detail,
        })

    return {
        "molecule": "Ethanol (6 constrained C-H / O-H bonds)",
        "tolerance_A": 1e-10,
        "levels": levels,
    }


def _slim_compliance(rep: dict) -> dict:
    """Drop unused rollout trajectory frames on compliance snapshots to keep JSON compact."""
    out = dict(rep)
    out.pop("viewer_frames", None)
    out.pop("viewer_elements", None)
    out.pop("rollout_series", None)
    return out


def write_social_poster_png(path: Path):
    """Render a crisp 1200x630 PNG social card for OpenGraph / Twitter / GitHub previews."""
    w, h = 1200, 630
    img = Image.new("RGB", (w, h), (248, 250, 252))
    draw = ImageDraw.Draw(img)

    # Subtle soft porcelain radial tint
    for r in range(420, 20, -12):
        frac = 1.0 - r / 420.0
        col = (int(248 - 16 * frac), int(250 - 4 * frac), int(252 - 2 * frac))
        draw.ellipse((600 - r * 1.35, 250 - r * 0.82, 600 + r * 1.35, 250 + r * 0.82), fill=col)

    # Crisp border frame
    draw.rounded_rectangle((36, 36, w - 36, h - 36), radius=24, outline=(226, 232, 240), width=2)

    # Draw figure-eight orbital emblem
    pts = []
    for i in range(181):
        th = (i / 180.0) * 2.0 * np.pi
        px = 600.0 + 68.0 * np.cos(th)
        py = 148.0 + 23.0 * np.sin(2.0 * th)
        pts.append((px, py))
    draw.line(pts, fill=(13, 148, 136), width=5)
    draw.ellipse((663, 143, 673, 153), fill=(37, 99, 235))
    draw.ellipse((527, 143, 537, 153), fill=(124, 58, 237))

    def load_font(size: int, bold: bool = False):
        candidates = (
            ["segoeuib.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
        )
        for name in candidates:
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default()

    f_hero = load_font(64, bold=True)
    f_sub = load_font(50, bold=True)
    f_deck = load_font(23, bold=False)

    t1 = "WignerFlow: "
    t2 = "Integrator-Free"
    w1 = draw.textlength(t1, font=f_hero)
    w2 = draw.textlength(t2, font=f_hero)
    x0 = (w - (w1 + w2)) * 0.5
    draw.text((x0, 228), t1, font=f_hero, fill=(13, 148, 136))
    draw.text((x0 + w1, 228), t2, font=f_hero, fill=(15, 23, 42))

    sub = "Equivariant Molecular Dynamics"
    ws = draw.textlength(sub, font=f_sub)
    draw.text(((w - ws) * 0.5, 316), sub, font=f_sub, fill=(15, 23, 42))

    draw.line([(420, 412), (780, 412)], fill=(13, 148, 136), width=3)

    deck = "50 fs Jumps  ·  FrameSplit CPU API  ·  Lab [physics] Gate  ·  Interactive 3D Studio"
    wd = draw.textlength(deck, font=f_deck)
    draw.text(((w - wd) * 0.5, 448), deck, font=f_deck, fill=(71, 85, 105))

    img.save(path, format="PNG", optimize=True)


def main():
    out_dir = ROOT / "site" / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)

    comp_snaps: dict[str, dict] = {}
    comp_snaps["aspirin_hetero_o3__none__float64"] = _slim_compliance(
        run_compliance_audit(
            target_id="aspirin_hetero_o3", fault="none", precision="float64", euler_deg=(37.0, 64.0, 118.0), rollout_steps=12
        )
    )
    for f in FAULT_CATALOG:
        fid = f["id"]
        comp_snaps[f"aspirin_hetero_o3__{fid}"] = _slim_compliance(
            run_compliance_audit(
                target_id="aspirin_hetero_o3", fault=fid, precision="float32", euler_deg=(37.0, 64.0, 118.0), rollout_steps=12
            )
        )
    for tid in ("argon_flow_v1", "argon_flow_v0", "aspirin_chiral_so3"):
        prec = "float64" if tid == "aspirin_chiral_so3" else "float32"
        comp_snaps[f"{tid}__none"] = _slim_compliance(
            run_compliance_audit(target_id=tid, fault="none", precision=prec, euler_deg=(37.0, 64.0, 118.0), rollout_steps=12)
        )

    rollout_snaps = {
        p: run_studio_rollout(preset=p, n_frames=54, substeps=8, dt_fs=1.5, temperature_k=300.0)
        for p in ("aspirin", "ala2", "ethanol", "li_ec_pf6")
    }

    binding_snaps = {}
    for p in ("aspirin_cox_stable", "aspirin_induced_fit", "aspirin_decoy_clash"):
        rep = pocket.triage_binding(preset=p, n_frames=60, n_replicas=3, seed=0)
        binding_snaps[p] = {"report": rep, "receipt": pocket.format_receipt(rep)}

    electrolyte_snaps = {}
    for p in ("li_ec_fec_additive", "li_ec_base", "li_ec_vc_additive", "li_high_salt_agg"):
        rep = pocket.triage_electrolyte(preset=p, n_frames=80, seed=0)
        electrolyte_snaps[p] = {"report": rep, "receipt": pocket.format_receipt(rep)}

    payload = {
        "argon_seeds": build_argon_seeds_panel(),
        "argon_cinema": build_argon_rollout_and_curves(),
        "cutoff_continuity": build_cutoff_panel(),
        "argon_ablations": build_ablation_summary(),
        "cpu_api": build_cpu_api_panel(),
        "surrogate_lab": build_surrogate_lab_panel(),
        "constraints_demo": build_constraints_and_hybrid_panel(),
        "benchmarks": load_benchmark_suites(),
        "studio_defaults": {
            "compliance_targets": list(AUDIT_TARGETS.values()),
            "fault_catalog": FAULT_CATALOG,
            "default_xyz": DEFAULT_ETHANOL_XYZ,
            "compliance_snapshots": comp_snaps,
            "rollout_aspirin": rollout_snaps["aspirin"],
            "rollout_snapshots": rollout_snaps,
            "binding_snapshots": binding_snaps,
            "electrolyte_snapshots": electrolyte_snaps,
        },
    }

    json_path = out_dir / "homepage-data.json"
    json_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    write_social_poster_png(out_dir / "wignerflow-poster.png")
    print(f"Wrote {json_path} ({json_path.stat().st_size / 1024:.1f} KB) and wignerflow-poster.png")


if __name__ == "__main__":
    main()
