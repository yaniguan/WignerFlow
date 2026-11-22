"""PocketMD CLI and local zero-dependency web calculator.

Fast CPU/edge pre-screening tool to decide whether a candidate small-molecule binding pose,
electrolyte additive formulation, or WignerFlow neural checkpoint is worth
expensive production Big MD / RB-FEP / AIMD.

Examples:
    python scripts/pocket_md.py presets
    python scripts/pocket_md.py binding --preset aspirin_cox_stable
    python scripts/pocket_md.py binding --pdb systems/aspirin.pdb --xml systems/aspirin.xml
    python scripts/pocket_md.py electrolyte --preset li_ec_fec_additive
    python scripts/pocket_md.py checkpoint --ckpt checkpoints/argon_flow_v0.pt
    python scripts/pocket_md.py batch --json-out pocket_batch.json
    python scripts/pocket_md.py serve --port 8765
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from equitraj import pocket  # noqa: E402

BINDING_PRESETS = ["aspirin_cox_stable", "aspirin_induced_fit", "aspirin_decoy_clash"]
ELECTROLYTE_PRESETS = ["li_ec_fec_additive", "li_ec_base", "li_ec_vc_additive", "li_high_salt_agg"]


def _write_json(path, payload):
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def cmd_presets(_args):
    print("=" * 72)
    print(" PocketMD Built-In Screening Presets")
    print("=" * 72)
    print(" [Drug Binding Poses]   :", ", ".join(BINDING_PRESETS))
    print(" [Battery Electrolytes] :", ", ".join(ELECTROLYTE_PRESETS))
    ckpts = sorted(p.name for p in (REPO_ROOT / "checkpoints").glob("*.pt"))
    print(" [Neural Checkpoints]   :", ", ".join(ckpts) if ckpts else "(none)")
    print("-" * 72)
    print(" Public MD Datasets Ready for `scripts/import_md.py` + WignerFlow:")
    for ds in pocket.DATASET_CATALOG:
        print(f"  - [{ds['domain']:<11s}] {ds['title']}")
        print(f"      URL   : {ds['url']}")
        print(f"      Import: {ds['import_hint']}")
    print("=" * 72)


def cmd_binding(args):
    rep = pocket.triage_binding(
        preset=args.preset,
        pdb_path=args.pdb,
        xml_path=args.xml,
        n_frames=args.frames,
        n_replicas=args.replicas,
        seed=args.seed,
    )
    print(pocket.format_receipt(rep))
    _write_json(args.json_out, rep)


def cmd_electrolyte(args):
    rep = pocket.triage_electrolyte(
        preset=args.preset,
        n_frames=args.frames,
        seed=args.seed,
    )
    print(pocket.format_receipt(rep))
    _write_json(args.json_out, rep)


def cmd_checkpoint(args):
    rep = pocket.triage_checkpoint(
        ckpt_path=args.ckpt,
        steps=args.steps,
        sampling_steps=args.sampling_steps,
        thermostat_on=args.thermostat,
        seed=args.seed,
    )
    print(pocket.format_receipt(rep))
    _write_json(args.json_out, rep)


def run_batch(include_checkpoint=True):
    reports = []
    for p in BINDING_PRESETS:
        reports.append(pocket.triage_binding(p))
    for p in ELECTROLYTE_PRESETS:
        reports.append(pocket.triage_electrolyte(p))
    if include_checkpoint:
        reports.append(pocket.triage_checkpoint())
    return reports


def cmd_batch(args):
    reports = run_batch(include_checkpoint=not args.skip_checkpoint)
    total_s = sum(r["wall_s"] for r in reports)

    print("=" * 100)
    print(f" PocketMD v2 Batch Triage Summary  |  {len(reports)} candidates screened in {total_s:.2f} s on CPU")
    print("-" * 100)
    print(f" {'Domain':<12s} | {'Candidate':<25s} | {'Time(s)':>7s} | {'Verdict':<7s} | Literature-Aligned Diagnostics")
    print("-" * 100)
    for r in reports:
        m = r["metrics"]
        if r["mode"] == "binding":
            diag = (f"CompScore={m['comp_score']:+.2f}, Pose={m['pose_score_A']:.2f}A, "
                    f"vdW={m['posebusters_min_vdw_ratio']:.2f}, tau50={m['tau_ramd_blocks_50fs']:.0f}blk")
        elif r["mode"] == "electrolyte":
            cags_txt = "base" if m["cags_additive_score"] is None else f"{m['cags_additive_score']:+.2f}"
            diag = (f"sig_cNE={m['sigma_cNE_mS_cm']:.1f} (NE {m['sigma_NE_mS_cm']:.1f}) mS/cm, "
                    f"Ion={m['ionicity_haven_inv']:.2f}, CAGS={cags_txt}")
        else:
            diag = (f"T={m['T_mean_K']:.1f}K, RDF_L1={m['rdf_l1_err_A']}A, "
                    f"dODE={m['ode_curvature_delta_A']}A, sFlow={m['flow_epistemic_spread_A']}A")
        print(f" {r['mode']:<12s} | {r['name']:<25s} | {r['wall_s']:>7.2f} | {r['verdict']:<7s} | {diag}")
    print("-" * 100)
    print("=" * 100)
    _write_json(args.json_out, {"total_wall_s": round(total_s, 3), "reports": reports})


def cmd_serve(args):
    from equitraj.studio import create_studio_server
    srv = create_studio_server(args.host, args.port)
    print(f"WignerFlow Studio running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        srv.server_close()


def build_parser():
    ap = argparse.ArgumentParser(description="PocketMD: fast CPU/edge MD triage calculator for WignerFlow.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_pre = sub.add_parser("presets", help="List built-in screening presets and public MD dataset import recipes")
    p_pre.set_defaults(func=cmd_presets)

    p_bind = sub.add_parser("binding", help="Screen a small-molecule + protein pocket binding pose on CPU")
    p_bind.add_argument("--preset", default="aspirin_cox_stable", choices=BINDING_PRESETS)
    p_bind.add_argument("--pdb", help="Custom complex or ligand PDB file")
    p_bind.add_argument("--xml", help="Optional OpenMM force-field XML file")
    p_bind.add_argument("--frames", type=int, default=70)
    p_bind.add_argument("--replicas", type=int, default=3, help="Number of short OpenBPMD thermal replicas")
    p_bind.add_argument("--seed", type=int, default=0)
    p_bind.add_argument("--json-out", help="Write full triage JSON report (including curves & viewer frames)")
    p_bind.set_defaults(func=cmd_binding)

    p_elec = sub.add_parser("electrolyte", help="Screen a battery electrolyte + new additive formulation on CPU")
    p_elec.add_argument("--preset", default="li_ec_fec_additive", choices=ELECTROLYTE_PRESETS)
    p_elec.add_argument("--frames", type=int, default=90)
    p_elec.add_argument("--seed", type=int, default=0)
    p_elec.add_argument("--json-out", help="Write full triage JSON report")
    p_elec.set_defaults(func=cmd_electrolyte)

    p_ck = sub.add_parser("checkpoint", help="Run fast CPU Verlet rollout triage on a WignerFlow .pt checkpoint")
    p_ck.add_argument("--ckpt", help="Path to WignerFlow .pt checkpoint (defaults to CPU reduced Argon checkpoint)")
    p_ck.add_argument("--steps", type=int, default=24)
    p_ck.add_argument("--sampling-steps", type=int, default=3)
    p_ck.add_argument("--thermostat", action="store_true", help="Enable CSVR velocity-rescaling thermostat")
    p_ck.add_argument("--seed", type=int, default=0)
    p_ck.add_argument("--json-out", help="Write full triage JSON report")
    p_ck.set_defaults(func=cmd_checkpoint)

    p_bat = sub.add_parser("batch", help="Screen all candidate poses & electrolyte formulations in one shot")
    p_bat.add_argument("--skip-checkpoint", action="store_true")
    p_bat.add_argument("--json-out", help="Write combined batch JSON report")
    p_bat.set_defaults(func=cmd_batch)

    p_srv = sub.add_parser("serve", help="Launch local zero-dependency browser calculator UI")
    p_srv.add_argument("--host", default="127.0.0.1")
    p_srv.add_argument("--port", type=int, default=8765)
    p_srv.set_defaults(func=cmd_serve)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
