"""PocketMD CLI & Local Zero-Dependency Web Calculator ("随身科学计算器").

Fast CPU/edge pre-screening tool to decide whether a candidate small-molecule binding pose,
electrolyte additive formulation, or WignerFlow neural checkpoint merits 50-500 GPU-hours
of production Big MD / RB-FEP / AIMD.

Examples:
    python scripts/pocket_md.py presets
    python scripts/pocket_md.py binding --preset aspirin_cox_stable
    python scripts/pocket_md.py binding --pdb systems/aspirin.pdb --xml systems/aspirin.xml
    python scripts/pocket_md.py electrolyte --preset li_ec_fec_additive
    python scripts/pocket_md.py checkpoint --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt
    python scripts/pocket_md.py batch --json-out pocket_batch.json
    python scripts/pocket_md.py serve --port 8765
"""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    saved_gpu_h = sum(80 if r["verdict"] == "REJECT" else (30 if r["verdict"] == "HOLD" else 0) for r in reports)

    print("=" * 88)
    print(f" PocketMD Batch Triage Summary  |  {len(reports)} candidates screened in {total_s:.2f} s on CPU")
    print("-" * 88)
    print(f" {'Domain':<12s} | {'Candidate':<24s} | {'Time(s)':>7s} | {'Verdict':<8s} | Key Diagnostic")
    print("-" * 88)
    for r in reports:
        m = r["metrics"]
        if r["mode"] == "binding":
            diag = f"RMSD={m['ligand_rmsd_plateau_A']:.2f} A, R={m['contact_persistence_R']:.2f}, HB={m['hbond_contacts_mean']:.1f}"
        elif r["mode"] == "electrolyte":
            diag = f"SCN={m['solvent_cn']:.1f}+Add={m['additive_cn']:.1f}, SSIP={m['ssip_fraction']:.2f}, sig={m['sigma_NE_mS_cm']:.1f} mS/cm"
        else:
            diag = f"T={m['T_mean_K']:.1f}K (ref {m['T_ref_K']}K), RDF_L1={m['rdf_l1_err_A']} A, D={m['diffusion_cm2_s']}"
        print(f" {r['mode']:<12s} | {r['name']:<24s} | {r['wall_s']:>7.2f} | {r['verdict']:<8s} | {diag}")
    print("-" * 88)
    print(f" Estimated Big-MD / RB-FEP compute saved by filtering REJECT/HOLD candidates: ~{saved_gpu_h} GPU-hours")
    print("=" * 88)
    _write_json(args.json_out, {"total_wall_s": round(total_s, 3), "saved_gpu_hours_est": saved_gpu_h, "reports": reports})


_WEB_UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PocketMD — WignerFlow 随身科学计算器</title>
<style>
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background: #0f172a; color: #e2e8f0; }
  .wrap { max-width: 980px; margin: 28px auto; padding: 0 18px; }
  .card { background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 20px; margin-bottom: 16px; }
  .row { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
  button { background: #2563eb; color: #fff; border: 0; border-radius: 8px; padding: 9px 14px; cursor: pointer; font-weight: 600; }
  button.sec { background: #334155; }
  button:hover { opacity: 0.92; }
  pre { background: #090d16; border: 1px solid #334155; border-radius: 10px; padding: 14px; overflow-x: auto; font-size: 13px; line-height: 1.45; }
  .badge { display: inline-block; padding: 4px 10px; border-radius: 999px; font-weight: 700; font-size: 13px; }
  .GO { background: #065f46; color: #6ee7b7; }
  .HOLD { background: #78350f; color: #fcd34d; }
  .REJECT { background: #7f1d1d; color: #fca5a5; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h2 style="margin:0 0 6px 0;">PocketMD · WignerFlow 随身科学计算器 (Live CPU Server)</h2>
    <div style="color:#94a3b8;font-size:14px;">秒级 CPU 快速轨迹推演：决定小分子结合构象 / 电解液添加剂配方是否值得投入 50–500 GPU 小时大 MD。</div>
    <div class="row">
      <button onclick="run('binding','aspirin_cox_stable')">药物口袋：Aspirin 稳定构象</button>
      <button class="sec" onclick="run('binding','aspirin_induced_fit')">药物口袋：诱导契合构象</button>
      <button class="sec" onclick="run('binding','aspirin_decoy_clash')">药物口袋：碰撞假阳性构象</button>
    </div>
    <div class="row">
      <button onclick="run('electrolyte','li_ec_fec_additive')">电解液：EC/DMC + 10% FEC</button>
      <button class="sec" onclick="run('electrolyte','li_ec_base')">电解液：1M LiPF6/EC 基准</button>
      <button class="sec" onclick="run('electrolyte','li_ec_vc_additive')">电解液：高粘度 VC 添加剂</button>
      <button class="sec" onclick="run('electrolyte','li_high_salt_agg')">电解液：强配对 AGG 陷阱</button>
      <button onclick="run('checkpoint','argon_flow_v1_cpu_reduced')">WignerFlow 神经网络推演 (CPU)</button>
    </div>
  </div>
  <div class="card">
    <div id="status">点击上方任一候选体系，立即在本地 CPU 实时推演 MD 轨迹...</div>
    <pre id="receipt">Waiting for calculation...</pre>
  </div>
</div>
<script>
async function run(mode, preset) {
  document.getElementById('status').innerHTML = '正在本地 CPU 上快速推演 <b>' + preset + '</b> ...';
  const r = await fetch('/api/triage', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({mode, preset})
  });
  const data = await r.json();
  document.getElementById('status').innerHTML =
    '<span class="badge ' + data.report.verdict + '">' + data.report.verdict + '</span> ' +
    '<b>' + (data.report.title || data.report.name) + '</b> · ' + data.report.wall_s + 's on CPU';
  document.getElementById('receipt').textContent = data.receipt;
}
</script>
</body>
</html>
"""


class _PocketHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # keep terminal clean

    def _send_json(self, code, payload):
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            raw = _WEB_UI_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif self.path == "/api/catalog":
            self._send_json(200, {
                "binding_presets": BINDING_PRESETS,
                "electrolyte_presets": ELECTROLYTE_PRESETS,
                "datasets": pocket.DATASET_CATALOG,
            })
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/triage":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
        mode = body.get("mode", "binding")
        preset = body.get("preset")
        try:
            if mode == "binding":
                rep = pocket.triage_binding(preset=preset or "aspirin_cox_stable")
            elif mode == "electrolyte":
                rep = pocket.triage_electrolyte(preset=preset or "li_ec_fec_additive")
            elif mode == "checkpoint":
                rep = pocket.triage_checkpoint()
            else:
                self._send_json(400, {"error": f"unknown mode {mode!r}"})
                return
            self._send_json(200, {"report": rep, "receipt": pocket.format_receipt(rep)})
        except Exception as exc:  # pragma: no cover - defensive HTTP handler
            self._send_json(500, {"error": str(exc)})


def cmd_serve(args):
    srv = ThreadingHTTPServer((args.host, args.port), _PocketHandler)
    print(f"PocketMD Calculator running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
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
