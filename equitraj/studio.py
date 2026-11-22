"""WignerFlow research homepage and live interactive Studio HTTP server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import platform
import threading
import time

import torch

from equitraj import pocket
from equitraj.compliance import AUDIT_TARGETS, FAULT_CATALOG, run_compliance_audit
from equitraj.studio_data import BENCHMARK_SUITES, DEFAULT_ETHANOL_XYZ, run_studio_rollout

ROOT_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT_DIR / "site"

BINDING_PRESETS = ["aspirin_cox_stable", "aspirin_induced_fit", "aspirin_decoy_clash"]
ELECTROLYTE_PRESETS = ["li_ec_fec_additive", "li_ec_base", "li_ec_vc_additive", "li_high_salt_agg"]
ROLLOUT_PRESETS = ["aspirin", "ala2", "ethanol", "li_ec_pf6"]

_MIME_OVERRIDES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".webp": "image/webp",
}

_CACHED_PREDICTOR = None
_INFER_LOCK = threading.Lock()


def _get_argon_predictor():
    global _CACHED_PREDICTOR
    if _CACHED_PREDICTOR is None:
        from equitraj.api import ArgonPredictor

        _CACHED_PREDICTOR = ArgonPredictor(threads=None, sampling_steps=2)
    return _CACHED_PREDICTOR


def build_overview_payload() -> dict:
    return {
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cpu_threads": int(torch.get_num_threads()),
            "cuda": bool(torch.cuda.is_available()),
        },
        "compliance_targets": list(AUDIT_TARGETS.values()),
        "fault_catalog": FAULT_CATALOG,
        "binding_presets": BINDING_PRESETS,
        "electrolyte_presets": ELECTROLYTE_PRESETS,
        "rollout_presets": ROLLOUT_PRESETS,
        "default_xyz": DEFAULT_ETHANOL_XYZ,
        "datasets": pocket.DATASET_CATALOG,
        "benchmarks": BENCHMARK_SUITES,
    }


class WignerFlowStudioHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return

    def _set_headers(self, content_type: str, length: int):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))

    def _send_bytes(self, code: int, content_type: str, raw: bytes):
        self.send_response(code)
        self._set_headers(content_type, len(raw))
        self.end_headers()
        self.wfile.write(raw)

    def _send_json(self, code: int, payload: dict):
        self._send_bytes(code, "application/json; charset=utf-8", json.dumps(payload).encode("utf-8"))

    def do_OPTIONS(self):
        self._send_bytes(204, "text/plain; charset=utf-8", b"")

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/overview":
            self._send_json(200, build_overview_payload())
            return
        if path == "/api/catalog":
            self._send_json(200, {
                "binding_presets": BINDING_PRESETS,
                "electrolyte_presets": ELECTROLYTE_PRESETS,
                "datasets": pocket.DATASET_CATALOG,
            })
            return
        if path == "/api/health":
            self._send_json(200, {"status": "ok", "ts": time.time()})
            return

        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        candidate = (SITE_DIR / rel).resolve()
        if SITE_DIR.resolve() in candidate.parents and candidate.is_file():
            ctype = _MIME_OVERRIDES.get(candidate.suffix.lower()) or mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            self._send_bytes(200, ctype, candidate.read_bytes())
            return

        self._send_json(404, {"error": f"Unknown GET endpoint {path!r}"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = max(0, int(self.headers.get("Content-Length", "0")))
            raw_body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            body = json.loads(raw_body or "{}")
            if not isinstance(body, dict):
                raise ValueError("JSON request body must be an object")
        except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": f"Invalid JSON request body: {exc}"})
            return

        try:
            with _INFER_LOCK:
                if path == "/api/compliance":
                    euler = body.get("euler_deg", [37.0, 64.0, 118.0])
                    rep = run_compliance_audit(
                        target_id=body.get("target_id", "argon_flow_v1"),
                        fault=body.get("fault", "none"),
                        precision=body.get("precision", "float32"),
                        euler_deg=tuple(euler) if euler else None,
                        rollout_steps=int(body.get("rollout_steps", 14)),
                        sampling_steps=int(body.get("sampling_steps", 3)),
                        seed=int(body.get("seed", 0)),
                    )
                    self._send_json(200, rep)
                elif path == "/api/rollout":
                    rep = run_studio_rollout(
                        preset=body.get("preset", "aspirin"),
                        custom_xyz=body.get("custom_xyz"),
                        n_frames=int(body.get("n_frames", 60)),
                        substeps=int(body.get("substeps", 8)),
                        dt_fs=float(body.get("dt_fs", 1.5)),
                        temperature_k=float(body.get("temperature_k", 300.0)),
                        friction_per_ps=float(body.get("friction_per_ps", 2.0)),
                        seed=int(body.get("seed", 0)),
                    )
                    self._send_json(200, rep)
                elif path == "/api/cpu_predict":
                    import numpy as np
                    from equitraj.api import AMU_PER_A3_TO_G_CM3, AR_DENSITY_G_CM3

                    pred = _get_argon_predictor()
                    demo = pred.ck["demo"]
                    s_steps = max(1, min(10, int(body.get("sampling_steps", 2))))
                    therm = body.get("thermostat_K")
                    therm_val = None if therm in (None, "", "off", False) else float(therm)
                    n_atoms = int(body.get("n_atoms", 256))
                    if n_atoms == 108:
                        side = float((108 * pred.mass * AMU_PER_A3_TO_G_CM3 / AR_DENSITY_G_CM3) ** (1 / 3))
                        base = np.array([[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]])
                        cells = np.array([[i, j, k] for i in range(3) for j in range(3) for k in range(3)])
                        x = (cells[:, None, :] + base[None]).reshape(-1, 3) * (side / 3.0)
                        rng = np.random.default_rng(0)
                        v0 = rng.normal(0, np.sqrt(0.83144626e-4 * 95.0 / pred.mass), (108, 3))
                        v0 -= v0.mean(0)
                        pos = np.stack([x - (2 - f) * 50.0 * v0 for f in range(3)])
                        vel = np.stack([v0] * 3)
                        box_in = [side, side, side]
                    else:
                        n_atoms = 256
                        pos, vel, box_in = demo["pos"], demo["vel"], demo["box"]
                    res = pred.predict(
                        pos,
                        vel,
                        box_in,
                        n_steps=1,
                        sampling_steps=s_steps,
                        seed=int(body.get("seed", 0)),
                        thermostat_K=therm_val,
                    )
                    # Trim large coordinate arrays to first 2 atoms for compact UI JSON inspection
                    preview = {
                        "n_atoms": n_atoms,
                        "time_fs": res["time_fs"],
                        "temperature_K": [round(float(x), 2) for x in res["temperature_K"]],
                        "input_temperature_K": round(float(res["input_temperature_K"]), 2),
                        "density_g_cm3": round(float(res["density_g_cm3"]), 4),
                        "crash_step": res["crash_step"],
                        "warnings": res["warnings"],
                        "latency_ms": round(float(res["latency_ms"]), 1),
                        "model": res["model"],
                        "positions_sample_A": [[round(float(c), 3) for c in atom] for atom in res["positions"][0][:2]],
                    }
                    self._send_json(200, preview)
                elif path == "/api/surrogate":
                    from lab import surrogate_query as sq

                    query = body.get("query") or {"system": "argon", "n_steps": 2, "sampling_steps": 2, "seed": 0}
                    rep = sq.run_query(query)
                    self._send_json(200, {"query": query, "report": rep, "context_lines": sq.to_reasoner_context(rep)})
                elif path == "/api/triage":
                    mode = body.get("mode", "binding")
                    if mode == "binding":
                        rep = pocket.triage_binding(
                            preset=body.get("preset", "aspirin_cox_stable"),
                            pdb_path=body.get("pdb_path"),
                            xml_path=body.get("xml_path"),
                            n_frames=int(body.get("frames", 70)),
                            n_replicas=int(body.get("replicas", 3)),
                            seed=int(body.get("seed", 0)),
                        )
                    elif mode == "electrolyte":
                        rep = pocket.triage_electrolyte(
                            preset=body.get("preset", "li_ec_fec_additive"),
                            n_frames=int(body.get("frames", 90)),
                            seed=int(body.get("seed", 0)),
                        )
                    elif mode == "checkpoint":
                        rep = pocket.triage_checkpoint(
                            ckpt_path=body.get("ckpt_path"),
                            steps=int(body.get("steps", 24)),
                            sampling_steps=int(body.get("sampling_steps", 3)),
                            thermostat_on=bool(body.get("thermostat_on", False)),
                            seed=int(body.get("seed", 0)),
                        )
                    else:
                        self._send_json(400, {"error": f"Unknown triage mode {mode!r}"})
                        return
                    self._send_json(200, {"report": rep, "receipt": pocket.format_receipt(rep)})
                else:
                    self._send_json(404, {"error": f"Unknown POST endpoint {path!r}"})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def create_studio_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), WignerFlowStudioHandler)
