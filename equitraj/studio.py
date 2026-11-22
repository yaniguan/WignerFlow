"""WignerFlow research homepage and live interactive Studio HTTP server."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
from pathlib import Path
import platform
import threading
import time

import torch

from equitraj import pocket
from equitraj.api import InputError
from equitraj.compliance import AUDIT_TARGETS, FAULT_CATALOG, run_compliance_audit
from equitraj.studio_data import BENCHMARK_SUITES, DEFAULT_ETHANOL_XYZ, run_studio_rollout

ROOT_DIR = Path(__file__).resolve().parents[1]
SITE_DIR = ROOT_DIR / "site"

BINDING_PRESETS = ["aspirin_cox_stable", "aspirin_induced_fit", "aspirin_decoy_clash"]
ELECTROLYTE_PRESETS = ["li_ec_fec_additive", "li_ec_base", "li_ec_vc_additive", "li_high_salt_agg"]
ROLLOUT_PRESETS = ["aspirin", "ala2", "ethanol", "li_ec_pf6"]

MAX_BODY_BYTES = 1_048_576

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


def _checked_int(val, name: str, lo: int, hi: int) -> int:
    if isinstance(val, bool):
        raise ValueError(f"{name} must be an integer, got boolean")
    iv = int(val)
    if not lo <= iv <= hi:
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {iv}")
    return iv


def _checked_float(val, name: str, lo: float, hi: float) -> float:
    if isinstance(val, bool):
        raise ValueError(f"{name} must be a number, got boolean")
    fv = float(val)
    if not math.isfinite(fv) or not lo <= fv <= hi:
        raise ValueError(f"{name} must be a finite number in [{lo}, {hi}], got {fv}")
    return fv


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

    def _allowed_origin(self) -> str:
        origin = (self.headers.get("Origin") or "").strip()
        if origin.startswith(("http://127.0.0.1", "http://localhost", "https://127.0.0.1", "https://localhost")):
            return origin
        return "http://127.0.0.1"

    def _set_headers(self, content_type: str, length: int):
        self.send_header("Access-Control-Allow-Origin", self._allowed_origin())
        self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))

    def _send_bytes(self, code: int, content_type: str, raw: bytes, *, head_only: bool = False):
        self.send_response(code)
        self._set_headers(content_type, len(raw))
        self.end_headers()
        if not head_only:
            self.wfile.write(raw)

    def _send_json(self, code: int, payload: dict, *, head_only: bool = False):
        raw = json.dumps(payload, allow_nan=False).encode("utf-8")
        self._send_bytes(code, "application/json; charset=utf-8", raw, head_only=head_only)

    def do_OPTIONS(self):
        self._send_bytes(204, "text/plain; charset=utf-8", b"")

    def _handle_get_or_head(self, *, head_only: bool):
        path = self.path.split("?", 1)[0]
        if path == "/api/overview":
            self._send_json(200, build_overview_payload(), head_only=head_only)
            return
        if path == "/api/catalog":
            self._send_json(200, {
                "binding_presets": BINDING_PRESETS,
                "electrolyte_presets": ELECTROLYTE_PRESETS,
                "datasets": pocket.DATASET_CATALOG,
            }, head_only=head_only)
            return
        if path == "/api/health":
            self._send_json(200, {"status": "ok", "ts": time.time()}, head_only=head_only)
            return

        rel = "index.html" if path in ("", "/") else path.lstrip("/")
        candidate = (SITE_DIR / rel).resolve()
        if SITE_DIR.resolve() in candidate.parents and candidate.is_file():
            ctype = _MIME_OVERRIDES.get(candidate.suffix.lower()) or mimetypes.guess_type(str(candidate))[0] or "application/octet-stream"
            self._send_bytes(200, ctype, candidate.read_bytes(), head_only=head_only)
            return

        self._send_json(404, {"error": f"Unknown GET endpoint {path!r}"}, head_only=head_only)

    def do_GET(self):
        self._handle_get_or_head(head_only=False)

    def do_HEAD(self):
        self._handle_get_or_head(head_only=True)

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 0:
                raise ValueError("Content-Length must be nonnegative")
            if length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Request body exceeds {MAX_BODY_BYTES} bytes"})
                return
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
                    precision = body.get("precision", "float32")
                    if precision not in ("float32", "float64"):
                        raise ValueError("precision must be 'float32' or 'float64'")
                    euler = body.get("euler_deg", [37.0, 64.0, 118.0])
                    if euler is not None:
                        if not isinstance(euler, (list, tuple)) or len(euler) != 3:
                            raise ValueError("euler_deg must be a list of 3 angles in degrees")
                        euler = tuple(_checked_float(a, "euler_deg", -3600.0, 3600.0) for a in euler)
                    rep = run_compliance_audit(
                        target_id=str(body.get("target_id", "argon_flow_v1")),
                        fault=str(body.get("fault", "none")),
                        precision=precision,
                        euler_deg=euler,
                        rollout_steps=_checked_int(body.get("rollout_steps", 14), "rollout_steps", 1, 200),
                        sampling_steps=_checked_int(body.get("sampling_steps", 3), "sampling_steps", 1, 20),
                        seed=_checked_int(body.get("seed", 0), "seed", -1_000_000, 1_000_000),
                    )
                    self._send_json(200, rep)
                elif path == "/api/rollout":
                    rep = run_studio_rollout(
                        preset=str(body.get("preset", "aspirin")),
                        custom_xyz=body.get("custom_xyz"),
                        n_frames=_checked_int(body.get("n_frames", 60), "n_frames", 2, 240),
                        substeps=_checked_int(body.get("substeps", 8), "substeps", 1, 32),
                        dt_fs=_checked_float(body.get("dt_fs", 1.5), "dt_fs", 0.05, 10.0),
                        temperature_k=_checked_float(body.get("temperature_k", 300.0), "temperature_k", 1.0, 5000.0),
                        friction_per_ps=_checked_float(body.get("friction_per_ps", 2.0), "friction_per_ps", 0.0, 100.0),
                        seed=_checked_int(body.get("seed", 0), "seed", -1_000_000, 1_000_000),
                    )
                    self._send_json(200, rep)
                elif path == "/api/cpu_predict":
                    import numpy as np
                    from equitraj.api import AMU_PER_A3_TO_G_CM3, AR_DENSITY_G_CM3

                    pred = _get_argon_predictor()
                    demo = pred.ck["demo"]
                    s_steps = _checked_int(body.get("sampling_steps", 2), "sampling_steps", 1, 10)
                    therm = body.get("thermostat_K")
                    if isinstance(therm, bool):
                        raise ValueError("thermostat_K must be null, 'off', or a number in [1, 1000]")
                    therm_val = None if (therm is None or therm in ("", "off")) else _checked_float(therm, "thermostat_K", 1.0, 1000.0)
                    n_atoms = _checked_int(body.get("n_atoms", 256), "n_atoms", 2, 256)
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
                    elif n_atoms == 256:
                        pos, vel, box_in = demo["pos"], demo["vel"], demo["box"]
                    else:
                        raise ValueError("Studio CPU sandbox supports n_atoms=108 or n_atoms=256")
                    res = pred.predict(
                        pos,
                        vel,
                        box_in,
                        n_steps=1,
                        sampling_steps=s_steps,
                        seed=_checked_int(body.get("seed", 0), "seed", -1_000_000, 1_000_000),
                        thermostat_K=therm_val,
                    )
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
                    for forbidden in ("pdb_path", "xml_path", "ckpt_path"):
                        if body.get(forbidden) is not None:
                            raise ValueError(f"{forbidden} is not accepted over HTTP; use the wignerflow-pocket CLI for local files")
                    mode = body.get("mode", "binding")
                    seed = _checked_int(body.get("seed", 0), "seed", -1_000_000, 1_000_000)
                    if mode == "binding":
                        preset = str(body.get("preset", "aspirin_cox_stable"))
                        if preset not in BINDING_PRESETS:
                            raise ValueError(f"Unknown binding preset {preset!r}")
                        rep = pocket.triage_binding(
                            preset=preset,
                            n_frames=_checked_int(body.get("frames", 70), "frames", 4, 240),
                            n_replicas=_checked_int(body.get("replicas", 3), "replicas", 1, 10),
                            seed=seed,
                        )
                    elif mode == "electrolyte":
                        preset = str(body.get("preset", "li_ec_fec_additive"))
                        if preset not in ELECTROLYTE_PRESETS:
                            raise ValueError(f"Unknown electrolyte preset {preset!r}")
                        rep = pocket.triage_electrolyte(
                            preset=preset,
                            n_frames=_checked_int(body.get("frames", 90), "frames", 4, 240),
                            seed=seed,
                        )
                    elif mode == "checkpoint":
                        rep = pocket.triage_checkpoint(
                            steps=_checked_int(body.get("steps", 24), "steps", 1, 200),
                            sampling_steps=_checked_int(body.get("sampling_steps", 3), "sampling_steps", 1, 20),
                            thermostat_on=bool(body.get("thermostat_on", False)),
                            seed=seed,
                        )
                    else:
                        self._send_json(400, {"error": f"Unknown triage mode {mode!r}"})
                        return
                    self._send_json(200, {"report": rep, "receipt": pocket.format_receipt(rep)})
                else:
                    self._send_json(404, {"error": f"Unknown POST endpoint {path!r}"})
        except (ValueError, TypeError, KeyError, InputError) as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})


def create_studio_server(host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, int(port)), WignerFlowStudioHandler)
