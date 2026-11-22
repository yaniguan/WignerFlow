"""CPU inference API for the released argon model: 3 context frames -> next frame(s), 50 fs apart.

    from equitraj.api import ArgonPredictor
    p = ArgonPredictor()                      # checkpoints/argon_flow_v0.pt, 1 thread, 2 flow steps
    out = p.predict(positions, velocities, box, n_steps=1)

CLI: python -m equitraj.api --bundled --out next.json  (or --input request.json)
HTTP (optional, needs fastapi + uvicorn): uvicorn equitraj.api:app  ->  POST /predict

Argon only. The model was trained on 256 Lennard-Jones argon atoms at 95 K and 1.374 g/cm^3,
so other species are rejected and other states are flagged as out of distribution.
"""
import argparse
import contextlib
import hashlib
import json
import os
import threading
import time
from pathlib import Path

import torch

from equitraj.layers import check_attention_version
from equitraj.model import EquiTraj, build_graph, remove_com
from equitraj.rollout import rollout, temperature

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "argon_flow_v0.pt"
AR_DENSITY_G_CM3 = 1.374
AMU_PER_A3_TO_G_CM3 = 1.66053907
MAX_ATOMS = 256
MAX_STEPS = 20
MIN_BOX = 12.0  # 2 x the 6 A cutoff (minimum image)


class InputError(ValueError):
    """Bad request: the caller should fix the input."""


@contextlib.contextmanager
def _float32():
    """The release weights are float32; ignore a float64 default set elsewhere in the process."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def load_release(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if ck.get("format") != "wignerflow-v0":
        raise ValueError(f"{path} is not a WignerFlow v0 release checkpoint")
    check_attention_version(ck, source=path)
    with _float32():
        model = EquiTraj(n_types=len(ck["elements"]), k=ck["k"], stride_fs=ck["stride_fs"], **ck["model_cfg"])
        model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


class FrameSplitPredictor:
    """Same numbers as EquiTraj.predict, but the spatial blocks run one frame at a time.

    The spatial blocks never mix frames, so this is exact; it is ~2x faster on CPU because each
    frame's edge tensors stay small. The newest frame's graph is reused for the fusion blocks.
    Supports the plain argon configuration only (no charges, bonds, stride conditioning).
    """

    def __init__(self, model):
        if (model.use_charges or model.max_bond_hops or model.n_ff_types or model.stride_conditioning
                or model.geometry_aware or model.head_type != "flow" or not model.use_velocity):
            raise ValueError("FrameSplitPredictor supports the plain argon flow model only")
        self.model = model

    def encode(self, pos, vel, types, box):
        m = self.model
        b, k, n, _ = pos.shape
        if b != 1:
            raise ValueError("one system per call")
        v_norm = vel[0] / m.v_std[types][None, :, None]
        disp = torch.zeros_like(pos[0])
        disp[1:] = pos[0, 1:] - pos[0, :-1]
        d_norm = disp / m.dx_std[types][None, :, None]
        frames = []
        for f in range(k):
            graph = build_graph(pos[:, f], box, m.cutoff, m.lmax, m.neighbor_method)
            x = m.embed(types, v_norm[f], d_norm[f], graph, n)
            for blk in m.spatial:
                x = blk(x, graph)
            frames.append(x)
        x = m.temporal(torch.stack(frames)[None])[0]  # (N, S, c) for the newest frame
        for blk in m.fusion:
            x = blk(x, graph)
        return x, graph

    @torch.no_grad()
    def predict(self, pos, vel, types, box, masses, noise=None, n_steps=2):
        m = self.model
        b, _, n, _ = pos.shape
        cond, graph = self.encode(pos, vel, types, box)
        y = m.com_free_noise(b, types, masses, like=cond) if noise is None else noise
        dt = 1.0 / n_steps
        for i in range(n_steps):  # Euler, as in EquiTraj.predict
            tau = torch.full((b * n,), i * dt, dtype=cond.dtype, device=cond.device)
            y = y + dt * m.denoiser(cond, graph, y, tau)
        dx, v = m.y_to_targets(y, types, vel[:, -1], masses=masses)
        dx, v = remove_com(dx, masses), remove_com(v, masses)
        if m.prior is not None:
            dx, v = m.prior(pos[:, -1], dx, v, types, box, masses)
        return dx, v


class ArgonPredictor:
    """Load once, then call predict() per request."""

    def __init__(self, ckpt=DEFAULT_CKPT, *, threads=1, sampling_steps=2):
        if threads:
            torch.set_num_threads(threads)
        with _float32():
            self.model, self.ck = load_release(ckpt)
        self.fast = FrameSplitPredictor(self.model)
        self.sampling_steps = sampling_steps
        self.sha256 = hashlib.sha256(Path(ckpt).read_bytes()).hexdigest()
        self.name = Path(ckpt).name
        self.mass = float(self.ck["masses"][0])
        demo = self.ck["demo"]
        self.predict(demo["pos"], demo["vel"], demo["box"])  # warm up

    def _check(self, positions, velocities, box, species, n_steps, sampling_steps):
        try:
            pos = torch.as_tensor(positions, dtype=torch.float64)
            vel = torch.as_tensor(velocities, dtype=torch.float32)
            box = torch.as_tensor(box, dtype=torch.float32)
        except (TypeError, ValueError, RuntimeError) as err:
            raise InputError("positions, velocities and box must be numeric arrays") from err
        if pos.ndim != 3 or pos.shape[0] != self.ck["k"] or pos.shape[2] != 3 or vel.shape != pos.shape:
            raise InputError(f"positions and velocities must both have shape ({self.ck['k']}, N, 3)")
        n = pos.shape[1]
        if not 2 <= n <= MAX_ATOMS:
            raise InputError(f"need 2..{MAX_ATOMS} atoms, got {n}")
        names = [species] * n if isinstance(species, str) else list(species)
        if len(names) != n or any(str(s).strip().lower() not in ("ar", "argon", "18") for s in names):
            raise InputError("this model is argon only: every species must be 'Ar'")
        if box.shape == (3,):
            box = torch.diag(box)
        if box.shape != (3, 3) or (box - torch.diag(torch.diagonal(box))).abs().max() > 0:
            raise InputError("box must be [Lx, Ly, Lz] or a diagonal 3x3 matrix in angstrom")
        if not (torch.isfinite(pos).all() and torch.isfinite(vel).all() and torch.isfinite(box).all()):
            raise InputError("inputs must be finite")
        if torch.diagonal(box).min() < MIN_BOX:
            raise InputError(f"every box edge must be >= {MIN_BOX} A (2 x cutoff)")
        if not 1 <= n_steps <= MAX_STEPS:
            raise InputError(f"n_steps must be 1..{MAX_STEPS}")
        if not 1 <= sampling_steps <= 10:
            raise InputError("sampling_steps must be 1..10")
        return pos, vel, box

    def predict(self, positions, velocities, box, *, species="Ar", n_steps=1, sampling_steps=None, seed=0,
                thermostat_K=None):
        """positions (3, N, 3) A and velocities (3, N, 3) A/fs, oldest frame first, 50 fs apart.

        Returns the next n_steps frames (each 50 fs later), their temperatures and warnings.
        Cost is about n_steps x one step. thermostat_K turns on the repo's CSVR thermostat
        (switch S6, tau = 10 strides) at that temperature; off by default (model only).
        """
        with _float32():
            return self._predict(positions, velocities, box, species=species, n_steps=n_steps,
                                 sampling_steps=sampling_steps, seed=seed, thermostat_K=thermostat_K)

    def _predict(self, positions, velocities, box, *, species, n_steps, sampling_steps, seed, thermostat_K):
        t0 = time.perf_counter()
        sampling_steps = sampling_steps or self.sampling_steps
        pos, vel, box = self._check(positions, velocities, box, species, n_steps, sampling_steps)
        stride = float(self.ck["stride_fs"])
        therm = None
        if thermostat_K is not None:
            try:
                thermostat_K = float(thermostat_K)
            except (TypeError, ValueError) as err:
                raise InputError("thermostat_K must be a number") from err
            if not 1.0 <= thermostat_K <= 1000.0:
                raise InputError("thermostat_K must be 1..1000")
            therm = {"t_target": thermostat_K, "tau_fs": 10 * stride, "dt_fs": stride}
        n = pos.shape[1]
        types = torch.zeros(n, dtype=torch.long)
        masses = torch.full((n,), self.mass)
        warnings = []
        density = n * self.mass * AMU_PER_A3_TO_G_CM3 / float(torch.det(box.double()))
        t_in = float(temperature(vel[-1:], masses))
        if abs(density / AR_DENSITY_G_CM3 - 1) > 0.15:
            warnings.append(f"density {density:.3f} g/cm^3 is >15% off the training density {AR_DENSITY_G_CM3}")
        if not 40 <= t_in <= 250:
            warnings.append(f"input temperature {t_in:.1f} K is far from the 95 K training data")
        out = rollout(self.model, pos[None], vel[None], types, box[None], masses, n_steps,
                      sampling_steps=sampling_steps, predictor=self.fast, check_every=1, t_max=3 * 95.0,
                      thermostat=therm, generator=torch.Generator().manual_seed(seed))
        x, v = out["positions"][:, 0], out["velocities"][:, 0]
        crash = int(out["crash_step"][0])
        if crash >= 0:
            warnings.append(f"stability monitor fired at step {crash}; frames from there on are unreliable")
        return {
            "positions": x.tolist(),
            "velocities": v.tolist(),
            "time_fs": [stride * (i + 1) for i in range(len(x))],
            "temperature_K": temperature(v, masses).tolist(),
            "input_temperature_K": t_in,
            "density_g_cm3": density,
            "crash_step": None if crash < 0 else crash,
            "warnings": warnings,
            "model": {"checkpoint": self.name, "sha256": self.sha256, "sampling_steps": sampling_steps,
                      "sampler": "euler", "seed": seed, "thermostat_K": thermostat_K},
            "latency_ms": (time.perf_counter() - t0) * 1e3,
        }


def create_app(predictor=None):
    """Small FastAPI app: POST /predict with the predict() keyword arguments as JSON."""
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="WignerFlow argon CPU API")
    state = {"predictor": predictor}
    lock = threading.Lock()  # one request at a time: parallel torch calls would fight over cores
    allowed = {"positions", "velocities", "box", "species", "n_steps", "sampling_steps", "seed", "thermostat_K"}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.post("/predict")
    def predict(req: dict = Body(...)):
        if state["predictor"] is None:
            state["predictor"] = ArgonPredictor(os.environ.get("WIGNERFLOW_CKPT", DEFAULT_CKPT))
        try:
            if set(req) - allowed or not {"positions", "velocities", "box"} <= set(req):
                raise InputError(f"request needs positions, velocities, box; optional keys {sorted(allowed)}")
            req = dict(req)
            with lock:
                return state["predictor"].predict(req.pop("positions"), req.pop("velocities"),
                                                  req.pop("box"), **req)
        except InputError as err:
            raise HTTPException(status_code=422, detail=str(err)) from err

    return app


def __getattr__(name):  # `uvicorn equitraj.api:app` builds the app lazily
    if name == "app":
        return create_app()
    raise AttributeError(name)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="JSON with positions, velocities, box (and optional species)")
    src.add_argument("--bundled", action="store_true", help="use the 3 starting frames stored in the checkpoint")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--n-steps", type=int, default=1)
    ap.add_argument("--sampling-steps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--thermostat-K", type=float, help="CSVR thermostat target; off by default")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--out", help="write the full JSON here; stdout gets a short summary")
    args = ap.parse_args(argv)
    p = ArgonPredictor(args.ckpt, threads=args.threads, sampling_steps=args.sampling_steps)
    if args.bundled:
        d = p.ck["demo"]
        req = {"positions": d["pos"], "velocities": d["vel"], "box": d["box"]}
    else:
        req = json.loads(Path(args.input).read_text())
    try:
        res = p.predict(req["positions"], req["velocities"], req["box"], species=req.get("species", "Ar"),
                        n_steps=args.n_steps, seed=args.seed, thermostat_K=args.thermostat_K)
    except InputError as err:
        raise SystemExit(f"error: {err}") from err
    if args.out:
        Path(args.out).write_text(json.dumps(res))
    print(json.dumps({k: res[k] for k in ("time_fs", "temperature_K", "crash_step", "warnings", "latency_ms")},
                     indent=2))


if __name__ == "__main__":
    main()
