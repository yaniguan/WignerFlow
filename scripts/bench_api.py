"""CPU latency (p50/p95 per request) and accuracy of equitraj.api.ArgonPredictor.

    python scripts/bench_api.py --out results/cpu_api/bench.json

Latency (wall and process CPU time): single-step requests on the bundled 256-atom context (and a 108-atom FCC argon box at
the training density, for scaling only), for several flow sampling steps.
Accuracy, against the released model's own settings (EquiTraj.predict, 10 Euler steps):
  * one step: the frame-split path must match EquiTraj.predict exactly for the same noise;
    fewer sampling steps are compared to 10 steps with the same noise (RMS error of dx and v,
    relative to the RMS of dx and v themselves);
  * rollouts: --rollout-steps steps from the bundled context, second-half temperature and the
    RDF first peak against the ground-truth reference stored in the checkpoint.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from equitraj import analysis  # noqa: E402
from equitraj.api import AMU_PER_A3_TO_G_CM3, AR_DENSITY_G_CM3, DEFAULT_CKPT, ArgonPredictor  # noqa: E402
from equitraj.rollout import rollout, temperature  # noqa: E402


def cpu_name():
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _load_avg():
    return os.getloadavg() if hasattr(os, "getloadavg") else None


def fcc_argon(m, mass, temp_k=95.0, stride_fs=50.0, seed=0):
    """4 m^3 argon atoms on an FCC lattice at the training density, 3 coasting frames."""
    n = 4 * m ** 3
    side = (n * mass * AMU_PER_A3_TO_G_CM3 / AR_DENSITY_G_CM3) ** (1 / 3)
    base = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    cells = np.array([[i, j, k] for i in range(m) for j in range(m) for k in range(m)])
    x = ((cells[:, None, :] + base[None]).reshape(-1, 3) * side / m)
    rng = np.random.default_rng(seed)
    kb_amu = 0.83144626e-4  # kB in amu A^2 fs^-2 K^-1
    v = rng.normal(0, np.sqrt(kb_amu * temp_k / mass), (n, 3))
    v -= v.mean(0)
    pos = np.stack([x - (2 - f) * stride_fs * v for f in range(3)])
    return pos, np.stack([v] * 3), [side] * 3


def timed(fn, reps):
    """Wall time and process CPU time per call (ms). With 1 thread, CPU time is what the request
    costs on an otherwise idle core; wall time also includes waiting for a busy shared CPU."""
    wall, cpu = [], []
    for _ in range(reps):
        t, c = time.perf_counter(), time.process_time()
        fn()
        wall.append((time.perf_counter() - t) * 1e3)
        cpu.append((time.process_time() - c) * 1e3)
    q = lambda xs: {"p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95)), "max": max(xs)}  # noqa: E731
    return {"n": reps, "wall": q(wall), "cpu": q(cpu)}


def rollout_stats(p, steps, sampling_steps, seed, fast=True, thermostat_K=None):
    ck = p.ck
    d, ref = ck["demo"], ck["reference"]
    stride = float(ck["stride_fs"])
    therm = None if thermostat_K is None else {"t_target": thermostat_K, "tau_fs": 10 * stride, "dt_fs": stride}
    out = rollout(p.model, d["pos"][None], d["vel"][None], ck["types"], d["box"][None], ck["masses"], steps,
                  sampling_steps=sampling_steps, predictor=p.fast if fast else None, t_max=3 * ref["T_K"],
                  thermostat=therm, generator=torch.Generator().manual_seed(seed))
    x, v = out["positions"][:, 0].numpy(), out["velocities"][:, 0]
    half = steps // 2
    idx = np.arange(x.shape[1])
    r_ref, g_ref = np.array(ref["rdf_r"]), np.array(ref["rdf_g"])
    box = np.tile(d["box"].numpy(), (steps - half, 1, 1))
    r, g, _ = analysis.rdf(x[half:], box, idx, idx, float(r_ref[-1]) + 1e-6, len(r_ref))
    pm, hm = analysis.first_peak(r, g)
    pg, hg = analysis.first_peak(r_ref, g_ref)
    return {"crash_step": int(out["crash_step"][0]), "wall_s": out["wall_s"],
            "T_second_half_K": float(temperature(v[half:], ck["masses"]).mean()), "T_ref_K": float(ref["T_K"]),
            "rdf_peak_pos_err_A": float(abs(pm - pg)), "rdf_peak_height_rel_err": float(abs(hm - hg) / hg)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--sampling-steps", type=int, nargs="+", default=[1, 2, 4, 10])
    ap.add_argument("--rollout-steps", type=int, default=100)
    ap.add_argument("--rollout-seeds", type=int, default=3)
    ap.add_argument("--skip-rollouts", action="store_true")
    ap.add_argument("--thermostat-K", type=float, help="run the rollouts with the CSVR thermostat at this T")
    ap.add_argument("--out")
    args = ap.parse_args()
    p = ArgonPredictor(args.ckpt, threads=args.threads)
    ck, d = p.ck, p.ck["demo"]
    rep = {"cpu": cpu_name(), "logical_cpus": os.cpu_count(), "threads": args.threads, "torch": torch.__version__,
           "load_avg_start": _load_avg(), "checkpoint": p.name, "latency_ms": {}, "one_step": {}, "rollouts": {}}

    lat_pos, lat_vel, lat_box = fcc_argon(3, p.mass)
    for s in args.sampling_steps:
        rep["latency_ms"][f"N=256,S={s}"] = timed(lambda: p.predict(d["pos"], d["vel"], d["box"], sampling_steps=s),
                                                  args.reps)
        rep["latency_ms"][f"N=108,S={s}"] = timed(lambda: p.predict(lat_pos, lat_vel, lat_box, sampling_steps=s),
                                                  max(5, args.reps // 3))
        print(f"S={s}", rep["latency_ms"][f"N=256,S={s}"], flush=True)
    model_only = lambda: p.model.predict(d["pos"][None], d["vel"][None], ck["types"], d["box"][None],  # noqa: E731
                                         ck["masses"], n_steps=10)
    rep["latency_ms"]["N=256,S=10,EquiTraj.predict (reference path)"] = timed(model_only, max(5, args.reps // 3))

    pos, vel, box = d["pos"][None], d["vel"][None], d["box"][None]
    errs = {s: {"dx": [], "v": []} for s in args.sampling_steps}
    exact = []
    for seed in range(5):
        noise = p.model.com_free_noise(1, ck["types"], ck["masses"], like=vel[0],
                                       generator=torch.Generator().manual_seed(seed))
        with torch.inference_mode():
            dx10, v10 = p.model.predict(pos, vel, ck["types"], box, ck["masses"], n_steps=10, noise=noise)
            for s in args.sampling_steps:
                dx, v = p.fast.predict(pos, vel, ck["types"], box, ck["masses"], n_steps=s, noise=noise)
                if s == 10:
                    exact.append(max((dx - dx10).abs().max().item(), (v - v10).abs().max().item()))
                errs[s]["dx"].append(((dx - dx10).pow(2).mean() / dx10.pow(2).mean()).sqrt().item())
                errs[s]["v"].append(((v - v10).pow(2).mean() / v10.pow(2).mean()).sqrt().item())
    rep["one_step"] = {"frame_split_vs_EquiTraj_predict_max_abs_diff": max(exact) if exact else None,
                       **{f"S={s}_vs_S=10_rel_rms": {k: float(np.mean(e)) for k, e in errs[s].items()}
                          for s in args.sampling_steps}}
    print(json.dumps(rep["one_step"]), flush=True)

    if not args.skip_rollouts:
        for s in args.sampling_steps:
            runs = [rollout_stats(p, args.rollout_steps, s, seed, thermostat_K=args.thermostat_K)
                    for seed in range(args.rollout_seeds)]
            key = f"S={s}" if args.thermostat_K is None else f"S={s},CSVR {args.thermostat_K:g} K"
            rep["rollouts"][key] = {"runs": runs, **{k: float(np.mean([r[k] for r in runs]))
                                         for k in ("T_second_half_K", "rdf_peak_pos_err_A", "rdf_peak_height_rel_err")}}
            print(f"rollout {key}", {k: v for k, v in rep["rollouts"][key].items() if k != "runs"}, flush=True)
    rep["load_avg_end"] = _load_avg()
    text = json.dumps(rep, indent=2)
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")


if __name__ == "__main__":
    main()
