"""Paired start/seed evaluation with a cluster bootstrap over starting windows.

Physical metrics are calculated per surviving rollout using evaluate.py conventions.
CIs resample starts (all their seeds together), conditional on the available test data.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from equitraj import analysis
from equitraj.evaluation import first_zero, load_model
from equitraj.rollout import rollout, temperature

METRICS = ("temperature_rel_err", "rdf_l1", "vacf_zero_rel_err", "D_rel_err",
           "crash_fraction", "steps_survived")


def bootstrap_summary(rows, n_bootstrap=2000, seed=1234):
    """Percentile 95% CI of the mean; preserve within-start seed dependence.

    Missing/nonfinite metrics are excluded, never imputed. Replicates with no
    valid observations are excluded and counted. One start cannot identify
    between-start uncertainty and yields only within-sample point intervals.
    """
    starts = sorted({row["start"] for row in rows})
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(starts), size=(n_bootstrap, len(starts)))
    result = {}
    for metric in METRICS:
        groups = [np.array([r[metric] for r in rows if r["start"] == s
                            and r.get(metric) is not None and np.isfinite(r[metric])], dtype=float)
                  for s in starts]
        values = np.concatenate(groups)
        sums = np.array([g.sum() for g in groups])
        counts = np.array([len(g) for g in groups])
        denom = counts[draws].sum(axis=1)
        boot = sums[draws].sum(axis=1)[denom > 0] / denom[denom > 0]
        result[metric] = {"mean": float(values.mean()) if len(values) else None,
                          "ci95": np.quantile(boot, [.025, .975]).tolist() if len(boot) else [None, None],
                          "n_valid": len(values), "n_missing": len(rows) - len(values),
                          "n_bootstrap_valid": len(boot)}
    return result


def reference(ds, steps):
    x = [np.asarray(p[::ds.f]) for p in ds.pos]
    v = [np.asarray(p[::ds.f]) for p in ds.vel]
    dt = ds.stride_fs
    if min(map(len, x)) <= ds.k + 1:
        raise ValueError("Reference trajectories are too short")
    lag = min(max(2, steps // 4), int(2000 / dt), min(map(len, v)))
    lags = np.unique(np.linspace(0, min(steps // 2, int(20000 / dt), min(map(len, x))-1), 100).astype(int))
    t_lag = lags * dt
    fit = (.25 * t_lag[-1], t_lag[-1])
    idx = np.arange(len(ds.types))
    rmax = min(10., .49 * min(float(b.diag().min()) for b in ds.boxes)) if ds.periodic else 8.
    gs = []
    for xx, box in zip(x, ds.boxes):
        frames = xx[::max(1, len(xx)//400)]
        r, g, _ = analysis.rdf(frames, np.tile(box.numpy(), (len(frames), 1, 1)) if ds.periodic else None,
                               idx, idx, rmax, 200)
        gs.append(g)
    c = np.mean([analysis.vacf(vv[:len(vv)//2*2], lag) for vv in v], axis=0)
    zero = first_zero(np.arange(lag)*dt, c / c[0])
    d = analysis.diffusion_coefficient(t_lag, np.mean([analysis.msd(xx, lags) for xx in x], axis=0), *fit)[1] if ds.periodic else None
    return dict(x=x, v=v, T=float(np.mean([temperature(torch.from_numpy(vv[::10].copy()), ds.masses).mean().item() for vv in v])),
                lag=lag, lags=lags, fit=fit, zero=zero, D=d, r=r, g=np.mean(gs, axis=0), rmax=rmax)


def physical_metrics(x, v, ds, box, ref):
    """Same second-half T/RDF/VACF and full-rollout MSD convention as evaluate.py."""
    half = len(x)//2
    temp = temperature(torch.from_numpy(v[half:].copy()), ds.masses).mean().item()
    idx = np.arange(len(ds.types))
    frames = x[half::max(1, len(x)//400)]
    r, g, _ = analysis.rdf(frames, np.tile(box, (len(frames), 1, 1)) if ds.periodic else None,
                           idx, idx, ref["rmax"], 200)
    c = analysis.vacf(v[half:], ref["lag"])
    z = first_zero(np.arange(ref["lag"])*ds.stride_fs, c / c[0]) if c[0] > 0 else None
    d = analysis.diffusion_coefficient(ref["lags"]*ds.stride_fs, analysis.msd(x, ref["lags"]), *ref["fit"])[1] if ds.periodic else None
    return {"temperature_rel_err": abs(temp-ref["T"])/ref["T"],
            "rdf_l1": float(analysis.rdf_l1(r, g, ref["g"])),
            "vacf_zero_rel_err": None if z is None or ref["zero"] is None or ref["zero"] == 0 else float(abs(z-ref["zero"])/abs(ref["zero"])),
            "D_rel_err": None if d is None or ref["D"] == 0 else float(abs(d-ref["D"])/abs(ref["D"]))}


def evaluate_checkpoint(path, args):
    model, cfg, ds, step = load_model(str(path.parent), path.name, torch.device(args.device), args.split, args.data_dir)
    ref = reference(ds, args.steps)
    rows = []
    for i in range(args.starts):
        traj = i % len(ref["x"])
        available = len(ref["x"][traj]) - cfg["k"] - 1
        fraction = (i // len(ref["x"])) / max(1, int(np.ceil(args.starts/len(ref["x"]))))
        s = int(fraction * available)
        for seed in range(args.seeds):
            actual_seed = args.seed0 + seed
            torch.manual_seed(actual_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(actual_seed)
            pos = torch.from_numpy(ref["x"][traj][s:s+cfg["k"]].copy())[None]
            vel = torch.from_numpy(ref["v"][traj][s:s+cfg["k"]].copy())[None]
            box = ds.boxes[traj][None] if ds.periodic else None
            out = rollout(model, pos, vel, ds.types, box, ds.masses, args.steps,
                          sampling_steps=args.sampling_steps, save_every=1, t_max=3*ref["T"])
            crash = int(out["crash_step"][0])
            row = {"start": i, "trajectory_id": int(ds.traj_ids[traj]), "model_frame": s, "saved_frame": s*ds.f,
                   "seed": actual_seed, "crash_fraction": float(crash >= 0),
                   "steps_survived": args.steps if crash < 0 else crash, "wall_s": out["wall_s"],
                   **{m: None for m in METRICS[:4]}}
            if crash < 0:
                row.update(physical_metrics(out["positions"][:, 0].numpy(), out["velocities"][:, 0].numpy(), ds,
                                            ds.boxes[traj].numpy(), ref))
            for metric in METRICS:
                if row[metric] is not None and not np.isfinite(row[metric]):
                    row[metric] = None
            rows.append(row)
            print(f"{path.name}: start={i} seed={actual_seed} survived={row['steps_survived']} wall={out['wall_s']:.2f}s", flush=True)
    return {"checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "train_step": int(step),
            "stride_fs": ds.stride_fs, "head": model.head_type,
            "reference": {"temperature_K": ref["T"], "vacf_zero_fs": ref["zero"], "D_cm2_s": ref["D"]},
            "rollouts": rows, "metrics": bootstrap_summary(rows, args.bootstrap, args.bootstrap_seed)}


def write_report(report, out):
    out.mkdir(parents=True, exist_ok=True)
    (out/"metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    lines = ["# Multistart / multiseed evaluation", "", report["method"], "",
             "Physical metrics condition on survival; missing VACF zeros are excluded and counted. Errors are fractions; RDF L1 is in Å.", "",
             "| Checkpoint | Metric | Mean | Bootstrap 95% CI | Valid / total |", "|---|---|---:|---|---:|"]
    def fmt(v):
        return "null" if v is None else f"{v:.8g}"
    for ck in report["checkpoints"]:
        for name, m in ck["metrics"].items():
            lines.append(f"| {Path(ck['checkpoint']).name} | {name} | {fmt(m['mean'])} | [{fmt(m['ci95'][0])}, {fmt(m['ci95'][1])}] | {m['n_valid']} / {len(ck['rollouts'])} |")
    lines += ["", f"Measured total wall time: {report['wall_s']:.3f} s.", "",
              "These intervals quantify start-window sampling, conditional on these checkpoints and test trajectories. Overlapping windows and few starts limit coverage. Short-time diffusion need not be in the diffusive regime; this is not a release certification."]
    (out/"metrics.md").write_text("\n".join(lines)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--split", default="test", choices=("test", "val"))
    for name, default in [("starts", 3), ("seeds", 2), ("steps", 50), ("sampling_steps", 10), ("threads", 2), ("bootstrap", 2000)]:
        ap.add_argument("--"+name, type=int, default=default)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--bootstrap_seed", type=int, default=1234)
    args = ap.parse_args()
    if min(args.starts, args.seeds, args.sampling_steps, args.threads, args.bootstrap) < 1 or args.steps < 8:
        ap.error("counts must be positive; steps must be >=8 for dynamical metrics")
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    report = {"protocol": {k: str(v) if isinstance(v, Path) else [str(p) for p in v] if k == "checkpoints" else v for k, v in vars(args).items()},
              "method": "Per-rollout metrics; paired explicit RNG seeds across checkpoints. Percentile 95% cluster bootstrap of means over starts, retaining all seeds in each sampled start. Physical metrics use evaluate.py analysis conventions.",
              "checkpoints": []}
    for path in args.checkpoints:
        report["checkpoints"].append(evaluate_checkpoint(path, args))
        report["wall_s"] = time.perf_counter()-t0
        write_report(report, args.out)
    print(f"Wrote {args.out}/metrics.json and metrics.md", flush=True)


if __name__ == "__main__":
    main()
