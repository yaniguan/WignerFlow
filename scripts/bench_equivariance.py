"""Equivariance error of the model outputs under random rotations and reflections, in float32 and float64.

    python scripts/bench_equivariance.py                         # untrained B_S3a-size models, both heads
    python scripts/bench_equivariance.py --ckpt checkpoints/argon_flow_v0.pt   # also the v0 release checkpoint
    python scripts/bench_equivariance.py --out results/bench/equivariance.json

For each head (regression, flow), dtype, transform (proper rotation, improper rotation = rotation times
inversion) and system (vacuum, periodic box rotated with the atoms), runs --trials random transforms and
input seeds and compares f(R x + t) with R f(x) for Δx and v_new. Prints one row per setting: worst
max error and average mean error over trials, absolute (Å, Å/fs) and relative to the largest output.
The layers are O(3)-equivariant (parity=True), so reflections should be as exact as rotations.
"""

import argparse
import json
import os
import sys
import time

import torch

from equitraj.bench import equivariance_error, make_liquid, make_model, random_orthogonal, summarize_trials


def bench(model, inputs_fn, trials, sampling_steps, reflect, seed):
    g = torch.Generator().manual_seed(seed)
    res = []
    for t in range(trials):
        inp = inputs_fn(t)
        rot = random_orthogonal(reflect, generator=g)
        shift = 5.0 * torch.randn(3, generator=g, dtype=torch.float64)
        res.append(equivariance_error(model, inp, rot, shift, n_steps=sampling_steps, generator=g))
    return summarize_trials(res)


def print_row(r):
    s = r["stats"]
    print(f"{r['model']:>10} {r['head']:>10} {r['dtype']:>8} {r['transform']:>10} {r['system']:>8} | "
          f"dx max {s['dx']['max_abs']:.2e} ({s['dx']['max_rel']:.1e} rel) mean {s['dx']['mean_abs']:.2e} | "
          f"v max {s['v']['max_abs']:.2e} ({s['v']['max_rel']:.1e} rel) mean {s['v']['mean_abs']:.2e}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--atoms", type=int, default=64, help="atoms per system (64 is the smallest argon box wider than 2 x cutoff 6 Å)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sampling_steps", type=int, default=2, help="flow-matching Euler steps per prediction")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--ckpt", default=None, help="also test this release checkpoint on its bundled 256-atom context")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write all rows as JSON here")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t_start = time.time()
    rows = []

    for head in ["regression", "flow"]:
        for dtype in [torch.float32, torch.float64]:
            model = make_model(head, dtype, seed=args.seed, k=args.k, channels=args.channels, cutoff=args.cutoff)
            for system in ["vacuum", "periodic"]:
                def inputs_fn(t, system=system, dtype=dtype):
                    return make_liquid(args.atoms, k=args.k, batch=args.batch, periodic=system == "periodic",
                                       seed=args.seed + t, dtype=dtype)
                for reflect in [False, True]:
                    stats = bench(model, inputs_fn, args.trials, args.sampling_steps, reflect, args.seed)
                    rows.append({"model": "untrained", "head": head, "dtype": str(dtype).replace("torch.", ""),
                                 "transform": "reflection" if reflect else "rotation", "system": system,
                                 "atoms": args.atoms, "batch": args.batch, "k": args.k, "trials": args.trials,
                                 "stats": stats})
                    print_row(rows[-1])

    if args.ckpt:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from demo import load_release  # noqa: PLC0415

        for dtype in [torch.float32, torch.float64]:
            model, ck = load_release(args.ckpt, "cpu")
            model = model.to(dtype)
            d = ck["demo"]
            inp = {"pos": d["pos"][None].to(dtype), "vel": d["vel"][None].to(dtype), "types": ck["types"],
                   "masses": ck["masses"].to(dtype), "box": None if d["box"] is None else d["box"][None].to(dtype)}
            for reflect in [False, True]:
                stats = bench(model, lambda t, inp=inp: inp, args.trials, args.sampling_steps, reflect, args.seed)
                rows.append({"model": os.path.basename(args.ckpt), "head": model.head_type,
                             "dtype": str(dtype).replace("torch.", ""), "transform": "reflection" if reflect else "rotation",
                             "system": "periodic" if d["box"] is not None else "vacuum", "atoms": len(ck["types"]),
                             "batch": 1, "k": inp["pos"].shape[1], "trials": args.trials, "stats": stats})
                print_row(rows[-1])

    print(f"total {time.time() - t_start:.1f} s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
