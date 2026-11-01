"""CPU forward/backward time and peak memory vs number of atoms and number of context frames.

    python scripts/bench_speed_memory.py
    python scripts/bench_speed_memory.py --atoms 64,128,256 --frames 1,2,3,5,8 --out results/bench/speed_memory.json

Two sweeps on synthetic argon-like inputs (jittered lattice at the argon density, periodic box):
  * atoms in --atoms at k = --k frames;
  * frames in --frames at --frames_atoms atoms.
Each setting runs in a fresh subprocess (equitraj.bench.bench_config_isolated), so the peak resident
memory is not polluted by earlier settings; a setting that runs out of memory is reported as an error
and the sweep goes on. Reported per setting: training forward (model.loss) and backward wall time,
one no-grad prediction step with --sampling_steps flow steps, and peak RSS above the RSS after setup.
"""

import argparse
import json
import os
import time

import torch

from equitraj.bench import bench_config_isolated


def ints(s):
    return [int(x) for x in s.split(",") if x]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atoms", type=ints, default=ints("64,128,256"))
    ap.add_argument("--k", type=int, default=3, help="frames for the atom sweep")
    ap.add_argument("--frames", type=ints, default=ints("1,2,3,5,8"))
    ap.add_argument("--frames_atoms", type=int, default=64, help="atoms for the frame sweep")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--head", default="flow", choices=["flow", "regression"])
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--sampling_steps", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    settings = [("atoms", n, args.k) for n in args.atoms] + [("frames", args.frames_atoms, k) for k in args.frames]
    print(f"torch {torch.__version__}, {args.threads} threads, head {args.head}, channels {args.channels}, "
          f"cutoff {args.cutoff} Å, batch {args.batch}", flush=True)
    print(f"{'sweep':>6} {'atoms':>6} {'frames':>6} {'edges/frame':>11} {'fwd s':>8} {'bwd s':>8} "
          f"{'predict s':>9} {'peak MiB':>9}", flush=True)
    rows = []
    t_start = time.time()
    for sweep, n, k in settings:
        r = bench_config_isolated(n_atoms=n, k=k, batch=args.batch, head=args.head, repeats=args.repeats,
                                  warmup=args.warmup, threads=args.threads, sampling_steps=args.sampling_steps,
                                  channels=args.channels, cutoff=args.cutoff)
        r.update(sweep=sweep, n_atoms=n, frames=k)
        rows.append(r)
        if "error" in r:
            print(f"{sweep:>6} {n:>6} {k:>6}  error: {r['error']}", flush=True)
        else:
            print(f"{sweep:>6} {n:>6} {k:>6} {r['edges_per_frame']:>11.0f} {r['forward_s']:>8.3f} {r['backward_s']:>8.3f} "
                  f"{r['predict_s']:>9.3f} {r['peak_delta_mb']:>9.0f}", flush=True)
    print(f"total {time.time() - t_start:.1f} s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
