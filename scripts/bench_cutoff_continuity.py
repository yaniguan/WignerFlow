"""Continuity of the model outputs when a neighbor crosses the cutoff.

    python scripts/bench_cutoff_continuity.py
    python scripts/bench_cutoff_continuity.py --out results/bench/cutoff_continuity.json

The model predicts (Δx, v_new) and has no energy or forces, so there is no force-consistency or NVE
energy check to run; this is the replacement. A small vacuum cluster (equitraj.bench.make_cutoff_probe)
has a probe atom on the +x axis at distance r from atom 0, and r is scanned on a uniform grid of
spacing h through the cutoff. Only the edge pair (0, probe) appears or disappears in the scan.

For each spacing h, the scan reports the largest output change over the grid step where the graph
changes ("crossing step") and over all other steps. A continuous output gives a crossing step that
shrinks like h; a jump gives a crossing step that stays at the same size as h -> 0. A control scan
the same width but 1 Å inside the cutoff shows the ordinary step size. Float64, untrained models.
"""

import argparse
import json
import os
import time

import torch

from equitraj.bench import jump_stats, make_cutoff_probe, make_model, scan_outputs


def fmt(s, key, spec):
    return "-" if s.get(key) is None else format(s[key], spec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacings", default="1e-2,1e-3,1e-4", help="grid spacings h in Å, comma separated")
    ap.add_argument("--points", type=int, default=40, help="grid points per scan: 3/4 inside the cutoff, 1/4 outside")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--sampling_steps", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=3, help="model initialisations per head")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    spacings = [float(h) for h in args.spacings.split(",")]
    t_start = time.time()
    rows = []
    print(f"{'head':>10} {'seed':>4} {'scan':>8} {'h':>7} | {'crossings':>9} {'max crossing':>12} "
          f"{'max other':>10} {'ratio':>8} {'rel jump':>9}", flush=True)
    for head in ["regression", "flow"]:
        for seed in range(args.seeds):
            model = make_model(head, torch.float64, seed=seed, channels=args.channels, cutoff=args.cutoff)
            base = make_cutoff_probe(seed=seed)
            g = torch.Generator().manual_seed(seed)
            noise = (model.com_free_noise(1, base["types"], base["masses"], generator=g, like=base["pos"])
                     if head == "flow" else None)
            for scan in ["cutoff", "control"]:
                for h in spacings:
                    # grid offset by h/3 so the cutoff never falls on a grid point
                    center = args.cutoff if scan == "cutoff" else args.cutoff - 1.0
                    radii = center + h / 3 + h * (torch.arange(args.points, dtype=torch.float64) - 0.75 * args.points)
                    outputs, n_edges = scan_outputs(model, base, radii, n_steps=args.sampling_steps, noise=noise)
                    s = jump_stats(outputs, n_edges)
                    s.update(head=head, seed=seed, scan=scan, h=h, r_min=radii[0].item(), r_max=radii[-1].item())
                    rows.append(s)
                    print(f"{head:>10} {seed:>4} {scan:>8} {h:>7.0e} | {s['n_crossing_steps']:>9} "
                          f"{fmt(s, 'max_crossing_step', '.3e'):>12} {fmt(s, 'max_smooth_step', '.3e'):>10} "
                          f"{fmt(s, 'crossing_to_max_smooth', '.1e'):>8} {fmt(s, 'crossing_rel', '.1e'):>9}", flush=True)
    print(f"total {time.time() - t_start:.1f} s", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "torch": torch.__version__, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
