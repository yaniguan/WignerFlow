"""Block-wise stability timeline for an equitraj HDF5 trajectory (OpenMM units: nm, nm/ps, ps).

    python scripts/stability_timeline.py rollout.h5 --blocks 20 [--reference gt.h5] [--out report.json]
"""
import argparse
import json

from equitraj.timeline import stability_timeline, timeline_markdown
from equitraj.trajfile import read_traj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("traj")
    p.add_argument("--reference")
    p.add_argument("--blocks", type=int, default=10)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--max-temp-dev", type=float, default=50.0)
    p.add_argument("--max-rdf-l1", type=float, default=0.5)
    p.add_argument("--min-distance", type=float, default=0.5)
    p.add_argument("--target-temperature", type=float)
    p.add_argument("--out")
    a = p.parse_args()
    d = read_traj(a.traj, stride=a.stride)
    ref = read_traj(a.reference, stride=a.stride) if a.reference else None
    rep = stability_timeline(
        d["positions"] * 10.0, d["velocities"] * 0.01, d["masses"], d["time"] * 1000.0, d["box"] * 10.0,
        n_blocks=a.blocks, potential_energy=d["potential_energy"],
        reference_positions=None if ref is None else ref["positions"] * 10.0,
        reference_box=None if ref is None else ref["box"] * 10.0,
        max_temp_dev=a.max_temp_dev, max_rdf_l1=a.max_rdf_l1, min_distance=a.min_distance,
        target_temperature=a.target_temperature)
    print(timeline_markdown(rep))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()
