"""Compare two repo HDF5 trajectories; emit Markdown and optional full JSON."""
import argparse
import json
from pathlib import Path

from equitraj.trajfile import read_traj
from equitraj.units import NM_TO_A, NM_PER_PS_TO_A_PER_FS
from equitraj.validation import validate, report_markdown


def load(path):
    d = read_traj(path)
    d["positions"] *= NM_TO_A
    d["velocities"] *= NM_PER_PS_TO_A_PER_FS
    d["box"] = d["box"] * NM_TO_A if d["attrs"].get("periodic", True) else None
    d["dt_fs"] = float(d["attrs"]["dt_fs"]) * int(d["attrs"]["save_every"])
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rollout")
    ap.add_argument("reference")
    ap.add_argument("--r-max", type=float)
    ap.add_argument("--n-bins", type=int, default=100)
    ap.add_argument("--max-lag", type=int)
    ap.add_argument("--fit-window-fs", type=float, nargs=2)
    ap.add_argument("--charges", help='JSON mapping, e.g. {"Na":1,"Cl":-1}')
    ap.add_argument("--temperature-K", type=float)
    ap.add_argument("--dof", type=int)
    ap.add_argument("--min-distance", type=float)
    ap.add_argument("--max-temperature", type=float)
    ap.add_argument("--wrapped", action="store_true")
    ap.add_argument("--json-out")
    ap.add_argument("--markdown-out")
    args = ap.parse_args()
    result = validate(load(args.rollout), load(args.reference), r_max=args.r_max, n_bins=args.n_bins,
                      max_lag=args.max_lag, fit_window_fs=args.fit_window_fs,
                      charges=json.loads(args.charges) if args.charges else None,
                      temperature_K=args.temperature_K, dof=args.dof, wrapped=args.wrapped,
                      min_distance=args.min_distance, max_temperature=args.max_temperature)
    markdown = report_markdown(result)
    print(markdown)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    if args.markdown_out:
        Path(args.markdown_out).write_text(markdown)


if __name__ == "__main__":
    main()
