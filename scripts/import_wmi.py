"""Import `02_Na_MD` (`MU-PFF` / WMI) build directories and trajectories into WignerFlow HDF5 format.

Examples:
    # Inspect build topology, LP charge folding, and species counts (no DCD required):
    wignerflow import-wmi --inspect_only \
        --build_dirs /path/to/best-napf6-ec-dmc-m4a__r1-npt /path/to/best-napf6-ec-dmc-m4a__r1 /path/to/best-napf6-ec-dmc-m4a__r2

    # Import 3 replicas (r1, r1-npt -> train; r2 -> val/test) at 610-node molecular resolution:
    wignerflow import-wmi \
        --build_dirs /data/build/best-napf6-ec-dmc-m4a__r1 \
                     /data/build/best-napf6-ec-dmc-m4a__r1-npt \
                     /data/build/best-napf6-ec-dmc-m4a__r2 \
        --trajs /data/pvc/md_Na/production/best-napf6-ec-dmc-m4a__r1/output/production3_200fs.dcd \
                /data/pvc/md_Na/production/best-napf6-ec-dmc-m4a__r1-npt/output/production3_200fs.dcd \
                /data/pvc/md_Na/production/best-napf6-ec-dmc-m4a__r2/output/production3_200fs.dcd \
        --resolution molecular --dt_frame_fs 200.0 \
        --out_dir data/napf6_ec_dmc_mol200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from equitraj.electrolyte import import_wmi_replica, load_wmi_build_dir


def _parse_int_list(text):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Import 02_Na_MD polarizable electrolyte replicas into WignerFlow.")
    ap.add_argument("--build_dirs", nargs="+", required=True, help="One or more 02_Na_MD/build_outputs/<sys> paths")
    ap.add_argument("--trajs", nargs="*", default=None, help="Matching .dcd/.npz/.h5 trajectory files")
    ap.add_argument("--state_logs", nargs="*", default=None, help="Optional OpenMM state CSV logs for box lengths")
    ap.add_argument("--out_dir", default=None, help="Output WignerFlow dataset directory")
    ap.add_argument("--resolution", choices=("molecular", "polar_beads", "all_atom"), default="molecular")
    ap.add_argument("--dt_frame_fs", type=float, default=200.0, help="Saved trajectory frame spacing in fs")
    ap.add_argument("--no_canonicalize_npt_box", action="store_true",
                    help="Keep raw fluctuating NPT box(t) instead of rigid-centroid canonicalization")
    ap.add_argument("--unwrap", action="store_true", help="Unwrap PBC jumps (off by default for 02_Na_MD DCDs)")
    ap.add_argument("--train_ids", default=None, help="Comma-separated train replica indices (default: all except last)")
    ap.add_argument("--val_ids", default=None, help="Comma-separated val replica indices (default: last replica)")
    ap.add_argument("--test_ids", default=None, help="Comma-separated test replica indices (default: last replica)")
    ap.add_argument("--inspect_only", action="store_true", help="Only inspect build_dirs and print JSON summary")
    args = ap.parse_args(argv)

    if args.inspect_only:
        summaries = []
        for bdir in args.build_dirs:
            w = load_wmi_build_dir(bdir)
            summaries.append({
                "build_dir": w["build_dir"],
                "temperature_K": w["temperature_K"],
                "n_total_sites": w["n_total_sites"],
                "n_virtual_sites": w["n_virtual_sites"],
                "n_phys_atoms": w["n_phys_atoms"],
                "n_molecules": w["n_molecules"],
                "equilibrium_box_L_A": round(w["equilibrium_box_L_A"], 4),
                "total_mass_amu": round(w["total_mass_amu"], 3),
                "species_table": w["species_table"],
            })
        print(json.dumps(summaries if len(summaries) > 1 else summaries[0], indent=2))
        return 0

    if not args.trajs or args.out_dir is None:
        ap.error("--trajs and --out_dir are required unless --inspect_only is set")

    n_rep = len(args.trajs)
    build_dirs = args.build_dirs * n_rep if len(args.build_dirs) == 1 else args.build_dirs
    if len(build_dirs) != n_rep:
        ap.error(f"Expected 1 or {n_rep} --build_dirs to match {n_rep} --trajs, got {len(args.build_dirs)}")
    logs = [None] * n_rep
    if args.state_logs:
        if len(args.state_logs) != n_rep:
            ap.error(f"--state_logs length ({len(args.state_logs)}) must match --trajs ({n_rep})")
        logs = args.state_logs

    canonical = not args.no_canonicalize_npt_box
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i in range(n_rep):
        h5_path = import_wmi_replica(
            build_dirs[i], args.trajs[i], out_dir,
            traj_index=i, resolution=args.resolution, dt_frame_fs=args.dt_frame_fs,
            canonicalize_npt_box=canonical, state_log=logs[i], unwrap=args.unwrap,
        )
        written.append(str(h5_path))

    if n_rep == 1:
        default_train, default_val, default_test = [0], [0], [0]
    else:
        default_train = list(range(n_rep - 1))
        default_val = [n_rep - 1]
        default_test = [n_rep - 1]

    cfg = {
        "system": f"wmi_{args.resolution}",
        "resolution": args.resolution,
        "dt_fs": float(args.dt_frame_fs),
        "save_every": 1,
        "allow_variable_box": not canonical,
        "split": {
            "train": _parse_int_list(args.train_ids) if args.train_ids else default_train,
            "val": _parse_int_list(args.val_ids) if args.val_ids else default_val,
            "test": _parse_int_list(args.test_ids) if args.test_ids else default_test,
        },
    }
    cfg_path = out_dir / "config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir), "trajectories": written, "config": cfg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
