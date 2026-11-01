"""Import external MD as traj_NNN.h5 plus a trajectory split config."""
import argparse
import json

from equitraj.adapters import import_trajectory, write_data_config


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("src", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--topology")
    ap.add_argument("--format", choices=["mdtraj", "ase", "generic"])
    ap.add_argument("--unit", choices=["angstrom", "nm"])
    ap.add_argument("--velocity-unit", choices=["angstrom/fs", "angstrom/ps", "nm/ps"])
    ap.add_argument("--velocity-source", nargs="+")
    ap.add_argument("--dt-fs", type=float, required=True, help="integrator step, not frame interval")
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--system", default="imported")
    ap.add_argument("--keep-drude", action="store_true")
    ap.add_argument("--nonperiodic", action="store_true")
    ap.add_argument("--already-unwrapped", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split", help='JSON object with train/val/test trajectory ID lists')
    args = ap.parse_args()
    if args.velocity_source and len(args.velocity_source) != len(args.src):
        ap.error("provide one --velocity-source per input")
    for i, src in enumerate(args.src):
        path = import_trajectory(src, args.out_dir, topology=args.topology, format=args.format,
                                 unit=args.unit, velocity_unit=args.velocity_unit,
                                 velocity_source=args.velocity_source[i] if args.velocity_source else None,
                                 dt_fs=args.dt_fs, save_every=args.save_every, traj_index=i,
                                 system=args.system, periodic=False if args.nonperiodic else None,
                                 drop_drude=not args.keep_drude, unwrap=not args.already_unwrapped,
                                 overwrite=args.overwrite)
        print(path)
    write_data_config(args.out_dir, len(args.src), json.loads(args.split) if args.split else None)


if __name__ == "__main__":
    main()
