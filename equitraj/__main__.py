"""Unified command-line entry point (`wignerflow` or `python -m wignerflow` / `python -m equitraj`)."""

import argparse
import sys


COMMANDS = {
    "train": "Train EquiTraj, PlainTraj, or TrajCastBaseline from a YAML config (`equitraj.train`).",
    "demo": "Run a standalone `.pt` release checkpoint without OpenMM or data files (`scripts.demo`).",
    "evaluate": "Run 10^4-step periodic NVE evaluation against HDF5 ground truth (`scripts.evaluate`).",
    "evaluate-molecule": "Run 10^4-step vacuum molecule evaluation (`scripts.evaluate_molecule`).",
    "multiseed": "Paired multi-start / multi-seed cluster-bootstrap evaluation (`equitraj.multiseed`).",
    "distill": "Progressive flow-step distillation 10 -> 4 -> 2 -> 1 (`scripts.distill`).",
    "export": "Export a training snapshot into a standalone v0/v1 `.pt` file (`scripts.export_ckpt`).",
}


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        ap = argparse.ArgumentParser(
            prog="wignerflow",
            description="WignerFlow: autoregressive E(3)-equivariant molecular dynamics without an integrator.",
        )
        sub = ap.add_subparsers(dest="command", metavar="<command>")
        for cmd, desc in COMMANDS.items():
            sub.add_parser(cmd, help=desc)
        ap.print_help()
        return 0

    cmd, rest = args[0], args[1:]
    if cmd not in COMMANDS:
        print(f"wignerflow: unknown command {cmd!r}. Choose from: {', '.join(COMMANDS)}", file=sys.stderr)
        return 2

    saved = sys.argv
    sys.argv = [f"wignerflow {cmd}", *rest]
    try:
        if cmd == "train":
            from equitraj.train import main as run_cmd
            run_cmd()
        elif cmd == "demo":
            from scripts.demo import main as run_cmd
            run_cmd()
        elif cmd == "evaluate":
            from scripts.evaluate import main as run_cmd
            run_cmd(rest)
        elif cmd == "evaluate-molecule":
            from scripts.evaluate_molecule import main as run_cmd
            run_cmd(rest)
        elif cmd == "multiseed":
            from equitraj.multiseed import main as run_cmd
            run_cmd()
        elif cmd == "distill":
            from scripts.distill import main as run_cmd
            run_cmd()
        elif cmd == "export":
            from scripts.export_ckpt import main as run_cmd
            run_cmd()
    finally:
        sys.argv = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
