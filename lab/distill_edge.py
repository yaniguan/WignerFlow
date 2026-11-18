"""Distillation plan for an edge-sized WignerFlow student (teacher -> fewer flow steps -> smaller net -> low precision).

    python -m lab.distill_edge --dry-run                 # print the plan, check files; no torch
    python -m lab.distill_edge --inspect-teacher         # also count teacher parameters (loads torch)

This script does not train. Stages with a command can be run by hand; stages marked
"todo" need code first. Each stage must pass the gate before the next one starts.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "distill_edge.json"
REQUIRED_KEYS = ("name", "system", "teacher", "gate", "stages", "targets")


def load_config(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise ValueError(f"config missing keys: {missing}")
    ids = [s["id"] for s in cfg["stages"]]
    if len(ids) != len(set(ids)):
        raise ValueError("stage ids must be unique")
    return cfg


def check_files(cfg):
    """Map every file the plan depends on to True/False (exists in this repo)."""
    paths = [cfg["teacher"]["checkpoint"], cfg["gate"]["script"]]
    for stage in cfg["stages"]:
        paths.extend(stage.get("requires", []))
    return {p: (ROOT / p).exists() for p in dict.fromkeys(paths)}


def plan_lines(cfg, files):
    lines = [f"plan: {cfg['name']} (system: {cfg['system']})",
             f"teacher: {cfg['teacher']['checkpoint']} at {cfg['teacher']['sampling_steps']} flow steps",
             f"gate: {cfg['gate']['criteria']}; {cfg['gate']['rule']}"]
    for i, stage in enumerate(cfg["stages"], 1):
        lines.append(f"{i}. {stage['id']} [{stage['status']}]: {stage['goal']}")
        if stage.get("command"):
            lines.append(f"   run: {stage['command']}")
        for req in stage.get("requires", []):
            lines.append(f"   needs {req}: {'ok' if files[req] else 'MISSING'}")
    t = cfg["targets"]
    lines.append(f"targets: student params {t['student_params']}; latency {t['latency']}; "
                 f"hardware {', '.join(t['edge_hardware'])}")
    lines.append("no results yet: nothing in this plan has been run.")
    return lines


def inspect_teacher(cfg):
    """Facts about the teacher file: size on disk and parameter count."""
    sys.path.insert(0, str(ROOT))
    from lab.surrogate_query import load_release

    path = ROOT / cfg["teacher"]["checkpoint"]
    model, ck = load_release(path)
    return {"checkpoint": cfg["teacher"]["checkpoint"], "file_mb": round(path.stat().st_size / 1e6, 2),
            "params": sum(p.numel() for p in model.parameters()), "train_step": ck["train_step"],
            "model_cfg": ck["model_cfg"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print the plan; no torch")
    mode.add_argument("--inspect-teacher", action="store_true", help="plan plus teacher facts (loads torch)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    files = check_files(cfg)
    print("\n".join(plan_lines(cfg, files)))
    if args.inspect_teacher:
        print(json.dumps(inspect_teacher(cfg), indent=2))
    return 0 if all(files.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
