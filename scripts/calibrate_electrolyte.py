"""Differentiable Einstein-Helfand / Onsager post-training to align WignerFlow conductivity to experiment.

Anchors local phase-space dynamics to the pre-trained `MU-PFF` teacher checkpoint while nudging collective
ion-current correlations (`<|sum_i q_i dX_i|^2>`) toward experimental conductivity (e.g. `8.60 mS/cm` for
`best-napf6-ec-dmc-m4a`), bridging the discrete `-16%` (`MU-PFF`) vs `+50.9%` (`ByteFF-pol`) gap.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from equitraj.electrolyte import transport_calibration_loss
from equitraj.evaluation import load_model
from equitraj.layers import ATTENTION_VERSION
from equitraj.train import atomic_save


def main(argv=None):
    ap = argparse.ArgumentParser(description="Calibrate WignerFlow electrolyte conductivity to experiment.")
    ap.add_argument("--run", default=None, help="Pre-trained WignerFlow run directory")
    ap.add_argument("--ckpt", default="best_stable.pt", help="Teacher checkpoint filename or path")
    ap.add_argument("--out_dir", required=True, help="Output directory for calibrated checkpoint")
    ap.add_argument("--target_sigma", type=float, default=8.60, help="Target experimental sigma in mS/cm")
    ap.add_argument("--temperature_K", type=float, default=298.15)
    ap.add_argument("--steps", type=int, default=400, help="Post-training gradient steps")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--unroll_steps", type=int, default=6)
    ap.add_argument("--sampling_steps", type=int, default=2)
    ap.add_argument("--anchor_weight", type=float, default=2.0, help="Phase-space teacher anchor weight")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--split", default="train")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)

    ckpt_arg = args.ckpt
    if args.run is not None and not os.path.isfile(ckpt_arg):
        if not os.path.isfile(os.path.join(args.run, ckpt_arg)) and ckpt_arg == "best_stable.pt":
            for fallback in ("best.pt", "ckpt.pt"):
                if os.path.isfile(os.path.join(args.run, fallback)):
                    ckpt_arg = fallback
                    break

    device = torch.device(args.device)
    student, cfg, ds, src_step = load_model(args.run, ckpt_arg, device, args.split, args.data_dir)
    if ds.charges is None or float(ds.charges.abs().max().item()) < 1e-6:
        raise ValueError("Dataset has no non-zero ionic charges in pff_metadata; cannot calibrate conductivity.")

    teacher = copy.deepcopy(student).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    student.train()

    types = ds.types.to(device)
    masses = ds.masses.to(device)
    charges = ds.charges.to(device)
    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.0)
    gen = torch.Generator().manual_seed(int(cfg.get("seed", 0)))
    loader = torch.utils.data.DataLoader(
        ds, batch_size=min(args.batch_size, len(ds)), shuffle=True, drop_last=True, generator=gen,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    it = iter(loader)
    last_loss = None
    for s in range(1, args.steps + 1):
        try:
            pos_b, vel_b, box_b = next(it)
        except StopIteration:
            it = iter(loader)
            pos_b, vel_b, box_b = next(it)
        pos_b = pos_b[:, :cfg["k"]].to(device)
        vel_b = vel_b[:, :cfg["k"]].to(device)
        box_b = box_b.to(device)

        opt.zero_grad(set_to_none=True)
        loss = transport_calibration_loss(
            student, pos_b, vel_b, types, box_b, masses, charges,
            target_sigma_mS_cm=args.target_sigma, temperature_K=args.temperature_K,
            unroll_steps=args.unroll_steps, sampling_steps=args.sampling_steps,
            teacher_model=teacher, anchor_weight=args.anchor_weight,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        opt.step()
        last_loss = float(loss.item())

    student.eval()
    payload = {
        "model": student.state_dict(),
        "step": int(src_step) + int(args.steps),
        "cfg": cfg,
        "elements": ds.elements,
        "attention_version": ATTENTION_VERSION,
        "calibration": {
            "target_sigma_mS_cm": args.target_sigma,
            "steps": args.steps,
            "anchor_weight": args.anchor_weight,
            "final_loss": last_loss,
        },
    }
    for fname in ("best_stable.pt", "best.pt", "ckpt.pt"):
        atomic_save(payload, str(out_dir / fname))
    print(json.dumps({"out_dir": str(out_dir), "steps": args.steps, "final_loss": last_loss}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
