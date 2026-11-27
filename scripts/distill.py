"""Distill an exported or training EquiTraj checkpoint using saved training context batches.

Input .pt: pos (B,k,N,3), vel, types (N,), masses (N,), box (B,3,3) or None.
Units are Å and Å/fs. Generate batches from the training split only.
Supports progressive step halving (4 -> 2 -> 1) and narrow student width/depth distillation.
"""
import argparse
import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from equitraj.distill import copy_normalization_stats, distillation_loss
from equitraj.model import make_model


def _build_student(teacher, base_cfg, student_channels=None, student_blocks=None):
    if student_channels is None and student_blocks is None:
        return copy.deepcopy(teacher).train().requires_grad_(True), dict(base_cfg)
    stu_cfg = dict(base_cfg)
    if student_channels is not None:
        stu_cfg["channels"] = int(student_channels)
        stu_cfg["heads"] = min(int(stu_cfg.get("heads", 4)), max(1, int(student_channels) // 8))
    if student_blocks is not None:
        nb = int(student_blocks)
        stu_cfg["n_spatial"] = nb
        stu_cfg["n_fusion"] = nb
        stu_cfg["n_denoiser"] = nb
    student = make_model(**stu_cfg).to(next(teacher.parameters()).device)
    copy_normalization_stats(student, teacher)
    return student.train().requires_grad_(True), stu_cfg


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", required=True, help="release or training checkpoint .pt")
    p.add_argument("--data", required=True, help="training contexts .pt")
    p.add_argument("--out", required=True)
    p.add_argument("--teacher-steps", type=int, default=4)
    p.add_argument("--student-steps", type=int, default=None, help="defaults to teacher-steps // 2")
    p.add_argument("--student-channels", type=int, default=None, help="optional narrower channel width")
    p.add_argument("--student-blocks", type=int, default=None, help="optional block count per stage")
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", default="cpu")
    p.add_argument("--supervised-weight", type=float, default=0.0)
    a = p.parse_args(argv)
    if a.iterations < 1:
        p.error("iterations must be positive")
    if a.student_steps is None and (a.teacher_steps < 2 or a.teacher_steps % 2):
        p.error("teacher-steps must be even and >= 2 when --student-steps is omitted")
    if a.student_steps is not None and (a.student_steps < 1 or a.teacher_steps < 1):
        p.error("teacher-steps and student-steps must be positive")

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        ck = torch.load(a.teacher, map_location="cpu", weights_only=False)
        raw_model_cfg = dict(ck["model_cfg"] if "model_cfg" in ck else ck["cfg"]["model"])
        k_ctx = int(ck["k"] if "k" in ck else ck["cfg"]["k"])
        stride_fs = float(ck.get("stride_fs", 1.0))
        build_kw = dict(raw_model_cfg, n_types=len(ck["elements"]), k=k_ctx, stride_fs=stride_fs)
        teacher = make_model(**build_kw).float().to(a.device)
        teacher.load_state_dict(ck["model"])
        teacher.eval().requires_grad_(False)
        student, stu_build_kw = _build_student(teacher, build_kw, a.student_channels, a.student_blocks)

        data = torch.load(a.data, map_location="cpu", weights_only=False)
        batches = data if isinstance(data, list) else [data]
        opt = torch.optim.AdamW(student.parameters(), lr=a.lr)
        stu_steps = a.student_steps if a.student_steps is not None else (a.teacher_steps // 2)

        for step in range(a.iterations):
            b = batches[step % len(batches)]
            pos = b["pos"].to(a.device, dtype=torch.float32)
            vel = b["vel"].to(a.device, dtype=torch.float32)
            types = b["types"].to(a.device, dtype=torch.long)
            masses = b["masses"].to(a.device, dtype=torch.float32)
            box = None if b.get("box") is None else b["box"].to(a.device, dtype=torch.float32)
            charges = None if b.get("charges") is None else b["charges"].to(a.device, dtype=torch.float32)
            ff_types = None if b.get("ff_types") is None else b["ff_types"].to(a.device, dtype=torch.long)
            opt.zero_grad(set_to_none=True)
            loss = distillation_loss(
                student, teacher, pos, vel, types, box, masses,
                teacher_steps=a.teacher_steps, student_steps=stu_steps,
                supervised_weight=a.supervised_weight,
                charges=charges, ff_types=ff_types, bonds=b.get("bonds"),
            )
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite distillation loss")
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(step, float(loss.detach()), flush=True)

        saved_model_cfg = {k: v for k, v in stu_build_kw.items() if k not in ("n_types", "k", "stride_fs")}
        out_ck = dict(ck)
        if "model_cfg" in out_ck:
            out_ck["model_cfg"] = saved_model_cfg
        elif "cfg" in out_ck and isinstance(out_ck["cfg"], dict):
            out_ck["cfg"] = {**out_ck["cfg"], "model": saved_model_cfg}
        out_ck.update(
            model={k: v.detach().cpu() for k, v in student.state_dict().items()},
            sampling_steps=stu_steps,
        )
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_ck, a.out)
    finally:
        torch.set_default_dtype(prev_dtype)


if __name__ == "__main__":
    main()
