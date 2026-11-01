"""Distil an exported EquiTraj checkpoint using saved training context batches.

Input .pt: pos (B,k,N,3), vel, types (N,), masses (N,), box (B,3,3) or None.
Units are Å and Å/fs. Generate batches from the training split only.
"""
import argparse
import copy
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from equitraj.model import EquiTraj
from equitraj.distill import distillation_loss


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher', required=True, help='export_ckpt.py format')
    p.add_argument('--data', required=True, help='training contexts .pt')
    p.add_argument('--out', required=True)
    p.add_argument('--teacher-steps', type=int, default=4)
    p.add_argument('--iterations', type=int, default=1000)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--supervised-weight', type=float, default=0.0)
    a = p.parse_args()
    if a.iterations < 1 or a.teacher_steps < 2 or a.teacher_steps % 2:
        p.error('iterations must be positive and teacher-steps must be even and >= 2')
    ck = torch.load(a.teacher, map_location=a.device, weights_only=False)
    cfg = dict(ck['model_cfg'])
    cfg.update(n_types=len(ck['elements']), k=ck['k'], stride_fs=ck['stride_fs'])
    teacher = EquiTraj(**cfg).to(a.device)
    teacher.load_state_dict(ck['model'])
    teacher.eval().requires_grad_(False)
    student = copy.deepcopy(teacher).train().requires_grad_(True)
    data = torch.load(a.data, map_location=a.device, weights_only=False)
    batches = data if isinstance(data, list) else [data]
    opt = torch.optim.AdamW(student.parameters(), lr=a.lr)
    for step in range(a.iterations):
        b = batches[step % len(batches)]
        opt.zero_grad(set_to_none=True)
        loss = distillation_loss(student, teacher, b['pos'], b['vel'], b['types'],
                                 b.get('box'), b['masses'], a.teacher_steps,
                                 supervised_weight=a.supervised_weight)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite distillation loss')
        loss.backward()
        opt.step()
        if step % 100 == 0: print(step, float(loss.detach()), flush=True)
    ck.update(model=student.state_dict(), sampling_steps=a.teacher_steps // 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, a.out)


if __name__ == '__main__':
    main()
