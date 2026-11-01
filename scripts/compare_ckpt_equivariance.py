"""Measure pathwise O(3) equivariance of released flow checkpoints on shared test windows.

Rotate positions, velocities, periodic box AND vector-valued flow noise together.
Float64 means float32 checkpoint weights and saved data promoted to float64.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from equitraj.dataset import TrajWindows
from equitraj.model import EquiTraj


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoints', nargs='+', required=True)
    ap.add_argument('--data', default='data/argon')
    ap.add_argument('--out', default='results/argon/cpu_reduced/equivariance.json')
    ap.add_argument('--trials', type=int, default=3)
    ap.add_argument('--sampling_steps', type=int, default=10)
    ap.add_argument('--seed', type=int, default=20261029)
    ap.add_argument('--threads', type=int, default=1)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t0 = time.perf_counter()
    results = []
    for path in args.checkpoints:
        ck = torch.load(path, map_location='cpu', weights_only=True)
        ds = TrajWindows(args.data, 'test', ck['stride_steps'], ck['k'])
        for dtype in (torch.float32, torch.float64):
            # Set before construction so constant buffers also start at requested precision.
            torch.set_default_dtype(dtype)
            model = EquiTraj(n_types=len(ck['elements']), k=ck['k'],
                             stride_fs=ck['stride_fs'], **ck['model_cfg']).to(dtype).eval()
            model.load_state_dict(ck['model'])
            for trial in range(args.trials):
                torch.manual_seed(args.seed + trial)
                index = trial * (len(ds) - 1) // max(1, args.trials - 1)
                pos, vel, box = ds[index]
                pos, vel = pos[:ck['k']][None].to(dtype), vel[:ck['k']][None].to(dtype)
                box, masses = box[None].to(dtype), ds.masses.to(dtype)
                # QR in double precision, then cast identically for both checkpoints.
                q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
                q = q * torch.linalg.det(q)  # det +1
                noise = model.com_free_noise(1, ds.types, masses, like=pos)
                dx, v = model.predict(pos, vel, ds.types, box, masses,
                                      n_steps=args.sampling_steps, noise=noise)
                for reflect in (False, True):
                    r = (-q if reflect else q).to(dtype)
                    dx_r, v_r = model.predict(pos @ r.T, vel @ r.T, ds.types,
                                              box @ r.T, masses,
                                              n_steps=args.sampling_steps, noise=noise @ r.T)
                    edx = (dx_r - dx @ r.T).abs().max().item()
                    ev = (v_r - v @ r.T).abs().max().item()
                    results.append(dict(checkpoint=path, dtype=str(dtype), trial=trial,
                                        window_index=index, reflection=reflect,
                                        determinant=torch.linalg.det(r).item(),
                                        max_abs_dx_error_A=edx,
                                        max_abs_velocity_error_A_per_fs=ev,
                                        max_abs_output_error=max(edx, ev)))
                    print(json.dumps(results[-1]), flush=True)
    torch.set_default_dtype(torch.float32)
    summary = []
    for path in args.checkpoints:
        for dtype in ('torch.float32', 'torch.float64'):
            rows = [r for r in results if r['checkpoint'] == path and r['dtype'] == dtype]
            summary.append(dict(checkpoint=path, dtype=dtype,
                                max_abs_dx_error_A=max(r['max_abs_dx_error_A'] for r in rows),
                                max_abs_velocity_error_A_per_fs=max(r['max_abs_velocity_error_A_per_fs'] for r in rows),
                                max_abs_output_error=max(r['max_abs_output_error'] for r in rows)))
    payload = dict(seed=args.seed, trials=args.trials, sampling_steps=args.sampling_steps,
                   threads=args.threads, data=args.data, split='test',
                   method='Rotate box and COM-free flow noise with inputs; compare dx and v_new with rotated original outputs.',
                   checkpoint_sha256={p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in args.checkpoints},
                   summary=summary, measurements=results, wall_s=time.perf_counter()-t0)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
