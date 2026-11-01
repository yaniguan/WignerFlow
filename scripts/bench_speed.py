"""Synthetic compute throughput sweeps (random weights; no accuracy claim)."""
import argparse
import gc
import itertools
import json
import math
from pathlib import Path
import time

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from equitraj.neighbors import brute_force, cell_list, VerletList


def synthetic(n, system='argon', device='cpu'):
    """Tile conventional FCC/rocksalt cells, truncating final cell to exact N.

    Cubic periodic box retains complete tile volume; non-cubic counts therefore
    contain vacancies. Both engines receive identical positions and box.
    """
    if n < 2:
        raise ValueError('at least two atoms required')
    basis = torch.tensor([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]],
                         dtype=torch.float32, device=device)
    if system == 'nacl':
        basis = torch.cat((basis, torch.remainder(basis + .5, 1)))
    side = math.ceil((n / len(basis)) ** (1 / 3))
    axis = torch.arange(side, device=device)
    origins = torch.cartesian_prod(axis, axis, axis).float()
    spacing = 6.4 if system == 'argon' else 6.0
    pos = ((origins[:, None] + basis) * spacing).reshape(-1, 3)[:n]
    types = (torch.arange(n, device=device) % 8 >= 4).long() if system == 'nacl' else torch.zeros(n, dtype=torch.long, device=device)
    mass = torch.where(types == 0, 22.99, 35.45) if system == 'nacl' else pos.new_full((n,), 39.948)
    return pos, torch.eye(3, device=device) * (side * spacing), types, mass


def sync(device):
    if torch.device(device).type == 'cuda': torch.cuda.synchronize(device)


def timed(fn, steps, warmup, device):
    for _ in range(warmup): fn()
    sync(device)
    start = time.perf_counter()
    for _ in range(steps): fn()
    sync(device)
    return time.perf_counter() - start


def openmm_bench(a, pos, box, types, masses):
    import openmm as mm
    import openmm.unit as u
    system = mm.System()
    force = mm.NonbondedForce()
    force.setNonbondedMethod(force.PME if a.system == 'nacl' else force.CutoffPeriodic)
    force.setCutoffDistance(min(.9, float(box.diag().min()) / 20 * .95) * u.nanometer)
    for t, m in zip(types.tolist(), masses.tolist()):
        system.addParticle(m)
        force.addParticle((1 if t == 0 else -1) if a.system == 'nacl' else 0,
                          .25 if a.system == 'nacl' else .34,
                          .1 if a.system == 'nacl' else .997)
    system.addForce(force)
    system.setDefaultPeriodicBoxVectors(*[mm.Vec3(*row) for row in (box.cpu() / 10).tolist()])
    integrator = mm.VerletIntegrator(a.dt_fs * u.femtosecond)
    platform = mm.Platform.getPlatformByName(a.platform)
    props = {'Precision': 'mixed'} if a.platform in {'CUDA', 'OpenCL'} else {}
    context = mm.Context(system, integrator, platform, props)
    try:
        context.setPositions(pos.cpu().numpy() / 10)
        context.setVelocitiesToTemperature(100 if a.system == 'argon' else 300, a.seed)
        integrator.step(a.openmm_warmup)
        context.getState(getEnergy=True)
        start = time.perf_counter()
        integrator.step(a.openmm_steps)
        state = context.getState(getEnergy=True)
        wall = time.perf_counter() - start
        energy = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
        if not math.isfinite(energy): raise RuntimeError('nonfinite OpenMM energy')
        sps = a.openmm_steps / wall
        return dict(engine='openmm', atoms=len(pos), platform=a.platform, steps_per_s=sps,
                    ns_per_day=sps * a.dt_fs * .0864, peak_memory_bytes=None, step_fs=a.dt_fs)
    finally:
        del context, integrator


def model_bench(a, n, precision, sampling, replicas, compile_on, neighbor):
    from equitraj.model import EquiTraj
    from equitraj.fast import FastRollout
    torch.manual_seed(a.seed)
    pos, box, types, mass = synthetic(n, a.system, a.device)
    model = EquiTraj(n_types=2 if a.system == 'nacl' else 1, channels=a.channels,
                     lmax=a.lmax, heads=a.heads, n_rbf=a.n_rbf, cutoff=a.cutoff,
                     k=a.k, stride_fs=a.stride_fs, n_spatial=a.n_spatial,
                     n_fusion=a.n_fusion, n_denoiser=a.n_denoiser).to(a.device).float().eval()
    model.dx_std.fill_(.1)
    model.v_std.fill_(.001)
    model.out_dx_std.fill_(.001)
    model.out_v_std.fill_(.00001)
    fast = FastRollout(model, precision=precision, sampling_steps=sampling, neighbor=neighbor,
                       compile=bool(compile_on), compile_mode=a.compile_mode,
                       cuda_graphs=a.cuda_graphs, skin=a.skin)
    del model
    x = pos[None, None].expand(replicas, a.k, n, 3).clone().double()
    v = torch.zeros_like(x, dtype=torch.float32)
    bx = box[None].expand(replicas, 3, 3)
    @torch.no_grad()
    def step():
        nonlocal x, v
        dx, vn = fast.predict(x, v, types, bx, mass)
        x = torch.cat((x[:, 1:], (x[:, -1] + dx.double())[:, None]), 1)
        v = torch.cat((v[:, 1:], vn[:, None]), 1)
    if torch.device(a.device).type == 'cuda': torch.cuda.reset_peak_memory_stats(a.device)
    wall = timed(step, a.steps, a.warmup, a.device)
    if not bool(torch.isfinite(x).all() and torch.isfinite(v).all()):
        raise RuntimeError('nonfinite random-model trajectory')
    sps = a.steps / wall
    result = dict(steps_per_s=sps, ns_per_day=sps * a.stride_fs * .0864,
                  ns_per_day_aggregate=sps * a.stride_fs * .0864 * replicas,
                  effective_precision=fast.effective_precision,
                  compile_active=fast._compiled is not None, cuda_graphs_active=fast._capture is not None,
                  peak_memory_bytes=(torch.cuda.max_memory_allocated(a.device)
                                     if torch.device(a.device).type == 'cuda' else None),
                  step_fs=a.stride_fs, random_init=True)
    if a.trace and not getattr(a, '_traced', False):
        from equitraj.profiling import torch_profile
        result['profile'] = torch_profile(step, 1, a.trace)
        a._traced = True
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro', action='store_true')
    p.add_argument('--sizes', nargs='+', type=int, default=[128, 512, 2048, 8192, 16384])
    p.add_argument('--system', choices=['argon', 'nacl'], default='argon')
    p.add_argument('--device', default='cpu')
    p.add_argument('--platform', default='CPU')
    p.add_argument('--precisions', nargs='+', choices=['fp32', 'bf16', 'fp8'], default=['fp32', 'bf16', 'fp8'])
    p.add_argument('--sampling-steps', nargs='+', type=int, default=[1, 2, 10])
    p.add_argument('--replicas', nargs='+', type=int, default=[1, 4])
    p.add_argument('--compile', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    p.add_argument('--compile-mode', choices=['reduce-overhead', 'max-autotune'], default='reduce-overhead')
    p.add_argument('--neighbors', nargs='+', choices=['brute', 'cell', 'verlet'], default=['brute', 'cell', 'verlet'])
    p.add_argument('--cuda-graphs', action='store_true')
    p.add_argument('--skip-openmm', action='store_true')
    p.add_argument('--out', default='speed.jsonl')
    p.add_argument('--trace', help='directory for one fast-step profiler trace')
    for flag, default in [('steps', 10), ('warmup', 2), ('openmm-steps', 1000),
                          ('openmm-warmup', 100), ('channels', 64), ('lmax', 2),
                          ('heads', 8), ('n-rbf', 32), ('k', 3), ('n-spatial', 2),
                          ('n-fusion', 2), ('n-denoiser', 2), ('seed', 2026)]:
        p.add_argument('--' + flag, type=int, default=default)
    p.add_argument('--cutoff', type=float, default=6)
    p.add_argument('--skin', type=float, default=1)
    p.add_argument('--stride-fs', type=float, default=50)
    p.add_argument('--dt-fs', type=float, default=2)
    a = p.parse_args(argv)
    if (min(a.steps, a.openmm_steps, a.stride_fs, a.dt_fs, *a.sizes,
            *a.sampling_steps, *a.replicas) <= 0 or min(a.warmup, a.openmm_warmup) < 0):
        p.error('steps, sizes, replicas and strides must be positive; warmup nonnegative')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(a.out, 'w') as out:
        def record(row):
            rows.append(row)
            out.write(json.dumps(row, allow_nan=False) + '\n')
            out.flush()
        for n in a.sizes:
            baseline = None
            if not a.micro and not a.skip_openmm:
                try:
                    baseline = openmm_bench(a, *synthetic(n, a.system))
                    record(baseline)
                except Exception as exc:
                    record(dict(engine='openmm', atoms=n, error=str(exc)))
            configs = itertools.product(a.precisions, a.sampling_steps, a.replicas, a.compile, a.neighbors)
            if a.micro: configs = [(None, None, 1, 0, name) for name in ['brute', 'cell', 'verlet']]
            for precision, sampling, replicas, compile_on, neighbor in configs:
                row = dict(engine='neighbor' if a.micro else 'wignerflow', atoms=n,
                           precision=precision, sampling_steps=sampling, replicas=replicas,
                           compile=bool(compile_on), neighbor=neighbor, system=a.system)
                try:
                    if a.micro:
                        pos, box, _, _ = synthetic(n, a.system, a.device)
                        fn = {'brute': brute_force, 'cell': cell_list,
                              'verlet': VerletList(a.cutoff, a.skin)}[neighbor]
                        # Include one initial rebuild and subsequent displacement checks.
                        wall = timed(lambda: fn(pos, box, a.cutoff) if neighbor != 'verlet'
                                     else fn(pos, box), a.steps, 0, a.device)
                        row.update(ms_per_call=wall / a.steps * 1000,
                                   rebuilds=fn.rebuilds if neighbor == 'verlet' else None)
                    else:
                        row.update(model_bench(a, n, precision, sampling, replicas, compile_on, neighbor))
                        row['speedup_vs_openmm'] = (row['ns_per_day'] / baseline['ns_per_day'] if baseline else None)
                        row['aggregate_speedup_vs_openmm'] = (row['ns_per_day_aggregate'] / baseline['ns_per_day'] if baseline else None)
                except Exception as exc:
                    row['error'] = f'{type(exc).__name__}: {exc}'
                record(row)
                gc.collect()
                if torch.device(a.device).type == 'cuda': torch.cuda.empty_cache()
    print('engine atoms precision sampling replicas neighbor steps/s ns/day peak_bytes speedup ms/call error')
    for r in rows:
        print(' '.join(str(r.get(k, '-')) for k in ['engine', 'atoms', 'precision', 'sampling_steps',
              'replicas', 'neighbor', 'steps_per_s', 'ns_per_day', 'peak_memory_bytes',
              'speedup_vs_openmm', 'ms_per_call', 'error']))
    return rows


if __name__ == '__main__':
    main()
