"""Compare OpenMM integration speed with batch-one autoregressive model speed.

Examples:
  python scripts/bench_openmm.py --system argon --model-random --channels 16 --heads 4
  python scripts/bench_openmm.py --xml data/nacl_proof/system.xml --pdb data/nacl_proof/system.pdb --model-random
Checkpoint species must match the benchmark system. Random models measure compute
cost only. Warmup and context construction are excluded from reported timing.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np


def rates(steps, wall, stride_fs):
    return dict(steps_per_s=steps / wall, ns_per_day=steps / wall * stride_fs * 86400 / 1e6,
                steps_per_ns=1e6 / stride_fs, wall_s=wall, step_fs=stride_fs)


def benchmark(args, built=None):
    """Optionally pass built=(OpenMM System, Topology, positions_nm)."""
    import openmm
    import openmm.app as app
    import openmm.unit as u
    import torch
    from equitraj.model import EquiTraj
    from equitraj.simulate import velocity_verlet

    if (min(args.steps, args.model_steps, args.dt_fs, args.stride_fs, args.sampling_steps, args.threads) <= 0
            or min(args.warmup, args.model_warmup) < 0):
        raise ValueError("steps and timesteps must be positive; warmup nonnegative")
    if built is not None:
        system, topology, positions = built
    elif args.xml:
        if not args.pdb:
            raise ValueError("--xml requires --pdb")
        system = openmm.XmlSerializer.deserialize(Path(args.xml).read_text())
        pdb = app.PDBFile(args.pdb)
        topology = pdb.topology
        positions = pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    else:
        from equitraj.systems import build
        system, topology, positions, _ = build(args.system)
    elements = [a.element.symbol if a.element is not None else "X" for a in topology.atoms()]
    masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
    if len(elements) != len(masses) or np.any(masses <= 0):
        raise ValueError("benchmark requires one topology atom per positive-mass particle")
    device = torch.device(args.device)
    species = sorted(set(elements))
    stride_fs = args.stride_fs
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        species = list(ck["elements"])
        if not set(elements).issubset(species):
            raise ValueError("checkpoint has no embedding for benchmark species")
        if "cfg" in ck:
            cfg = ck["cfg"]
            mc = dict(cfg["model"])
            k = cfg["k"]
            stride_fs = ck.get("stride_fs")
            if stride_fs is None:
                # Training checkpoints store stride in integration steps.
                import yaml
                data_cfg = Path(cfg["data_dir"]) / "config.yaml"
                if not data_cfg.exists():
                    raise ValueError("training checkpoint needs its data config to determine stride_fs")
                with data_cfg.open() as f:
                    dc = yaml.safe_load(f)
                if "dt_fs" in dc:
                    stride_fs = cfg["stride_steps"] * dc["dt_fs"]
                else:
                    import h5py
                    with h5py.File(Path(cfg["data_dir"]) / "traj_000.h5", "r") as f:
                        stride_fs = cfg["stride_steps"] * float(f.attrs["dt_fs"])
        else:
            mc, k, stride_fs = dict(ck["model_cfg"]), ck["k"], ck["stride_fs"]
        mc.update(n_types=len(species), k=k, stride_fs=stride_fs)
        model = EquiTraj(**mc)
        model.load_state_dict(ck["model"])
    else:
        torch.manual_seed(args.seed)
        model = EquiTraj(n_types=len(species), lmax=args.lmax, channels=args.channels,
                         heads=args.heads, n_rbf=args.n_rbf, cutoff=args.cutoff, k=args.k,
                         n_spatial=args.n_spatial, n_fusion=args.n_fusion, n_denoiser=args.n_denoiser,
                         head=args.head, stride_fs=stride_fs, neighbor_method=args.neighbor_method)
        k = args.k
        # Random weights still need physical scales to keep the timing context
        # finite. These nominal scales are not a calibration or accuracy claim.
        model.dx_std.fill_(0.1)
        model.v_std.fill_(0.001)
        model.out_dx_std.fill_(0.001)
        model.out_v_std.fill_(0.00001)
    model = model.to(device).eval()
    integ = velocity_verlet(args.dt_fs / 1000)
    platform = openmm.Platform.getPlatformByName(args.platform)
    props = {"Threads": str(args.threads)} if args.platform == "CPU" else {}
    if args.platform in ("CUDA", "OpenCL"):
        props["Precision"] = args.precision
    context = openmm.Context(system, integ, platform, props)
    context.setPositions(np.asarray(positions) * u.nanometer)
    openmm.LocalEnergyMinimizer.minimize(context, maxIterations=100)
    context.setVelocitiesToTemperature(args.temperature, args.seed)
    if args.warmup:
        integ.step(args.warmup)
    context.getState(getEnergy=True)
    start = time.perf_counter()
    integ.step(args.steps)
    context.getState(getEnergy=True)  # synchronize OpenMM GPU queue
    omm_wall = time.perf_counter() - start
    state = context.getState(getPositions=True, getVelocities=True)
    x = state.getPositions(asNumpy=True).value_in_unit(u.nanometer) * 10
    v = state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond) * 0.01
    cell = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer) * 10
    periodic = system.usesPeriodicBoundaryConditions()
    del context, integ
    # Coasting context avoids including MD context generation in model timing.
    history = x[None] + (np.arange(k) - k + 1)[:, None, None] * stride_fs * v[None]
    xctx = torch.tensor(history[None], dtype=torch.float64, device=device)
    vctx = torch.tensor(np.broadcast_to(v, history.shape).copy()[None], dtype=torch.float32, device=device)
    types = torch.tensor([species.index(e) for e in elements], dtype=torch.long, device=device)
    mass = torch.tensor(masses, dtype=torch.float32, device=device)
    box = torch.tensor(cell[None], dtype=torch.float32, device=device) if periodic else None

    def step():
        nonlocal xctx, vctx
        dx, vn = model.predict(xctx.float(), vctx, types, box, mass, n_steps=args.sampling_steps)
        xn = xctx[:, -1] + dx.double()
        xctx = torch.cat((xctx[:, 1:], xn[:, None]), dim=1)
        vctx = torch.cat((vctx[:, 1:], vn[:, None]), dim=1)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()

    with torch.no_grad():
        for _ in range(args.model_warmup):
            step()
        sync()
        start = time.perf_counter()
        for _ in range(args.model_steps):
            step()
        sync()
        model_wall = time.perf_counter() - start
    finite = bool(torch.isfinite(xctx).all() and torch.isfinite(vctx).all())
    if not finite:
        raise RuntimeError("model became nonfinite; timing is not a valid trajectory benchmark")
    return dict(atoms=len(masses), species=species, batch=1,
                openmm=dict(platform=args.platform, **rates(args.steps, omm_wall, args.dt_fs)),
                model=dict(device=str(device), random_init=not bool(args.checkpoint),
                           sampling_steps=args.sampling_steps, parameters=sum(p.numel() for p in model.parameters()),
                           **rates(args.model_steps, model_wall, stride_fs)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--xml")
    src.add_argument("--system", default="argon", help="equitraj.systems builder or saved system name")
    ap.add_argument("--pdb")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--checkpoint")
    mode.add_argument("--model-random", action="store_true")
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--precision", default="mixed")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--model-steps", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--model-warmup", type=int, default=2)
    ap.add_argument("--dt-fs", type=float, default=1)
    ap.add_argument("--stride-fs", type=float, default=10)
    ap.add_argument("--temperature", type=float, default=1200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sampling-steps", type=int, default=10)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lmax", type=int, default=1)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-rbf", type=int, default=16)
    ap.add_argument("--cutoff", type=float, default=5)
    ap.add_argument("--n-spatial", type=int, default=1)
    ap.add_argument("--n-fusion", type=int, default=1)
    ap.add_argument("--n-denoiser", type=int, default=1)
    ap.add_argument("--head", choices=["regression", "flow"], default="regression")
    ap.add_argument("--neighbor-method", choices=["brute_force", "cell_list"], default="brute_force")
    args = ap.parse_args()
    print(json.dumps(benchmark(args), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
