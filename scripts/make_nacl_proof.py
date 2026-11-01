"""Small NaCl LJ+Coulomb proof dataset (not a fitted molten-salt force field).

Default: 64 Na + 64 Cl at 1200 K, CPU PME, independent seeded NVT/NVE
trajectories. The short run exercises multi-species data flow; establishing a
liquid phase or converged transport coefficients needs longer production.
"""
import argparse
from pathlib import Path

import numpy as np

from equitraj.adapters import write_data_config
from equitraj.trajfile import TrajWriter


def build_nacl(shape=(4, 4, 8), spacing_nm=0.32):
    import openmm
    import openmm.app as app
    import openmm.unit as u

    if len(shape) != 3 or any(n < 2 or n % 2 for n in shape) or spacing_nm <= 0:
        raise ValueError("shape must contain three positive even dimensions >=2")
    grid = np.indices(shape).reshape(3, -1).T
    labels = np.where(grid.sum(axis=1) % 2 == 0, "Na", "Cl")
    positions = grid * spacing_nm
    lengths = np.array(shape) * spacing_nm
    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(*[openmm.Vec3(*row) for row in np.diag(lengths)])
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.PME)
    cutoff = min(0.9, 0.45 * lengths.min())
    nb.setCutoffDistance(cutoff * u.nanometer)
    nb.setEwaldErrorTolerance(1e-4)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(0.85 * cutoff * u.nanometer)
    nb.setUseDispersionCorrection(True)
    topology = app.Topology()
    chain = topology.addChain()
    params = {"Na": (22.989769, 1, 0.24, 0.2), "Cl": (35.45, -1, 0.40, 0.5)}
    for label in labels:
        mass, charge, sigma, epsilon = params[label]
        system.addParticle(mass)
        nb.addParticle(charge, sigma, epsilon)
        residue = topology.addResidue(label.upper(), chain)
        topology.addAtom(label, app.Element.getBySymbol(label), residue)
    system.addForce(nb)
    topology.setPeriodicBoxVectors(np.diag(lengths) * u.nanometer)
    return system, topology, positions


def generate(args):
    import openmm
    import openmm.app as app
    import openmm.unit as u
    from equitraj.simulate import velocity_verlet

    if (args.n_traj < 1 or args.dt_fs <= 0 or args.save_every < 1 or
            args.frames < 2 or args.equil_steps < 0 or args.temperature <= 0 or args.threads < 1):
        raise ValueError("invalid simulation parameters")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Check all destinations before starting an expensive simulation.
    for i in range(args.n_traj):
        path = out / f"traj_{i:03d}.h5"
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
    system, topology, positions = build_nacl(tuple(args.shape), args.spacing_nm)
    (out / "system.xml").write_text(openmm.XmlSerializer.serialize(system))
    with (out / "system.pdb").open("w") as f:
        app.PDBFile.writeFile(topology, positions * u.nanometer, f)
    masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
    elements = [a.element.symbol for a in topology.atoms()]
    platform = openmm.Platform.getPlatformByName(args.platform)
    props = {"Threads": str(args.threads)} if args.platform == "CPU" else {}
    for i in range(args.n_traj):
        seed = args.seed + i
        lang = openmm.LangevinMiddleIntegrator(args.temperature * u.kelvin, 1 / u.picosecond,
                                               args.dt_fs * u.femtosecond)
        lang.setRandomNumberSeed(seed)
        context = openmm.Context(system, lang, platform, props)
        context.setPositions(positions * u.nanometer)
        openmm.LocalEnergyMinimizer.minimize(context, maxIterations=200)
        context.setVelocitiesToTemperature(args.temperature, seed)
        if args.equil_steps:
            lang.step(args.equil_steps)
        state = context.getState(getPositions=True, getVelocities=True)
        x = state.getPositions(asNumpy=True)
        v = state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond)
        v -= np.sum(masses[:, None] * v, axis=0) / masses.sum()
        del context, lang
        vv = velocity_verlet(args.dt_fs / 1000)
        context = openmm.Context(system, vv, platform, props)
        context.setPositions(x)
        context.setVelocities(v * u.nanometer / u.picosecond)
        attrs = dict(system="nacl_lj_coulomb", periodic=True, complete=False,
                     force_field="illustrative LJ+Coulomb PME, not fitted Tosi-Fumi",
                     dt_fs=args.dt_fs, save_every=args.save_every,
                     frame_interval_fs=args.dt_fs * args.save_every, seed=seed,
                     temperature_K=args.temperature, ensemble="NVE", n_constraints=0,
                     integrator="velocity Verlet", platform=args.platform, positions_unwrapped=True)
        path = out / f"traj_{i:03d}.h5"
        for field in ("pos", "vel", "meta"):
            (out / "npy" / f"{path.stem}_{field}.npy").unlink(missing_ok=True)
        writer = TrajWriter(path, len(masses), masses, elements, attrs)
        try:
            for frame in range(args.frames):
                if frame:
                    vv.step(args.save_every)
                s = context.getState(getPositions=True, getVelocities=True, getForces=True, getEnergy=True,
                                     enforcePeriodicBox=False)
                writer.add(s.getPositions(asNumpy=True).value_in_unit(u.nanometer),
                           s.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond),
                           s.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer),
                           s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
                           s.getKineticEnergy().value_in_unit(u.kilojoule_per_mole),
                           s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer),
                           s.getTime().value_in_unit(u.picosecond))
            writer.flush()
            writer.f.attrs["complete"] = True
        finally:
            writer.close()
            del context, vv
        print(path, flush=True)
    write_data_config(out, args.n_traj)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/nacl_proof")
    ap.add_argument("--shape", type=int, nargs=3, default=[4, 4, 8])
    ap.add_argument("--spacing-nm", type=float, default=0.32)
    ap.add_argument("--temperature", type=float, default=1200)
    ap.add_argument("--n-traj", type=int, default=3)
    ap.add_argument("--frames", type=int, default=301)
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--equil-steps", type=int, default=2000)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--platform", default="CPU")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    generate(ap.parse_args())


if __name__ == "__main__":
    main()
