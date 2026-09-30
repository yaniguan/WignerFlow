"""Ground-truth MD with OpenMM: minimize -> NVT (Langevin) -> NVE (velocity Verlet) -> HDF5.

Why velocity Verlet and not openmm.VerletIntegrator: OpenMM's VerletIntegrator is a
leapfrog scheme, so the velocities it reports are half a step behind the positions.
The model needs positions and velocities at the same time, so we use a custom
velocity Verlet integrator (with RATTLE-style constraint handling).
"""

import time

import numpy as np
import openmm
import openmm.unit as u

from equitraj.systems import build
from equitraj.trajfile import TrajWriter


def velocity_verlet(dt_ps):
    """Velocity Verlet with position and velocity constraints (same recipe as openmmtools)."""
    integ = openmm.CustomIntegrator(dt_ps)
    integ.addPerDofVariable("x1", 0)
    integ.addUpdateContextState()
    integ.addComputePerDof("v", "v + 0.5*dt*f/m")
    integ.addComputePerDof("x", "x + dt*v")
    integ.addComputePerDof("x1", "x")
    integ.addConstrainPositions()
    # f is recomputed at the new positions here; (x - x1)/dt adds the velocity change from the constraints
    integ.addComputePerDof("v", "v + 0.5*dt*f/m + (x - x1)/dt")
    integ.addConstrainVelocities()
    return integ


def make_context(system, integrator, platform_name, precision):
    platform = openmm.Platform.getPlatformByName(platform_name)
    props = {}
    if platform_name in ("CUDA", "OpenCL"):
        props["Precision"] = precision
    try:
        context = openmm.Context(system, integrator, platform, props)
    except Exception:
        # Apple GPUs have no double precision, so OpenCL 'mixed' can fail. Fall back to single.
        props["Precision"] = "single"
        context = openmm.Context(system, integrator, platform, props)
    used = props.get("Precision", "platform default")
    return context, used


def remove_com_velocity(velocities, masses):
    p = (velocities * masses[:, None]).sum(axis=0)
    return velocities - p / masses.sum()


def equilibrate_npt(system, positions, cfg, seed):
    """NPT with a Monte Carlo barostat. Returns the state whose box volume is closest to the
    average volume over the second half, so that the NVT/NVE runs use an equilibrium density."""
    dt_ps = cfg["dt_fs"] / 1000.0
    npt_system = openmm.XmlSerializer.clone(system)
    npt_system.addForce(openmm.MonteCarloBarostat(cfg.get("pressure_bar", 1.0), cfg["temperature_K"], 25))
    integ = openmm.LangevinMiddleIntegrator(cfg["temperature_K"], cfg["friction_per_ps"], dt_ps)
    integ.setRandomNumberSeed(seed)
    integ.setConstraintTolerance(cfg["constraint_tol"])
    context, _ = make_context(npt_system, integ, cfg["platform"], cfg["precision"])
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(cfg["temperature_K"], seed)
    n_steps = int(round(cfg["npt_ps"] / dt_ps))
    integ.step(n_steps // 2)
    states = []
    for _ in range(100):
        integ.step(max(1, n_steps // 200))
        states.append(context.getState(getPositions=True, getVelocities=True))
    volumes = np.array([s.getPeriodicBoxVolume().value_in_unit(u.nanometer**3) for s in states])
    best = states[int(np.argmin(np.abs(volumes - volumes.mean())))]
    return (best.getPositions(asNumpy=True), best.getPeriodicBoxVectors(), best.getVelocities(asNumpy=True))


def run_trajectory(cfg, seed, out_path):
    """Run one independent trajectory and write it to out_path. Returns timing info."""
    system, topology, positions, info = build(cfg["system"])
    masses = np.array([system.getParticleMass(i).value_in_unit(u.dalton) for i in range(system.getNumParticles())])
    elements = [a.element.symbol for a in topology.atoms()]
    temperature = cfg["temperature_K"]
    dt_ps = cfg["dt_fs"] / 1000.0

    box = None
    velocities = None
    if cfg.get("npt_ps", 0) > 0:
        positions, box, velocities = equilibrate_npt(system, positions, cfg, seed)

    # --- NVT equilibration (Langevin) ---
    lang = openmm.LangevinMiddleIntegrator(temperature, cfg["friction_per_ps"], dt_ps)
    lang.setRandomNumberSeed(seed)
    lang.setConstraintTolerance(cfg["constraint_tol"])
    context, precision_used = make_context(system, lang, cfg["platform"], cfg["precision"])
    if box is not None:
        context.setPeriodicBoxVectors(*box)
    context.setPositions(positions)
    if velocities is None:
        openmm.LocalEnergyMinimizer.minimize(context)
        context.setVelocitiesToTemperature(temperature, seed)
    else:
        context.setVelocities(velocities)
    n_equil = int(round(cfg["equil_ps"] / dt_ps))
    # First half: relax. Second half: also record the total energy, to get its NVT average.
    lang.step(n_equil // 2)
    energies = []
    for _ in range(100):
        lang.step(max(1, n_equil // 200))
        s = context.getState(getEnergy=True)
        energies.append((s.getPotentialEnergy() + s.getKineticEnergy()).value_in_unit(u.kilojoule_per_mole))
    state = context.getState(getPositions=True, getVelocities=True, getEnergy=True)
    x0 = state.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    v0 = remove_com_velocity(state.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond), masses)
    box0 = state.getPeriodicBoxVectors()
    # NVE conserves the energy of the starting snapshot. A random snapshot can sit a few kJ/mol
    # away from the NVT average, which shifts the NVE temperature by several K in a small box.
    # Rescale velocities so that the starting total energy equals the NVT average.
    e_target = float(np.mean(energies))
    pe = state.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole)
    ke = 0.5 * np.sum(masses[:, None] * v0**2)
    v0 = v0 * np.sqrt(max(e_target - pe, 0.1 * ke) / ke)
    del context

    # --- NVE production (velocity Verlet) ---
    vv = velocity_verlet(dt_ps)
    vv.setConstraintTolerance(cfg["constraint_tol"])
    context, precision_used = make_context(system, vv, cfg["platform"], cfg["precision"])
    context.setPeriodicBoxVectors(*box0)
    context.setPositions(x0)
    context.setVelocities(v0)
    context.applyVelocityConstraints(cfg["constraint_tol"])

    save_every = cfg["save_every"]
    n_frames = int(round(cfg["production_ps"] / dt_ps / save_every))
    attrs = {
        **info,
        "temperature_K": temperature,
        "dt_fs": cfg["dt_fs"],
        "save_every": save_every,
        "frame_interval_fs": cfg["dt_fs"] * save_every,
        "equil_ps": cfg["equil_ps"],
        "npt_ps": cfg.get("npt_ps", 0),
        "nvt_mean_total_energy_kJmol": e_target,
        "seed": seed,
        "ensemble": "NVE",
        "integrator": "velocity Verlet (CustomIntegrator)",
        "platform": cfg["platform"],
        "precision": precision_used,
        "constraint_tol": cfg["constraint_tol"],
        "n_constraints": system.getNumConstraints(),
        "openmm_version": openmm.__version__,
        "complete": False,
    }
    writer = TrajWriter(out_path, len(masses), masses, elements, attrs, force_every=cfg["force_every"])

    t_start = time.time()
    for i in range(n_frames):
        if i > 0:
            vv.step(save_every)
        get_forces = i % cfg["force_every"] == 0
        s = context.getState(getPositions=True, getVelocities=True, getForces=get_forces, getEnergy=True)
        writer.add(
            positions=s.getPositions(asNumpy=True).value_in_unit(u.nanometer),
            velocities=s.getVelocities(asNumpy=True).value_in_unit(u.nanometer / u.picosecond),
            forces=s.getForces(asNumpy=True).value_in_unit(u.kilojoule_per_mole / u.nanometer) if get_forces else None,
            potential_energy=s.getPotentialEnergy().value_in_unit(u.kilojoule_per_mole),
            kinetic_energy=s.getKineticEnergy().value_in_unit(u.kilojoule_per_mole),
            box=s.getPeriodicBoxVectors(asNumpy=True).value_in_unit(u.nanometer),
            time=s.getTime().value_in_unit(u.picosecond),
        )
    wall = time.time() - t_start
    sim_ns = (n_frames - 1) * save_every * dt_ps / 1000.0
    writer.f.attrs["complete"] = True
    writer.f.attrs["production_wall_s"] = wall
    writer.f.attrs["ns_per_day_with_io"] = sim_ns / wall * 86400
    writer.close()
    return {"wall_s": wall, "ns_per_day_with_io": sim_ns / wall * 86400, "precision": precision_used}


def benchmark(cfg, n_steps=2000):
    """OpenMM speed without saving frames (pure integration), in ns/day."""
    system, _, positions, _ = build(cfg["system"])
    dt_ps = cfg["dt_fs"] / 1000.0
    vv = velocity_verlet(dt_ps)
    context, precision_used = make_context(system, vv, cfg["platform"], cfg["precision"])
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context)
    context.setVelocitiesToTemperature(cfg["temperature_K"], 0)
    vv.step(100)  # warm up
    context.getState(getEnergy=True)
    t = time.time()
    vv.step(n_steps)
    context.getState(getEnergy=True)  # forces the queue to finish on GPU platforms
    wall = time.time() - t
    return {"platform": cfg["platform"], "precision": precision_used, "ns_per_day": n_steps * dt_ps / 1000.0 / wall * 86400}
