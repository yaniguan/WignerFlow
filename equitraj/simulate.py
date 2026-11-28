"""Ground-truth MD with OpenMM: minimize -> NVT (Langevin) -> NVE (velocity Verlet) -> HDF5.

Why velocity Verlet and not openmm.VerletIntegrator: OpenMM's VerletIntegrator is a
leapfrog scheme, so the velocities it reports are half a step behind the positions.
The model needs positions and velocities at the same time, so we use a custom
velocity Verlet integrator (with RATTLE-style constraint handling).
"""

import math
import time

import numpy as np

try:
    import openmm
    import openmm.unit as u
    _HAS_OPENMM = True
except ImportError:  # pragma: no cover - fallback analytical LJ integrator used for argon
    openmm = None
    u = None
    _HAS_OPENMM = False

from equitraj.systems import AVOGADRO, build, fcc_lattice
from equitraj.trajfile import TrajWriter
from equitraj.units import KB


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


def _lj_argon_energy_forces_nm(pos_nm, box_nm, sigma=0.3405, epsilon=0.996, r_switch=0.80, r_cut=0.85):
    """Exact analytical switched Lennard-Jones potential (kJ/mol) and forces (kJ/mol/nm) in float64.

    Matches `equitraj.systems.build_argon` and `equitraj.energy.lj_potential` identically:
        V(r) = 4 * eps * [(sigma / r)^12 - (sigma / r)^6] * S(u),
        u = clamp((r - r_switch) / (r_cut - r_switch), 0, 1),
        S(u) = 1 - 10 u^3 + 15 u^4 - 6 u^5.
    """
    n = pos_nm.shape[0]
    dr = pos_nm[:, None, :] - pos_nm[None, :, :]
    dr = dr - box_nm * np.round(dr / box_nm)
    r2 = np.sum(dr * dr, axis=-1)
    np.fill_diagonal(r2, 1e12)
    mask = r2 < (r_cut * r_cut)
    if not np.any(mask):
        return 0.0, np.zeros_like(pos_nm)
    r = np.sqrt(r2)
    inv_r = np.where(mask, 1.0 / r, 0.0)
    sr2 = (sigma * inv_r) ** 2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    v_lj = 4.0 * epsilon * (sr12 - sr6)
    dv_lj_dr = 4.0 * epsilon * (-12.0 * sr12 + 6.0 * sr6) * inv_r

    w = r_cut - r_switch
    u_sw = np.clip((r - r_switch) / w, 0.0, 1.0)
    s_val = 1.0 - 10.0 * (u_sw ** 3) + 15.0 * (u_sw ** 4) - 6.0 * (u_sw ** 5)
    in_sw = (r > r_switch) & mask
    ds_dr = np.where(in_sw, (-30.0 * (u_sw ** 2) * ((1.0 - u_sw) ** 2)) / w, 0.0)

    v_pair = np.where(mask, v_lj * s_val, 0.0)
    dv_dr = np.where(mask, dv_lj_dr * s_val + v_lj * ds_dr, 0.0)
    # Force on atom i from j: -dV/dr * (r_i - r_j) / r
    f_mag_over_r = -dv_dr * inv_r
    forces = np.sum(f_mag_over_r[:, :, None] * dr, axis=1)
    pe = 0.5 * float(np.sum(v_pair))
    return pe, forces


def _run_argon_pure(cfg, seed, out_path):
    """Deterministic float64 NVT -> NVE switched-LJ liquid argon trajectory generator without OpenMM."""
    n_cells = 4
    mass = 39.948  # amu (g/mol); in (nm, ps, kJ/mol) units, 1 amu * (nm/ps)^2 == 1 kJ/mol!
    sigma, epsilon, switch_nm, cutoff_nm = 0.3405, 0.996, 0.80, 0.85
    density_g_cm3 = 1.374
    n_atoms = 4 * (n_cells ** 3)
    volume_cm3 = n_atoms * mass / AVOGADRO / density_g_cm3
    box_nm = float((volume_cm3 ** (1.0 / 3.0)) * 1e7)
    box_mat = np.eye(3, dtype=np.float64) * box_nm
    masses = np.full(n_atoms, mass, dtype=np.float64)
    elements = ["Ar"] * n_atoms

    rng = np.random.default_rng(int(seed))
    x = fcc_lattice(n_cells, box_nm / n_cells).astype(np.float64)
    x = x + rng.normal(0.0, 1e-3, size=x.shape)

    # Short clamped steepest-descent relaxation from perturbed FCC lattice
    for _ in range(40):
        _, f_min = _lj_argon_energy_forces_nm(x, box_nm, sigma, epsilon, switch_nm, cutoff_nm)
        f_norm = np.linalg.norm(f_min, axis=-1, keepdims=True)
        step_nm = np.where(f_norm > 1e-12, f_min / np.maximum(f_norm, 1.0) * 1e-4, 0.0)
        x = x + np.clip(step_nm, -2e-3, 2e-3)

    temp_k = float(cfg["temperature_K"])
    dt_ps = float(cfg["dt_fs"]) / 1000.0
    gamma = float(cfg.get("friction_per_ps", 1.0))
    v_std = math.sqrt(KB * temp_k / mass)  # nm/ps (since kJ/mol / (g/mol) = (nm/ps)^2)
    v = remove_com_velocity(rng.normal(0.0, v_std, size=(n_atoms, 3)), masses)

    # BAOAB Langevin NVT equilibration
    n_equil = int(round(float(cfg["equil_ps"]) / dt_ps))
    c1 = math.exp(-gamma * dt_ps)
    c2 = math.sqrt(max(0.0, 1.0 - c1 * c1)) * v_std
    _, forces = _lj_argon_energy_forces_nm(x, box_nm, sigma, epsilon, switch_nm, cutoff_nm)

    def baoab_steps(n_st):
        nonlocal x, v, forces
        half_dt = 0.5 * dt_ps
        inv_m = 1.0 / mass
        for _ in range(n_st):
            v = v + half_dt * forces * inv_m
            x = x + half_dt * v
            v = c1 * v + c2 * rng.standard_normal(size=v.shape)
            x = x + half_dt * v
            _, forces = _lj_argon_energy_forces_nm(x, box_nm, sigma, epsilon, switch_nm, cutoff_nm)
            v = v + half_dt * forces * inv_m

    baoab_steps(n_equil // 2)
    energies = []
    chunk = max(1, n_equil // 200)
    for _ in range(100):
        baoab_steps(chunk)
        pe_s, _ = _lj_argon_energy_forces_nm(x, box_nm, sigma, epsilon, switch_nm, cutoff_nm)
        v_clean = remove_com_velocity(v, masses)
        ke_s = 0.5 * float(np.sum(masses[:, None] * (v_clean ** 2)))
        energies.append(pe_s + ke_s)

    v0 = remove_com_velocity(v, masses)
    pe0, f0 = _lj_argon_energy_forces_nm(x, box_nm, sigma, epsilon, switch_nm, cutoff_nm)
    ke0 = 0.5 * float(np.sum(masses[:, None] * (v0 ** 2)))
    e_target = float(np.mean(energies)) if energies else (pe0 + ke0)
    v0 = v0 * math.sqrt(max(e_target - pe0, 0.1 * ke0) / max(ke0, 1e-30))

    # Float64 Velocity Verlet NVE production (unwrapped positions)
    save_every = int(cfg["save_every"])
    force_every = int(cfg["force_every"])
    n_frames = int(round(float(cfg["production_ps"]) / dt_ps / save_every))
    attrs = {
        "system_name": "argon",
        "force_field": f"LJ sigma={sigma} nm epsilon={epsilon} kJ/mol, switch {switch_nm}-{cutoff_nm} nm",
        "box_nm": box_nm,
        "density_g_cm3": density_g_cm3,
        "temperature_K": temp_k,
        "dt_fs": float(cfg["dt_fs"]),
        "save_every": save_every,
        "frame_interval_fs": float(cfg["dt_fs"]) * save_every,
        "equil_ps": float(cfg["equil_ps"]),
        "npt_ps": float(cfg.get("npt_ps", 0.0)),
        "nvt_mean_total_energy_kJmol": e_target,
        "seed": int(seed),
        "ensemble": "NVE",
        "integrator": "velocity Verlet (analytical switched-LJ, float64)",
        "platform": "CPU-numpy-float64",
        "precision": "double",
        "constraint_tol": float(cfg.get("constraint_tol", 1e-6)),
        "n_constraints": 0,
        "openmm_version": "none (analytical switched-LJ fallback)",
        "complete": False,
    }
    writer = TrajWriter(out_path, n_atoms, masses, elements, attrs, force_every=force_every)
    t_start = time.time()
    x_cur = x.copy()
    v_cur = v0.copy()
    pe_cur, f_cur = pe0, f0
    inv_m = 1.0 / mass
    half_dt = 0.5 * dt_ps
    for i in range(n_frames):
        if i > 0:
            for _ in range(save_every):
                v_cur = v_cur + half_dt * f_cur * inv_m
                x_cur = x_cur + dt_ps * v_cur
                pe_cur, f_cur = _lj_argon_energy_forces_nm(x_cur, box_nm, sigma, epsilon, switch_nm, cutoff_nm)
                v_cur = v_cur + half_dt * f_cur * inv_m
        ke_cur = 0.5 * float(np.sum(masses[:, None] * (v_cur ** 2)))
        get_forces = (i % force_every) == 0
        writer.add(
            positions=x_cur.astype(np.float32),
            velocities=v_cur.astype(np.float32),
            forces=f_cur.astype(np.float32) if get_forces else None,
            potential_energy=pe_cur,
            kinetic_energy=ke_cur,
            box=box_mat.astype(np.float32),
            time=i * save_every * dt_ps,
        )
    wall = max(time.time() - t_start, 1e-9)
    sim_ns = (n_frames - 1) * save_every * dt_ps / 1000.0
    writer.f.attrs["complete"] = True
    writer.f.attrs["production_wall_s"] = wall
    writer.f.attrs["ns_per_day_with_io"] = sim_ns / wall * 86400.0
    writer.close()
    return {"wall_s": wall, "ns_per_day_with_io": sim_ns / wall * 86400.0, "precision": "double"}


def run_trajectory(cfg, seed, out_path):
    """Run one independent trajectory and write it to out_path. Returns timing info."""
    if not _HAS_OPENMM or str(cfg.get("platform", "")).lower() == "numpy":
        if cfg.get("system") != "argon":
            raise RuntimeError("OpenMM is not installed; pure-NumPy fallback only supports system='argon'")
        return _run_argon_pure(cfg, seed, out_path)
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
