"""Build OpenMM systems.

Each builder returns (system, topology, positions_nm, info) where info is a
dict of metadata that gets stored in the HDF5 file.
"""

import json
import os

import numpy as np
import openmm
import openmm.app as app
import openmm.unit as u

AVOGADRO = 6.02214076e23


def fcc_lattice(n_cells, a):
    """Positions of a cubic fcc lattice with n_cells^3 unit cells and lattice constant a.

    4 atoms per unit cell, so n_cells=4 gives 256 atoms.
    """
    basis = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]])
    cells = np.array([[i, j, k] for i in range(n_cells) for j in range(n_cells) for k in range(n_cells)])
    pos = (cells[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    return pos * a


def build_argon(n_cells=4, density_g_cm3=1.374, cutoff_nm=0.85, switch_nm=0.80):
    """Lennard-Jones argon (Rahman 1964 parameters) in a cubic periodic box.

    epsilon/k_B = 119.8 K -> 0.996 kJ/mol, sigma = 3.405 Å.
    The potential is smoothly switched to zero between switch_nm and cutoff_nm,
    so that NVE energy is conserved (a hard truncation would cause drift).
    """
    mass = 39.948  # amu
    sigma = 0.3405  # nm
    epsilon = 0.996  # kJ/mol

    n_atoms = 4 * n_cells**3
    # Box volume from density: V = N m / (N_A rho)
    volume_cm3 = n_atoms * mass / AVOGADRO / density_g_cm3
    box_nm = (volume_cm3 ** (1 / 3)) * 1e7  # cm -> nm
    positions = fcc_lattice(n_cells, box_nm / n_cells)

    system = openmm.System()
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0), openmm.Vec3(0, box_nm, 0), openmm.Vec3(0, 0, box_nm)
    )
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(cutoff_nm)
    nb.setUseSwitchingFunction(True)
    nb.setSwitchingDistance(switch_nm)
    nb.setUseDispersionCorrection(False)
    for _ in range(n_atoms):
        system.addParticle(mass)
        nb.addParticle(0.0, sigma, epsilon)
    system.addForce(nb)

    topology = app.Topology()
    chain = topology.addChain()
    argon = app.Element.getBySymbol("Ar")
    for _ in range(n_atoms):
        residue = topology.addResidue("AR", chain)
        topology.addAtom("Ar", argon, residue)
    topology.setPeriodicBoxVectors(np.eye(3) * box_nm * u.nanometer)

    info = {
        "system_name": "argon",
        "force_field": f"LJ sigma={sigma} nm epsilon={epsilon} kJ/mol, switch {switch_nm}-{cutoff_nm} nm",
        "box_nm": box_nm,
        "density_g_cm3": density_g_cm3,
    }
    return system, topology, positions, info


def build_water_box(box_nm=2.25):
    """TIP3P water box with PME and HBonds constraints (~1100 atoms).

    Only used as a speed benchmark: it has about the same size and force types
    (PME, bonds, constraints) as the Li+/EC target system.
    """
    modeller = app.Modeller(app.Topology(), [])
    ff = app.ForceField("amber14/tip3p.xml")
    modeller.addSolvent(ff, boxSize=openmm.Vec3(box_nm, box_nm, box_nm) * u.nanometer)
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=0.9 * u.nanometer,
                             constraints=app.HBonds, rigidWater=False)
    positions = modeller.positions.value_in_unit(u.nanometer)
    info = {"system_name": "water_box", "force_field": "amber14 TIP3P, flexible, HBonds", "box_nm": box_nm}
    return system, modeller.topology, np.array(positions), info


SYSTEMS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "systems")


def load_saved(name):
    """Load a system written by build_systems.py (systems/<name>.xml, .pdb, .json).

    The CMMotionRemover is dropped: it edits velocities every step, which breaks NVE.
    The center-of-mass velocity is removed once before production instead.
    """
    with open(os.path.join(SYSTEMS_DIR, f"{name}.xml")) as f:
        system = openmm.XmlSerializer.deserialize(f.read())
    for i in reversed(range(system.getNumForces())):
        if isinstance(system.getForce(i), openmm.CMMotionRemover):
            system.removeForce(i)
    pdb = app.PDBFile(os.path.join(SYSTEMS_DIR, f"{name}.pdb"))
    positions = pdb.getPositions(asNumpy=True).value_in_unit(u.nanometer)
    with open(os.path.join(SYSTEMS_DIR, f"{name}.json")) as f:
        info = json.load(f)
    info = {k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in info.items()}  # HDF5 attrs need flat values
    return system, pdb.topology, np.asarray(positions), info


BUILDERS = {"argon": build_argon, "water_box": build_water_box}


def build(name):
    return BUILDERS[name]() if name in BUILDERS else load_saved(name)
