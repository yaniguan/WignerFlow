"""Build OpenFF-parametrized systems and save them as OpenMM XML + PDB + JSON in systems/.

Runs on the Mac only (needs openff-toolkit, AmberTools for AM1-BCC charges, packmol).
The clusters then only need OpenMM to load the saved files.

    python -m equitraj.build_systems            # builds every system below
    python -m equitraj.build_systems li_ec_q08  # builds one

Systems:
    li_ec_q08       1 Li+ in 100 EC, Li+ charge scaled to 0.8, HBonds constraints (default)
    li_ec_q10       same, full Li+ charge
    li_ec_pf6_q08   1 Li+ + 1 PF6- in 100 EC, ion charges scaled to 0.8
    li_ec_q08_flex  like li_ec_q08, without constraints (for 0.5 fs runs)
    ala2            alanine dipeptide in vacuum, HBonds constraints
    aspirin         aspirin in vacuum, HBonds constraints
"""

import json
import os
import sys

import numpy as np
import openmm
import openmm.app as app
from openff.interchange import Interchange
from openff.interchange.components._packmol import pack_box
from openff.toolkit import ForceField, Molecule
from openff.units import unit

FF_CONSTRAINED = "openff-2.2.1.offxml"
FF_FLEXIBLE = "openff_unconstrained-2.2.1.offxml"
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "systems")

SMILES = {
    "EC": "C1COC(=O)O1",
    "Li": "[Li+]",
    "PF6": "F[P-](F)(F)(F)(F)F",
    "ala2": "CC(=O)N[C@@H](C)C(=O)NC",  # Ace-Ala-Nme, L-alanine
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}


def charged_molecule(name):
    """Molecule with one conformer and AM1-BCC charges (library charges for Li+)."""
    mol = Molecule.from_smiles(SMILES[name])
    mol.generate_conformers(n_conformers=1)
    if mol.n_atoms > 1:
        mol.assign_partial_charges("am1bcc")
    else:
        mol.assign_partial_charges("formal_charge")
    return mol


def scale_charges(system, atom_indices, scale):
    """Multiply the charges of the given atoms by `scale` in the NonbondedForce."""
    nb = [f for f in system.getForces() if isinstance(f, openmm.NonbondedForce)][0]
    for i in atom_indices:
        q, sigma, eps = nb.getParticleParameters(i)
        nb.setParticleParameters(i, q * scale, sigma, eps)


def save(name, system, omm_topology, positions_nm, info):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{name}.xml"), "w") as f:
        f.write(openmm.XmlSerializer.serialize(system))
    with open(os.path.join(OUT_DIR, f"{name}.pdb"), "w") as f:
        app.PDBFile.writeFile(omm_topology, positions_nm * 10.0, f)  # PDB positions in Å
    with open(os.path.join(OUT_DIR, f"{name}.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"saved {name}: {system.getNumParticles()} atoms, {system.getNumConstraints()} constraints")


def build_electrolyte(name, n_ec=100, li_charge=0.8, pf6=False, flexible=False, density_g_cm3=1.20):
    """Li+ (and optionally PF6-) in EC, packed with packmol.

    Packed below the target density (1.32 g/cm^3 at 313 K); NPT equilibration fixes the density.
    Only the ions' charges are scaled; EC keeps its AM1-BCC charges.
    """
    ff = ForceField(FF_FLEXIBLE if flexible else FF_CONSTRAINED)
    mols = [charged_molecule("Li"), charged_molecule("EC")]
    counts = [1, n_ec]
    if pf6:
        mols.append(charged_molecule("PF6"))
        counts.append(1)

    mass_g_mol = sum(sum(a.mass.m for a in m.atoms) * c for m, c in zip(mols, counts))
    volume_nm3 = mass_g_mol / 6.02214076e23 / density_g_cm3 * 1e21
    box_nm = volume_nm3 ** (1 / 3)
    topology = pack_box(molecules=mols, number_of_copies=counts, box_vectors=np.eye(3) * box_nm * unit.nanometer)
    interchange = Interchange.from_smirnoff(ff, topology, charge_from_molecules=mols)
    system = interchange.to_openmm_system(combine_nonbonded_forces=True)

    # Atom order follows the packmol order: Li first, then EC molecules, then PF6.
    ion_atoms = [0]
    if pf6:
        n = system.getNumParticles()
        ion_atoms += list(range(n - 7, n))
    scale_charges(system, ion_atoms, li_charge)

    positions = interchange.positions.m_as(unit.nanometer)
    info = {
        "system_name": name,
        "force_field": (FF_FLEXIBLE if flexible else FF_CONSTRAINED) + f"; ion charges x{li_charge}",
        "composition": {"Li": 1, "EC": n_ec, "PF6": 1 if pf6 else 0},
        "ion_charge_scale": li_charge,
        "packed_box_nm": box_nm,
        "periodic": True,
        "li_index": 0,
    }
    save(name, system, interchange.to_openmm_topology(), positions, info)


def build_molecule_vacuum(name):
    """A single molecule in vacuum (no cutoff), HBonds constraints."""
    ff = ForceField(FF_CONSTRAINED)
    mol = charged_molecule(name)
    interchange = Interchange.from_smirnoff(ff, mol.to_topology(), charge_from_molecules=[mol])
    system = interchange.to_openmm_system(combine_nonbonded_forces=True)
    positions = interchange.positions.m_as(unit.nanometer)
    info = {"system_name": name, "force_field": FF_CONSTRAINED, "periodic": False, "smiles": SMILES[name]}
    save(name, system, interchange.to_openmm_topology(), positions, info)


BUILD = {
    "li_ec_q08": lambda: build_electrolyte("li_ec_q08", li_charge=0.8),
    "li_ec_q10": lambda: build_electrolyte("li_ec_q10", li_charge=1.0),
    "li_ec_pf6_q08": lambda: build_electrolyte("li_ec_pf6_q08", li_charge=0.8, pf6=True),
    "li_ec_q08_flex": lambda: build_electrolyte("li_ec_q08_flex", li_charge=0.8, flexible=True),
    "ala2": lambda: build_molecule_vacuum("ala2"),
    "aspirin": lambda: build_molecule_vacuum("aspirin"),
}

if __name__ == "__main__":
    for key in sys.argv[1:] or BUILD:
        BUILD[key]()
