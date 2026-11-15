# External MD tools

These commands assume an installed WignerFlow package. Install `ase` for
ASE-readable inputs, `mdtraj` for DCD/XTC, and `openmm` for simulation and timing.
Those readers are imported only when used.

## Import

```sh
python scripts/import_md.py run0.xtc run1.xtc run2.xtc --topology ions.pdb --out-dir data/imported --dt-fs 1 --save-every 10
python scripts/import_md.py trajectory.npz --unit angstrom --velocity-unit angstrom/fs --out-dir data/imported_npz --dt-fs 2
```

Each source produces `traj_NNN.h5`. The command writes `config.yaml` with explicit
train/val/test trajectory IDs. The Python APIs are
`equitraj.adapters.import_trajectory` and `write_data_config`; importing individual
files does not rewrite the data config. Supply `--split` as a JSON object to choose
splits. A single trajectory goes to train; two go to train/test; three or more
reserve the last two for val/test.

Generic NPZ/HDF5 files contain `positions` (T,N,3), optional `velocities` of the
same shape, optional `box` (3,), (3,3), or (T,3,3), and per-atom `elements` and/or
`masses`. `names` optionally identifies Drude particles. Position/box units must
be specified. Velocity defaults are Å/fs for angstrom inputs and nm/ps for nm
inputs. Separate velocity files (`--velocity-source`, one per trajectory) may be
NPY arrays or NPZ/HDF5 files with a `velocities` key. MDTraj reads coordinates in
nm. ASE momenta are converted from ASE internal time units; custom ASE arrays
named `velocities` are assumed to contain Å/fs.

The importer drops particles with name D* or mass below 1 amu by default;
`--keep-drude` preserves them, including zero-mass virtual particles. Mass is not
redistributed to parents. Provide explicit elements to override mass inference
or reader topology guesses. Partial periodicity is rejected. Boxes contain
lattice vectors as rows. Missing velocities are central differences of unwrapped
positions, with one-sided differences at the ends. They are marked
`velocities_estimated=True`. Crossings larger than half a cell between saved
frames cannot be recovered. Use `--already-unwrapped` for unwrapped sources.

Output follows TrajWriter: nm, nm/ps, ps, amu; missing potential energies and
forces are NaN. Kinetic energies are computed from velocities and masses.
`complete=True` is set only after frames have been flushed. Existing trajectory
files require `--overwrite`; replacing one invalidates its training cache.

## Metadata

To include PFF metadata, pass it to `import_trajectory(particle_metadata=...)`
or use a JSON file with `scripts/import_md.py --particle-metadata FILE`.
Metadata is off by default. The version 1 format in `equitraj/topology.py`
requires `version: 1`,
`units: {"charge": "e", "distance": "nm"}`, per-particle `charges`,
`ff_types`, `molecule_ids`, and unique `original_indices`, plus `bonds`
(index pairs), `constraints` (index, index, distance), and
`drude_parent_map` (Drude, parent pairs). Empty topology lists are valid.
Filtering updates particle indices and keeps `original_indices` unchanged.
It drops bonds and constraints that include removed particles. It raises an
error if a kept Drude's parent was removed. Charges and masses are never
combined. `read_traj` checks the metadata and returns it as `particle_metadata`.
Files without metadata still work.

## Preflight

Before using caches, training checks the HDF5 files in all configured splits.
It checks atom order and elements, masses, time steps, periodicity, positions,
velocities, and fixed cells. Each trajectory can have a different fixed cell.
Optional `atom_ids` or `particle_names` datasets let it check the order of atoms
with the same element. Without these, it cannot detect changes in their order.
To reject estimated velocities, set `reject_estimated_velocities: true` in the
training config or pass the same keyword to `TrajWindows`. It defaults to `false`.

## Constraints

Distance constraints are off by default. To enable them, pass
`constraint_projector=DistanceProjector([(i, j, distance_A), ...])` to
`equitraj.rollout.rollout`. Import `DistanceProjector` from `equitraj.constraints`.
Corrections use particle masses and keep the center of mass and total momentum
fixed. Masses must be positive, and constraints must be independent.
Corrections run after the optional thermostat. Temperature calculations account
for the constraints. Diagnostics record the remaining errors and correction
sizes. The rollout raises an error if corrections fail to reach the tolerances.
Periodic distances use the nearest image. Corrections can change kinetic energy
and transport, so check the results for your physical system.

## Validation

```sh
python scripts/validate_md.py model.h5 reference.h5 --charges '{"Na":1,"Cl":-1}' --fit-window-fs 100 500 --json-out validation.json --markdown-out validation.md
```

The CLI converts repo HDF5 units. `equitraj.validation.validate` accepts mappings
with `positions` in Å, `velocities` in Å/fs, `elements`, `masses`, optional `box`
in Å, and `dt_fs` equal to the **saved-frame interval**, rather than the
integration timestep. Shared keyword arguments can supply metadata for both
mappings. Trajectories are assumed unwrapped; use `wrapped=True` or `--wrapped`
for wrapped positions. Cells may be orthorhombic or reduced triclinic.

Results include every unordered species pair's partial RDF, integrated absolute
and relative RDF errors, per-species MSD, fitted D, VACF and cumulative VACF
integral, temperature and temperature drift, and the stable duration. Fits share
a window in fs. `max_lag` bounds cost; time-origin MSD is quadratic in frame
count without this limit. Nonperiodic RDFs are normalized pair-distance densities
and require `r_max`. A singleton species has zero same-species RDF because no
such pair exists. Periodic RDFs use finite-particle pair counts and frame volumes.

Conductivities are S/m, using charges in units of e, a common species-charge
mapping, and each trajectory's mean temperature (or `temperature_K` override).
Nernst–Einstein uses per-species D. Einstein–Helfand fits the time-origin MSD of
the collective charge coordinate, including cross correlations. Both use mean
volume. Nonneutral charge systems depend on the translational reference frame.
Choose a common rest frame; the validator does not subtract COM motion.

Default kinetic degrees of freedom are 3N; pass `dof`/`--dof` for constraints,
COM removal, or massless particles. Nonfinite frames terminate the stable prefix;
optional `min_distance` and `max_temperature` enable physical failure criteria.
A mapping can explicitly supply `stable_frames`. The optional Python `energy`
callable receives `(positions_A, velocities_A_per_fs, masses_amu, box_A_or_None)`
for each frame and must return total energy in kJ/mol. Its drift is kJ/mol/ns.
Short fitted transport estimates can be negative or unconverged.

`equitraj.validation.validate` accepts `charges` as one value per atom, in
units of the elementary charge, or as a dictionary of charges by element.
Pass integer `molecule_ids` to report each molecule's center of mass diffusion
and use its net charge for Nernst–Einstein conductivity. The center of mass
is weighted by atom mass. Molecular analysis is off by default.
Einstein–Helfand conductivity always uses the combined motion of atom charges.
Keep each molecule whole and use unwrapped coordinates in a common rest frame.
Motion of charges within a molecule can affect Einstein–Helfand estimates over
the measured time lags, even if those charges stay close together.
These estimates alone do not show that the trajectories are long enough.

## Pair exclusions

Pair exclusions are off by default. To exclude pairs, use a boolean `(N,N)`
mask with `True` for each excluded pair. The mask must be symmetric and use
the current particle order. Pass `excluded_pairs=mask` to `RepulsionPrior`,
`overlap_penalty`, `estimate_rmin`, and `rollout` (or `StabilityMonitor`).
These skip the excluded pairs when pushing atoms apart, damping velocities,
computing the overlap loss, estimating distance limits, and checking or
reporting overlaps. Pairs in the same molecule are not excluded automatically.
The prior's mask is not saved in checkpoints. Supply it again when loading
a model, using the same particle order. Other crash checks still apply.

## CPU proof and speed

```sh
python scripts/make_nacl_proof.py --out-dir data/nacl_proof --threads 2
python scripts/bench_openmm.py --xml data/nacl_proof/system.xml --pdb data/nacl_proof/system.pdb --platform CPU --model-random --channels 16 --heads 4 --stride-fs 10
```

The proof defaults to 64 Na and 64 Cl, 1200 K, three independently seeded runs,
2000 NVT equilibration steps and 301 saved NVE frames every five 1-fs steps.
`--shape`, `--spacing-nm`, `--temperature`, `--equil-steps`, `--frames`,
`--save-every`, `--n-traj` and `--dt-fs` control size and duration. It writes the
System XML and PDB for reuse. The LJ+Coulomb PME parameters illustrate the data
workflow; these short runs do not establish liquid phase or converged transport.
Runtime depends on the CPU and OpenMM build and has not been measured here.

The benchmark also accepts `--system argon` or another repo builder/saved system,
or a built `(system, topology, positions_nm)` via its Python `benchmark` function.
Use `--checkpoint` instead of `--model-random` for training/release checkpoints.
Checkpoint element embeddings must cover the benchmark species. Training
checkpoints need their original data config/HDF5 to resolve the stride.
Random models use nominal physical normalization scales solely to keep timing
contexts finite; no prediction accuracy is implied.

JSON reports atoms, steps/s, ns/day, steps/ns, and step duration for both engines.
The model is batch one and advances its context autoregressively, including
neighbor construction, sampling steps, and context updates. Warmup, compilation,
context initialization, minimization and file I/O are excluded. GPU work is
synchronized around timing. Model simulated time per step is the stride; OpenMM
simulated time per step is its integration timestep.

Synthetic tests are in `tests/test_adapters.py` and `tests/test_validation.py`.
ASE and MDTraj roundtrips skip when their optional libraries are absent. Tests
have not been executed as part of this change because Python execution was
prohibited.
