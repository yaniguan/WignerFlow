# Roadmap: stable 20 ns rollouts and partial replacement of OpenMM for PFF MD

The immediate target is a measured, resumable 20 ns argon rollout with correct equilibrium and transport statistics. Partial replacement of OpenMM for polarizable force-field (PFF) dynamics requires additional chemistry, electrostatics, constraints, and validated fallback interfaces. A completed trajectory alone is insufficient evidence of useful dynamics. No 20 ns success or speedup is claimed here.

## Audit and gaps

- Argon data use `dt_fs = 5 fs`; stride 10 integration steps means 50 fs per autoregressive step. Thus 20 ns requires **400,000 autoregressive steps**; at a 10 fs stride it requires **2,000,000**. Flow sampling multiplies the network work within each autoregressive step.
- The argon retrain at training step 6000 currently crashes at approximately 865 rollout steps, about 43 ps. This is the observed baseline, not evidence that the stability configuration reaches 20 ns. See `STABILITY.md`.
- Types are per element only. Distinct chemical environments share an embedding: carbonyl/ether O in EC/solvents versus O in TFSI, for example. PF6 itself contains no O; if oxygen-bearing species are present with PF6, their O atoms still share the element type. Add force-field atom types, partial charges, topology, molecular identity, and polarization parameters before interpreting molecular PFF transfer.
- Spatial message passing uses a 6 Å cutoff in the argon stability config. There is no long-range electrostatic or polarization model. Plan an Ewald/PME-like long-range head, or charge-conserving latent charges with explicit long-range interactions and induced dipoles/Drude dynamics. A local repulsion prior cannot supply these interactions.
- PBC uses minimum-image geometry, with orthorhombic and reduced triclinic support. Require the cutoff to be smaller than half the shortest relevant cell height, and test boundary crossings, skewed cells, and unwrapping. Arbitrary unreduced triclinic cells need explicit reduction/validation. A half-box check on diagonal entries alone is insufficient for skewed cells.
- No bonded topology enters the model. Bonds can be supplied to crash checks, but that does not teach bond/angle/torsion chemistry. Add connectivity, exclusions, 1–4 rules, atom typing, and molecular geometry losses.
- There are no integration constraints or rigid-water enforcement. Constraint projection and corresponding velocity/temperature degrees of freedom are needed for constrained force fields. Massless virtual sites and Drude mass partitioning need explicit handling.
- The default neighbor list is brute-force O(N²); `cell_list` exists. Benchmark and verify it under PBC and skewed cells. The repulsion prior and monitor also have pairwise costs, so changing the graph builder alone does not establish linear scaling.
- The flow head requires n sampling steps for every physical step. Profile sampling count against stability and transport accuracy; investigate distillation/regression only with matched validation.
- The learned transition has no Hamiltonian. Energy can only be checked post hoc through a supplied force-field evaluator. Add energy/force consistency or a structured conservative component; measure secular energy and temperature drift rather than hiding it with a thermostat.
- There is no NPT/barostat evolution. The dataset loader uses a fixed box per trajectory. Begin with fixed-volume NVE/weak NVT, then design cell/virial dynamics and ensemble tests before NPT claims.
- There is no production OpenMM stepping interoperability beyond data generation (plus offline timing/analysis hooks). Design a common state/units/topology interface, synchronized force-field checks, uncertainty-triggered fallback, and validated state handoff before hybrid deployment.
- Long-run orchestration is notebook-local: the existing rollout API does not return its final context or RNG. The notebook captures history with a monitor and commits HDF5 plus a state checkpoint. Promote this to a tested library API later. Training continuation currently restores weights/optimizer/step, not RNG/DataLoader state.

## What exists now on this branch

| Module/config | Existing capability and boundary |
|---|---|
| `equitraj/model.py`, `equitraj/neighbors.py` | Equivariant local trajectory model, regression/flow prediction, minimum-image geometry, brute-force and cell-list graphs |
| `equitraj/train.py` | Unroll curriculum, noise schedule, weighted unroll loss, overlap penalty, automatic `ckpt.pt` continuation, validation-only `best_stable.pt` selection |
| `equitraj/prior.py` | Data-estimated per-element-pair repulsion wall; optional velocity damping changes kinetics |
| `equitraj/rollout.py`, `equitraj/monitor.py` | Float64 position accumulation, crash criteria, offending atoms/history, temperature/KE drift, optional external energy diagnostics; optional CSVR switch |
| `equitraj/adapters.py`, `equitraj/trajfile.py`, `scripts/import_md.py` | External MD import, trajectory split config, repo HDF5 units, Drude retention/drop option; not a complete PFF parameter/topology representation |
| `scripts/gen_data.py`, `scripts/make_nacl_proof.py` | OpenMM reference generation and illustrative NaCl LJ+Coulomb PME proof with system XML/PDB |
| `equitraj/validation.py`, `scripts/validate_md.py` | Partial RDFs, MSD/diffusion, VACF, temperature drift, stable prefix, species-charge conductivity and optional Python energy callback |
| `equitraj/binding.py` | Pocket-aligned ligand RMSD, internal/backbone RMSD, contacts, COM displacement, geometric H-bond contacts, bond/angle distributions, interaction-energy and free-energy proxies |
| `scripts/evaluate.py`, `equitraj/evaluation.py` | Checkpoint loading, test multistart/multiseed rollout evaluation; argon-specific criteria are not generic molecular acceptance gates |
| `equitraj/profiling.py`, `scripts/profile_run.py`, `scripts/bench_openmm.py` | Training phase timing, graph timing, peak GPU memory, utilization sampler, torch.profiler traces, batch-one throughput vs same-system OpenMM |
| `configs/argon/B_S3a_stable.yaml`, `configs/nacl/proof_cpu.yaml` | Full argon stability recipe and short multi-species CPU proof recipe, respectively; neither proves a 20 ns result |
| `colab/long_rollout_20ns.ipynb` | A100/Drive workflow for generation/import, stable training, required profiling, chunked rollout continuation, held-out validation, and binding placeholder |

## Design for partial OpenMM replacement

Use one versioned state contract: atom order, unwrapped positions, velocities, row-vector cell, physical clock, atom types, masses, charges, bonded topology, exclusions, constraints, virtual-site/Drude parent mappings, and force-field identity. Preserve these fields through import, checkpoints, fallback, and export. Reject incompatible state/configuration changes on resume. Split train/validation/test by independent trajectory and later by chemistry, composition, protein, and ligand scaffold.

Start with a learned short-range transition and explicit long-range electrostatics/polarization, with conservation and constraint checks. Compare explicit Drude propagation against an adiabatic/latent polarization head; validate dipoles, dielectric response, and charge conservation, not only coordinate errors. Train across perturbations and progressively longer autoregressive contexts. Choose checkpoints by validation survival first and structural/thermal fidelity second, then freeze selection before testing.

Keep OpenMM as reference and safety integrator initially. Define uncertainty and geometry/energy thresholds before tests; on failure, hand a complete compatible state to OpenMM and record time in each engine. Test force/velocity/constraint consistency at every switch. Report model-only, hybrid, and thermostatted results separately. A force-field check every step may remove any speed advantage; measure the actual cadence and full end-to-end throughput.

Required evidence includes steps/s, ns/day, flow sampling count, batch size, atom count, physical stride, GPU model, OpenMM precision/platform/timestep, peak allocated GPU memory, utilization samples, and data/forward/backward/optimizer/neighbor-list timing. Archive Chrome traces and operator tables. Report compute-only and monitor/I/O-inclusive performance, uncertainty across runs, and all crashes. Random-weight timing is compute cost only. Speed comparisons must use the same physical system and an accuracy-qualified model.

## Phased roadmap and exit criteria

The thresholds below are proposed gates to preregister before testing, not measured accomplishments. Use at least five independent starts/seeds, trajectory-level confidence intervals, and reference uncertainty. Lengthen reference sampling when transport confidence intervals are too broad; a single 20 ns trajectory is not a convergence guarantee.

### Phase 0 — Argon stability

Reproduce the ~43 ps baseline, enable the repulsion prior, annealed noise, unroll curriculum, overlap penalty, and stability-first checkpoint selection. Grow selection horizons on validation only. Test uninterrupted versus resumed chunks with matching history/RNG and check no duplicated or missing frames. Ablate each stability mechanism and report any thermostat/damping explicitly.

Measure survival/stable length, failure reasons, minimum distance, RDF peak position/height and integrated error, MSD/D, VACF zero and spectrum, temperature/KE/force-field energy drift, and throughput. **Exit:** every preregistered test run reaches 20 ns without a crash; mean temperature within 5%, RDF first peak position within 0.05 Å and height within 10%, diffusion within 20%, and VACF first zero within 10% where resolved. Drift must fit reference uncertainty and a prespecified NVE budget; report failures where no VACF zero is resolved. Meet the gates without a thermostat before claiming NVE replacement.

### Phase 1 — Molten NaCl / NaCl(aq)

Use the illustrative NaCl proof to test multi-species plumbing, then acquire a validated molten-salt force field and phase diagram. Add long-range Coulomb treatment. Treat aqueous NaCl separately: water constraints, ions/solvent topology, and solvent polarization are additional requirements.

Measure all partial RDFs, coordination numbers, species diffusion, VACFs, charge neutrality, Nernst–Einstein and collective Einstein–Helfand conductivity, temperature/energy drift, density and phase. **Exit:** five independent 20 ns runs per preregistered state point survive; RDF/coordination errors below 10% and diffusion/conductivity errors below 20% or statistically indistinguishable within sufficiently tight reference intervals. Demonstrate correct phase and size/cutoff robustness. An illustrative short LJ+Coulomb proof is not this exit gate.

### Phase 2 — Li/Na-PFF electrolytes

Start with LiPF6 in EC/EMC and NaPF6 in PC, with force-field-specific composition, concentration, temperature, and density. Support AMOEBA multipoles/induced dipoles or Drude particles explicitly; do not treat one as a drop-in representation of the other. Introduce atom types, partial charges, bonded topology, molecular identity, constraints, and polarization features. Compare retained Drudes with a validated coarse polarization representation.

Measure ion–solvent/anion RDFs, coordination and residence times, ion-pair populations, molecular conformations, species/center-of-mass diffusion, VACF, correlated conductivity, transference proxies, dipole distributions/dielectric response, polarization energy, and drift. Atom-specific charges are required; the current element-charge CLI is inadequate for PFF conductivity. **Exit:** independent 20 ns trajectories survive across held-out compositions/temperatures; coordination/speciation errors below 10%, diffusion/conductivity within 20% with converged uncertainty, and polarization observables within reference intervals. Demonstrate useful ns/day after explicit electrostatics, polarization, constraints, monitoring, and fallback costs.

### Phase 3 — Drug-like small molecules in water

Use public MD17/rMD17 for vacuum molecular dynamics diagnostics, SPICE for chemical/energy/force coverage, alanine dipeptide in TIP3P for solvated torsional sampling, and FreeSolv molecules for hydration-oriented tests. These have different labels/ensembles: static SPICE structures do not supply temporal velocity training pairs, and vacuum MD17 is not a solvent trajectory dataset. Generate matched OpenMM/PFF trajectories where needed. Add scaffold and conformer holdouts, protonation/tautomer states, and stereochemistry checks.

Measure bond/angle/torsion distributions, conformer populations and transitions, molecular RMSD, solvent RDFs, diffusion, energy drift, and hydration free-energy proxies. Pocket-based ligand metrics are not applicable to isolated solutes; use internal RMSD and solute–water contacts instead. **Exit:** five independent 20 ns runs per selected test molecule survive with no chemistry/constraint failures, bond/angle/torsion distributions and major conformer populations within reference uncertainty (target ≤10% population error), and solvent/transport gates comparable to Phase 2. Validate hydration thermodynamics with a proper estimator before claiming FreeSolv agreement.

Include AIDD-designed molecules as an explicit out-of-distribution set after this gate: use scaffold/charge/protonation/functional-group holdouts, flag unsupported chemistry, generate new reference data, and require the same prospective tests. Novel designed chemistry cannot inherit confidence from in-distribution survival alone.

### Phase 4 — Protein–ligand binding

Begin with public T4 lysozyme L99A + benzene. Expand using MISATO, PLINDER/PDBbind subsets, mdCATH for protein dynamics, and the Schrödinger/OpenFE JACS benchmark set for free-energy evaluation. Verify availability, licensing, atom mappings, force fields, protonation, and exact benchmark definitions before ingestion; structures/affinity labels are not ready-made matched PFF dynamical trajectories. Use protein-family and ligand-scaffold splits. Generate replicated solvent trajectories and matched perturbation legs for thermodynamic claims.

Use `equitraj.binding.py` for pocket-aligned ligand RMSD, ligand internal RMSD, protein backbone RMSD, contact persistence, COM displacement, geometric H-bond contacts, and bond/angle distributions. Compare interaction energies, `mean_difference`, and `exponential_average` as free-energy proxies with stated units/beta and sampling overlap. Geometric H-bond contacts do not establish donor/acceptor chemistry; energy proxies do not estimate binding affinity without a valid thermodynamic cycle.

**Exit:** five independent 20 ns runs per held-out complex survive without geometry failures; reference-consistent ligand RMSD/contact distributions and protein flexibility (target contact occupancy error ≤10 percentage points), no systematic artificial pose locking or unbinding, and transport/thermal fidelity. For free-energy replacement, additionally demonstrate replicated, converged alchemical cycles, overlap, cycle closure, and prespecified error against the matched reference benchmark (proposed MAE ≤1 kcal/mol). Twenty nanoseconds of pose retention alone does not meet that gate. Apply prospective AIDD-designed ligand tests only after protein/scaffold holdout validation and uncertainty-triggered OpenMM fallback pass.

## Data needed from Yani for Li-PFF/Na-PFF

- Preferred: OpenMM DCD/XTC coordinates **plus actual velocity arrays**, PDB topology, and serialized system XML. DCD/XTC alone does not contain the needed velocities or force-field topology. Alternatively provide HDF5/extxyz with positions, velocities, box, elements/masses, charges, polarizabilities, and Drude flags/parent mappings. Include units, atom order, wrapping convention, force-field atom types, bonds/angles/torsions, constraints, multipoles/induced dipoles where relevant. The present generic importer does not preserve every PFF field; retain originals/sidecars until the schema is extended.
- Save at intervals **≤10 fs with velocities** and record the integrator timestep and save frequency separately. Supply NVE or weakly thermostatted segments and thermostat/Drude thermostat details. Retain temperature/energy/force diagnostics; avoid inferring fast dynamics from sparse coordinates.
- At least **five independent trajectories**, split by trajectory into train/validation/test; target total duration at least a few ns plus **one 20 ns held-out reference** for long validation. Reserve the long reference before model selection. More replicas may be needed for converged transport and uncertainty.
- All force-field files and versions (AMOEBA/Drude parameters, custom forces/plugins), system XML/PDB, water model, exclusions, constraints, virtual sites, masses/Drude mass partitioning, polarization convergence settings, integrator and timestep.
- Temperature, pressure/ensemble, density/box, salt concentration, composition, solvent ratios, molecule counts, initialization/equilibration, independent seeds, and any COM removal.
- OpenMM ns/day on her hardware: GPU/CPU model, precision, platform/plugin versions, timestep, constraints, PME settings, particle count including Drudes, output cadence, and timing scope. Obtain matched on-A100 timings for direct notebook comparisons.

## Measured numbers

TBD (filled in by PR)
