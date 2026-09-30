# Related work

How EquiTraj relates to existing work on learning MD with large time steps.

Sources: arXiv abstracts and HTML versions, read on 2026-09-29. For TrajCast and FlashMD, the numbers below come from the arXiv HTML. Everything else comes from abstracts only. Check the numbers against the papers before citing them.

## Summary table

| Work | Predicts | Deterministic or generative | Symmetry | History | Stride | Training | Stabilizers | Systems |
|---|---|---|---|---|---|---|---|---|
| **TrajCast** (Thiemann et al., arXiv 2503.23794, 2025) | Δx and v | deterministic | O(3)-equivariant MPNN, CG tensor products, l_max=2, both parities | 1 frame | 10–30× | single step, MSE | CSVR thermostat during rollout; momentum correction | paracetamol (gas), α-quartz (162 atoms), flexible SPC water (192 atoms) |
| **FlashMD** (Bigi, Chong, Kristiadi, Ceriotti, NeurIPS 2025) | mass-scaled Δq and p | deterministic | not equivariant (PET) with rotation augmentation | 1 frame | 1–64 fs | single step, MSE; energy-conservation loss term | momentum rescaling at inference so the reference MLIP energy is conserved; optional thermostats | water, solvated alanine dipeptide, Al surface, Li₃PS₄ |
| **Action learning** (Bigi, Spies, Ceriotti, arXiv 2508.01068) | symplectic, time-reversible map built from a learned generating function | deterministic | — | 1 frame | long steps | short reference trajectories | structure preservation by construction; can correct a direct predictor | several |
| **Hamiltonian flow maps** (Ripken et al., arXiv 2601.22123, 2026) | mean phase-space flow over Δt | deterministic | — | 1 frame | large | trained from force data only, no trajectories (Mean Flow consistency) | consistency condition | MLFF systems |
| **Timewarp** (Klein et al., NeurIPS 2023) | positions after 10^5–10^6 fs | generative (conditional normalizing flow) | permutation-equivariant transformer | 1 frame | huge | single step | Metropolis–Hastings correction, so it samples Boltzmann exactly but not dynamics | small peptides, implicit solvent |
| **ITO** (Schreiner, Winther, Olsson, NeurIPS 2023) | positions at many lag times | generative (diffusion) | SE(3)-equivariant | 1 frame | many lags | multi-lag | Chapman–Kolmogorov consistency | small molecules, peptides, coarse-grained |
| **TITO** (Viguera Diez, Schreiner, Olsson, arXiv 2510.07589) | positions | generative (equivariant flow matching) | equivariant, with rotational and permutational OT | 1 frame | fs to ns | multi-lag, transferable across molecules | Chapman–Kolmogorov tests | QM9 molecules, tetrapeptides, implicit solvent, no PBC |
| **MDGen** (Jing, Stärk, Jaakkola, Berger, NeurIPS 2024) | whole trajectory segments | generative (flow matching over time series) | SE(3) frames per residue | many frames | — | trajectory-level | — | peptides |

Background on autoregressive stability:
- Sanchez-Gonzalez et al. 2020 (GNS): noise injection on inputs.
- Brandstetter et al. 2022: pushforward trick.
- Lippe et al. 2023 (PDE-Refiner): refinement steps.

On symmetric layers: Passaro & Zitnick 2023 (eSCN) and Liao et al. 2024 (EquiformerV2).

## Closest work: TrajCast

TrajCast is nearly the same idea as our head A:
- equivariant MPNN;
- predicts displacements and velocities;
- fixed stride, one model per stride;
- evaluated with RDF, VDOS, MSD, and potential energy computed afterwards with the reference force field.

Useful details from the paper:
- Rollouts need a relative velocity MAE below ~2–4% to stay stable. A force field tolerates ~30% force error. This supports our view that error accumulation is the central problem.
- All reported rollouts run with a **CSVR thermostat** on the predicted velocities. The paper does not show pure NVE rollouts with no correction.
- Training is single step. No unrolling, no noise.
- MSD for water diverges beyond ~1 ps, which points to error accumulation.
- Code is open source (Apache-2.0, PyTorch, optional cuEquivariance) at https://github.com/IBM/trajcast.
- A search result suggests it was published in Nature Machine Intelligence in 2026. I could not open the page to confirm.

## Closest in spirit: FlashMD

Same goal of long-stride direct prediction, but a different philosophy:
- not equivariant; uses rotation augmentation;
- predicts mass-scaled quantities (Δq·√m and p/√m), which puts all elements on a similar scale;
- enforces energy conservation at inference by rescaling momenta with a reference MLIP energy. This needs a force-field evaluation at every step;
- the authors report equipartition violations and energy drift without the rescaling. Lyapunov divergence limits deterministic accuracy;
- single-step training.

## What EquiTraj adds

Written plainly, including where the overlap is large.

1. **Deterministic vs generative heads, compared directly.** TrajCast and FlashMD are deterministic. The generative methods (Timewarp, ITO, TITO, MDGen) predict positions only, without PBC, and target conformational sampling rather than NVE dynamics. We compare both heads on the same backbone and data, in full phase space (x, v), across strides.
2. **Training against drift, with ablations.** Both close works train single step. We measure noise injection, truncated unrolling, and full unrolling, and judge each by rollout stability.
3. **Model-only NVE first, fixes second.** We report rollouts with no thermostat and no energy rescaling. Then we add each fix as a separate switch: TrajCast-style CSVR thermostat, FlashMD-style energy rescaling, and constraint projection. This shows how much of the stability comes from the model itself.
4. **The stride sweep is a main result.** We report stability and accuracy against n for both heads, instead of picking one stride per system.
5. **Temporal context.** k > 1 past frames. Both close works are Markovian with one frame. At large strides with velocities as input this may not matter much. The ablation will show.
6. **Target system.** Li⁺ solvation in EC. FlashMD studies a solid electrolyte (Li₃PS₄); TrajCast studies water and quartz. The solvation shell and exchange dynamics are a new test.
7. **Layer type.** eSCN SO(2) convolutions instead of full CG tensor products. This is an efficiency choice, not a scientific claim.

What we do **not** add:
- Structure preservation. Symplecticity and time reversibility are the topic of action learning and Hamiltonian flow maps. Our model is a direct predictor, like TrajCast and FlashMD, and inherits the same weaknesses (energy drift, equipartition).
- Transferability across chemistry. ITO, TITO, Timewarp and FlashMD train universal or transferable models. We train one model per system and stride.

Risk: points 1–4 are a systematic study, not a new component. The contribution stands only if the ablations are clean and the results are reported honestly.

## Changes to PLAN.md because of this review

- Add **TrajCast as a baseline**, since the code is open. Train it on our data and run it with and without its CSVR thermostat.
- Add **inference switches**: S5 FlashMD-style energy rescaling with the reference force field; S6 CSVR thermostat. Report them separately and never mix them into model-only numbers.
- Add **mass-scaled targets** (FlashMD) as an alternative to per-element standardization.
- Track **relative velocity MAE** as an early warning metric (TrajCast threshold ~2–4%).

## Not yet read

- Memory-conditioned flow matching for stable autoregressive PDE solvers (arXiv 2602.06689). Possibly relevant to k > 1 context with a generative head.
- Kim, Lee, Shin, "Teaching Molecular Dynamics to a Non-Autoregressive Ionic Transport Predictor" (ICML 2026, arXiv 2605.09311). Argues against autoregressive surrogates for transport properties.

## Links

- TrajCast: https://arxiv.org/abs/2503.23794, code https://github.com/IBM/trajcast
- FlashMD: https://arxiv.org/abs/2505.19350
- Action learning: https://arxiv.org/abs/2508.01068
- Hamiltonian flow maps: https://arxiv.org/abs/2601.22123
- Timewarp: https://arxiv.org/abs/2302.01170
- ITO: https://arxiv.org/abs/2305.18046
- TITO: https://arxiv.org/abs/2510.07589
- MDGen: https://arxiv.org/abs/2409.17808
