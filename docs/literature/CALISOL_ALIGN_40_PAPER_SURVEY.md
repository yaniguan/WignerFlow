# 40-Paper Guide: Battery Conductivity Alignment & 1,000,000-Recipe Screening

Before building `WignerFlow-Align` (`equitraj/align.py`) and the 1,000,000-recipe funnel (`equitraj/funnel.py`), we reviewed **40 key papers** across three practical questions:
1. Why do conductivity simulations with polarizable force fields need large boxes and long runs, and how small can a periodic box be if we correct for box size?
2. Why does raw short-trajectory MD mispredict lab conductivity (`CALiSol-23`), and how can physics-guided post-training help?
3. How can a cheap multi-stage funnel screen a very large list of battery recipes on CPU?

The "How We Use It" column describes our design choices. It is not a result. Citations were spot-checked; please report any error you find.

---

## Group A (14 Papers): Battery Benchmarks, Polarizable Force Fields & Small-Box Corrections

| # | Paper | Venue | Plain-English Takeaway | How We Use It |
| :--- | :--- | :--- | :--- | :--- |
| **A1** | de Blasio, Elsborg, Vegge, Flores & Bhowmik (2024) | *Sci. Data* **11**, 750 (`CALiSol-23`) | 13,825 lab conductivity measurements from 27 papers, covering 14 lithium salts, 38 solvents, and `194–477 K`. | Our experimental test set; we split strictly by `(salt, solvent set)` so test recipes are truly unseen. |
| **A2** | Gong et al. (2025) | *Nat. Mach. Intell.* (`BAMBOO`) | Even a `3–5%` error in liquid density causes a `30–80%` error in viscosity and conductivity. | Match experimental molar density when building periodic battery boxes. |
| **A3** | Zheng et al. (2026) | *Nat. Commun.* **17**, 8276 (`ByteFF-Pol`; arXiv 2025) | A polarizable force field whose parameters come from a graph neural network trained on quantum data; predicts liquid and electrolyte density, viscosity, and conductivity. | Guides initial box packing and overlap checks. |
| **A4** | Dajnowicz et al. (2022) | *J. Phys. Chem. B* (`QRNN`) | Charge redistribution is needed to capture conductivity drop-off above `1.5 M` salt. | Motivates solvent-dependent charge screening. |
| **A5** | Bedrov et al. (2019) | *Chem. Rev.* | Review of polarizable force fields for ionic liquids and electrolytes; fixed full `±1e` charges tend to make ions stick too tightly and move too slowly. | Explains why polarizable or charge-scaled models are needed, and why their conductivity runs are expensive. |
| **A6** | Borodin (2009) | *J. Phys. Chem. B* (`APPLE&P`) | Separates ion motion into carrying the whole shell (vehicular) vs. hopping between shells. | Track shell-hopping rate `chi_hop` in trajectories. |
| **A7** | Huang et al. (2018) | *J. Comput. Chem.* (`OpenMM Drude`) | Drude polarizable particles double atom count and force `0.5–1.0 fs` steps. | Explains why polarizable MD cannot screen 1M recipes directly. |
| **A8** | Kirby & Jungwirth (2019) | *J. Phys. Chem. Lett.* (`ECC`) | Scaling ion charges by `1 / sqrt(eps_inf) ~ 0.75–0.80` mimics fast electron polarization at zero extra cost. | Built into our differentiable `ECC` effective-charge head. |
| **A9** | Yeh & Hummer (2004) | *J. Phys. Chem. B* | Exact formula `D_inf = D_box + 2.837 * kBT / (6 * pi * eta * L)` removes periodic self-drag in small boxes. | Used to correct diffusion coefficients from small periodic boxes. We have not yet measured the remaining error in this repo. |
| **A10** | Gabl et al. (2012) | *J. Chem. Phys.* | Local shell structure converges by `L ~ 25 Å`; collective dipole noise comes from tracking only 1 box vector. | Separate box-size error (`1/L` fix) from dipole noise (`cNE`). |
| **A11** | Celebi et al. (2021) | *Mol. Simul.* | Review of finite-size effects: diffusion from small periodic boxes is too low unless corrected. | Always apply `Yeh-Hummer` before comparing to lab conductivity. |
| **A12** | Maginn et al. (2019) | *LiveCoMS* (best practices for self-diffusivity and viscosity) | Running 8–20 short independent replicas in parallel converges faster in wall time than one huge run. | Matches batched `FastRollout` across replicas. |
| **A13** | France-Lanord & Grossman (2019) | *Phys. Rev. Lett.* (`cNE`) | **Cluster Nernst-Einstein (`cNE`)** groups stuck ions into clusters of net charge `z_c`: neutral `[Li+ PF6-]0` pairs get `z_c = 0` automatically! | Core conductivity estimator for short trajectories: fast convergence without counting dead neutral pairs. |
| **A14** | Fong et al. (2021) | *Macromolecules* | Connects ion-ion motion correlations to true `Li+` transference number `t+`. | Used to score top candidates in Tier 3. |

---

## Group B (14 Papers): Post-Training Neural MD Rollouts to Match Real Observables

| # | Paper | Venue | Plain-English Takeaway | How We Use It |
| :--- | :--- | :--- | :--- | :--- |
| **B1** | Thaler & Zavadlav (2021) | *Nat. Commun.* (`DiffTRe`) | Match lab measurements without backpropagating through millions of MD steps. | Combine short unrolled windows with a differentiable physics head. |
| **B2** | Fuchs et al. (2025) | *Comput. Phys. Commun.* (`chemtrain`) | Mix microscopic trajectory loss with macroscopic lab targets so the model stays physical. | Keep step loss + temperature/diffusion alignment together. |
| **B3** | Schoenholz & Cubuk (2020) | *NeurIPS* (`JAX MD`) | Differentiable periodic wrapping and neighbor lists. | Guides `EquiTraj.predict_with_grad` in periodic boxes. |
| **B4** | Wang et al. (2023) | *J. Chem. Phys.* (learning pair potentials with differentiable simulation) | Backpropagating through long chaotic MD runs gives unstable gradients. | Stage 1 post-training uses short unrolled windows. |
| **B5** | Fu et al. (2023) | *TMLR* (`Forces are not Enough`) | Low single-step error does NOT guarantee stable rollouts or accurate diffusion. | Directly explains why zero-shot WignerFlow needs transport post-training. |
| **B6** | Greener (2025) | *PNAS* (reversible simulation) | Gets gradients by running the simulation backward in time, with low memory; trains force fields to match diffusion and other time-dependent data. | Supervise multi-step mean-squared displacement (`MSD`) slope. |
| **B7** | Raja et al. (2025) | *TMLR* (`StABlE`) | Penalizing close atom overlaps during rollout training prevents crashes. | Add overlap and kinetic-temperature penalties in Stage 1. |
| **B8** | Bolhuis et al. (2023) | *J. Chem. Phys.* | Path reweighting connects short jump statistics to long-time rates. | Supports aligning jump statistics to bulk transport. |
| **B9** | Domingo-Enrich et al. (2025) | *ICLR* (`Adjoint Matching`) | Fine-tune flow-matching models toward physical rewards without forgetting the base distribution. | Keep EMA anchor weights during post-training. |
| **B10** | Havens et al. (2025) | *ICML* (`Adjoint Sampling`) | Trains a diffusion sampler from an energy function alone (no data), using adjoint matching; tested on molecular conformers. | Background on reward- or energy-driven fine-tuning of generative samplers. |
| **B11** | Schreiner et al. (2023) | *NeurIPS* (`ITO`) | Predict positions across larger time jumps to skip fast bond vibrations. | Matches `stride_conditioning` in `EquiTraj`. |
| **B12** | Klein et al. (2023) | *NeurIPS* (`Timewarp`) | A conditional normalizing flow proposes large time jumps for small peptides, with auxiliary velocity-like variables and optional Metropolis-Hastings correction. | Background for large-step models; our `VACF` term in Stage 1 is our own choice. |
| **B13** | Costa et al. (2024) | *arXiv* (`EquiJump`) | SO(3)-equivariant stochastic interpolants that take large time jumps in protein MD. | Background for equivariant large-step models (our 2-step Heun choice is our own). |
| **B14** | Jing et al. (2024) | *NeurIPS* (`MDGen`) | Multi-frame coherence improves diffusion and transition rates. | Supervise multi-step velocity and displacement smoothness. |

---

## Group C (12 Papers): Screening 1,000,000+ Candidates with Multi-Stage Funnels

| # | Paper | Venue | Plain-English Takeaway | How We Use It |
| :--- | :--- | :--- | :--- | :--- |
| **C1** | Qu et al. (2015) | *Comput. Mater. Sci.* (`Electrolyte Genome`) | Filter millions of candidates with cheap rules first, then run 3D simulations on survivors. | 3-tier funnel structure (`1,000,000 -> 180 -> 20 -> Top 15`). |
| **C2** | Dave et al. (2020) | *Cell Rep. Phys. Sci.* | Parameterize mixed-salt, mixed-solvent recipes over a clean grid of ratios and concentrations. | Defines our `10 salts × 20 solvent bases × 100 additives × 10 ratios × 5 concentrations = 1,000,000` space. |
| **C3** | Dave et al. (2022) | *Nat. Commun.* | Mixing a high-dielectric cyclic solvent with a thin linear solvent creates a conductivity peak. | Captures the `0.7–1.0 M` bell-curve peak in `EC:DMC`. |
| **C4** | Kim et al. (2023) | *PNAS* | Fluorine-to-carbon (`F/C`) and oxygen-to-carbon (`O/C`) ratios control `Li+` shell pairing and battery life. | Guides our 100 new fluorinated ester, ether, acetal, and nitrile additives. |
| **C5** | Kumar et al. (2025) | *Chem. Mater.* (`Electrolytomics`) | Trains ML models for conductivity, oxidative stability, and Coulombic efficiency, combines them into one electrolyte score (`eScore`), screens large molecule libraries, and tests top picks in the lab. | Inspires our combined room-temperature + `-20 °C` score (named `eScore` after this paper). |
| **C6** | Ruddigkeit et al. (2012) | *J. Chem. Inf. Model.* (`GDB-17`) | Enumerates small organic molecules systematically with RDKit validity checks. | Background for systematic enumeration of small molecules. Our 100 additive SMILES are hand-made and not yet checked with RDKit. |
| **C7** | Lyu et al. (2019) | *Nature* | Huge chemical libraries find better winners, but 3D physics checks are needed to remove false positives. | Tier 2 and Tier 3 run 3D periodic-box checks on Tier 1 hits. |
| **C8** | Gorgulla et al. (2020) | *Nature* (`VirtualFlow`) | Open-source platform for docking very large libraries on many CPUs. | Same idea of cheap filters before expensive checks. Our funnel run time is in `results/funnel_1m/top15_novel_electrolytes.json`. |
| **C9** | Gentile et al. (2022) | *Nat. Protoc.* (`Deep Docking`) | Trains a fast classifier on docking scores of a small sample, then uses it to skip most of a huge library. | Same idea of a cheap learned filter in Tier 1. |
| **C10** | Buttenschoen et al. (2024) | *Chem. Sci.* (`PoseBusters`) | Check van der Waals atom overlaps and bond lengths explicitly. | Used in Tier 2 `Electrolyte-Busters` box audit. |
| **C11** | Su et al. (2019) | *J. Chem. Inf. Model.* (`CASF-2016`) | Report both accuracy (`R²`, `MAE`) and ranking (`Spearman rho`, `AUROC`). | Standard metrics across all our benchmarks. |
| **C12** | Kokh et al. (2018) | *J. Chem. Theory Comput.* (`tau-RAMD`) | Random pushes on a ligand speed up unbinding so relative residence times can be ranked from short runs. | Used for drug-pocket residence ranking; using it for `Li+` shells is our own idea. |
