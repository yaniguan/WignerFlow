# 120-Paper Plain-English Survey: Fast CPU Screening for Drugs, Batteries & Neural MD

Before designing `PocketMD` (`equitraj/pocket.py`), we surveyed **120 papers** (40 per domain; the 20 most-used ones per domain are listed below) to find which short-simulation numbers actually predict real-world quality — and which ones are misleading traps.

---

## 1. Drug-Protein Pocket Stability (40 Papers — Key Lessons)

**The Core Problem**: Docking programs score poses in milliseconds, but many top-scored poses clash slightly or drift out of the protein pocket as soon as atoms are allowed to move at room temperature (`300 K`). Full free energy perturbation (`FEP`) is much more expensive (many GPU-hours per molecule).

**What Works in Short Simulations**:
1. **Run 3+ short independent replicas (`OpenBPMD` — Lukauskis et al., *JCIM* 2022; Clark et al., *JCTC* 2016)**: Combine pose drift (`PoseScore`, RMSD of heavy drug atoms) with hydrogen-bond / contact survival (`PersScore`) into `CompScore = PoseScore - 5 * PersScore`. True binders settle quickly; bad decoys wobble or pop out.
2. **Check atom overlaps first (`PoseBusters` — Buttenschoen et al., *Chem. Sci.* 2024)**: Two non-bonded atoms closer than `0.75 × (r_vdW,i + r_vdW,j)` indicate a bad clash.
3. **Push gently to measure unbinding time (`tau-RAMD` — Kokh et al., *JCTC* 2018)**: Applying a small random push to the drug center of mass ranks how tightly different poses hold onto the pocket in picoseconds instead of microseconds.

*Representative cited works (`D1–D40`)*: Wang et al. (*JACS* 2015, FEP+), Ross et al. (*Commun. Chem.* 2023, FEP+ benchmark), Hahn et al. (*LiveCoMS* 2022, OpenFF PLB), Clark et al. (*JCTC* 2016, BPMD), Lukauskis et al. (*JCIM* 2022, OpenBPMD), Kokh et al. (*JCTC* 2018, $\tau$-RAMD), Buttenschoen et al. (*Chem. Sci.* 2024, PoseBusters), Corso et al. (*ICLR* 2023, DiffDock), Abramson et al. (*Nature* 2024, AlphaFold3), Su et al. (*JCIM* 2019, CASF-2016), Guterres & Im (*JCIM* 2020, High-Throughput MD), Majewski et al. (*Nat. Commun.* 2023, Dynamic Undocking), Liu et al. (*Acc. Chem. Res.* 2017, PDBbind), Mobley & Gilson (*Annu. Rev. Biophys.* 2017), Jorgensen (*Science* 2004), Shirts & Chodera (*JCP* 2008, MBAR), Mey et al. (*LiveCoMS* 2020), Gapsys et al. (*Chem. Sci.* 2020), Cournia et al. (*JCIM* 2017), Deflorian et al. (*JCIM* 2020).

---

## 2. Lithium Battery Electrolyte Screening (40 Papers — Key Lessons)

**The Core Problem**: Standard Nernst-Einstein conductivity ($\sigma_{\text{NE}}$) assumes every diffusing `Li+` and anion carries full independent charge. In real battery liquids, when `Li+` and `PF6-` stick together as a neutral pair (`CIP`), they diffuse together while carrying **zero net electric current**!

**What Works in Short Simulations**:
1. **Cluster Nernst-Einstein (`cNE` — France-Lanord & Grossman, *PRL* 2019)**: Group touching ions (`r < r_cut`) into clusters and sum each cluster's net charge $z_c$. Neutral `[Li+ PF6-]0` pairs have $z_c = 0$, so neutral pairs no longer count as charge carriers. It also averages over particles, so it is less noisy than the collective (Green-Kubo) estimate from a short run.
2. **Track the `Li+` first neighbor shell (`SSIP` / `CIP` / `AGG` — Yamada et al., *Nat. Energy* 2019; Borodin et al., *Acc. Chem. Res.* 2017)**:
   - **`SSIP`**: `Li+` surrounded only by solvent (fast bulk current).
   - **`CIP`**: `Li+` touching 1 anion (good for forming protective `LiF` battery interfaces when balanced).
   - **`AGG-net`**: Large connected ion clumps (high viscosity risk if too high).
3. **Electronic Charge Screening (`ECC` — Kirby & Jungwirth, *JPCL* 2019)** and **Small-Box Size Correction (`Yeh & Hummer`, *JPCB* 2004)**: Scale ion charges by about `0.75–0.8` to mimic electron cloud polarization, and add $\Delta D_{\text{YH}} \propto 1/L$ to remove small-box periodic drag.

*Representative cited works (`E1–E40`)*: Xu (*Chem. Rev.* 2004; *Chem. Rev.* 2014), Yamada et al. (*Nat. Energy* 2019), Borodin et al. (*Acc. Chem. Res.* 2017), France-Lanord & Grossman (*PRL* 2019, cNE), Self, Fong & Persson (*ACS Energy Lett.* 2019), Fong et al. (*Macromolecules* 2021), Yeh & Hummer (*JPCB* 2004), Kirby & Jungwirth (*JPCL* 2019, ECC), Bedrov et al. (*Chem. Rev.* 2019), de Blasio et al. (*Sci. Data* 2024, CALiSol-23), Gong et al. (*Nat. Mach. Intell.* 2025, BAMBOO), Zheng et al. (*Chem. Sci.* 2025, ByteFF), Kim et al. (*PNAS* 2023), Dave et al. (*Nat. Commun.* 2022), Kumar et al. (*Chem. Mater.* 2025, Electrolytomics), Rajput et al. (*JACS* 2015), Suo et al. (*Nat. Commun.* 2013, WiSE), Ren et al. (*Joule* 2019, LHCE), Holoubek et al. (*Nat. Energy* 2021, Sub-zero Li metal), Yao et al. (*Angew. Chem.* 2023).

---

## 3. Coarse-Stride & Edge Neural MD Health Checks (40 Papers — Key Lessons)

**The Core Problem**: A neural MD model can have tiny 1-step error on test frames and still blow up, freeze, or violate momentum conservation after 50 rollout steps (`Fu et al., TMLR 2023`).

**What Works Without Training 5 Extra Models**:
1. **Compare 1-step Euler vs. 2-step Heun (`delta_ODE`)**: When the model enters unfamiliar atom arrangements, its flow vector field bends sharply, causing Euler and Heun to disagree before the trajectory explodes.
2. **Measure multi-seed 3D distance spread (`sigma_flow`)**: Drawing 3 random noise seeds through the conditional flow measures uncertainty in a rotation-invariant way (`Vita & Schwalbe-Koda, MLST 2023`).
3. **Check `BAOAB` heat balance & zero center-of-mass drift (`Leimkuhler & Matthews, AMRX 2013`; `Harvey et al., JCC 1998`)**: Ensure kinetic energy stays in internal vibrations rather than whole-system drift.

*Representative cited works (`M1–M40`)*: Klein et al. (*NeurIPS* 2023, Timewarp), Schreiner et al. (*NeurIPS* 2023, ITO),Jing et al. (*NeurIPS* 2024, MDGen), Costa et al. (*arXiv* 2024, EquiJump), Fu et al. (*TMLR* 2023, Forces are not Enough), Stocker et al. (*MLST* 2022), Morrow et al. (*JCP* 2023), Vita & Schwalbe-Koda (*MLST* 2023), Bigi, Langer & Ceriotti (*ICML* 2025), Thiemann et al. (*arXiv* 2025, TrajCast), Batzner et al. (*Nat. Commun.* 2022, NequIP), Batatia et al. (*NeurIPS* 2022, MACE), Musaelian et al. (*Nat. Commun.* 2023, Allegro), Liao et al. (*ICLR* 2024, EquiformerV2), Lipman et al. (*ICLR* 2023, Flow Matching), Klein, Krämer & Noé (*NeurIPS* 2023, Equivariant FM), Leimkuhler & Matthews (*AMRX* 2013, BAOAB), Bussi, Donadio & Parrinello (*JCP* 2007, CSVR), Harvey, Tan & Cheatham (*JCC* 1998, Flying Ice Cube), Chmiela et al. (*Nat. Commun.* 2018, sGDML / MD17-CCSD(T)).
