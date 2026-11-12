# Literature survey: Li+ solvation shells, MD17/MD22 rollout tests, and MD tools (40 papers)

**Plain-English summary.** This note collects 40 papers we read before writing three tools:
(1) a Li+ solvation-shell analyzer (who sits next to each Li+ ion: solvent, anion, or additive),
(2) a large-time-step test on public MD17/MD22 molecules, and (3) a small command-line tool.
Each entry gives the citation, what the paper does, and what we took from it.
Formulas and target ranges written here are our working choices for the code, not results
from WignerFlow. Citations were spot-checked against Crossref; please report any error you find.
**Scope:** Pre-implementation theoretical, algorithmic, and empirical foundation for WignerFlow's next three modules:
1. **Fine-Grained Battery Electrolyte Solvation & Exchange Kinetics Analyzer** (`SSIP`/`CIP`/`AGG`, partial RDFs, running $CN(r)$, additive enrichment, cage lifetimes $\tau_{\text{ex}}$, and `cNE` ion-pair transport)
2. **Pure-MD Trajectory Rollout & Coarse-Stride Benchmark Suite** (`MD17`/`rMD17`/`MD22`, `Fu et al.` stability/$h(r)$/RDF/`VDOS`, and coarse-stride operator baselines)
3. **Fast Composable Scientific MD CLI & Calculator Ecosystem**

---

## Part I: Fine-Grained Battery Electrolyte Solvation Shell (`SSIP`/`CIP`/`AGG`), Radial Distribution & Exchange Kinetics (Papers 1–20)

### Canonical Mathematical Formalism & Cutoff Conventions (Synthesized from Papers 1–20)

#### 1. Partial Radial Distribution Function (RDF) & Running Coordination Number
For central $\text{Li}^+$ ions ($N_{\text{Li}}$) and coordinating donor species $\beta \in \{\text{O}_{\text{solv}}, \text{O}_{\text{add}}, \text{O}_{\text{anion}}, \text{F}_{\text{anion}}, \text{N}_{\text{anion}}\}$ with bulk number density $\rho_\beta = N_\beta / V$ in periodic volume $V$:
$$g_{\text{Li},\beta}(r) = \frac{V}{N_{\text{Li}} N_\beta} \left\langle \sum_{i=1}^{N_{\text{Li}}} \sum_{j=1}^{N_\beta} \frac{\delta\!\left(r - \|\mathbf{r}_i - \mathbf{r}_j\|_{\text{PBC}}\right)}{4\pi r^2} \right\rangle_t, \qquad CN_{\text{Li},\beta}(r_c) = 4\pi \rho_\beta \int_0^{r_c} r^2 g_{\text{Li},\beta}(r)\,dr$$
where $r_c$ is placed at the **first local minimum** following the primary solvation peak of $g_{\text{Li},\beta}(r)$:
- **$\text{Li–O}_{\text{solv}}$ (EC, DMC, EMC, PC, FEC, DME, DOL, FDMB, F4DEE):** Peak $1.95\text{–}2.10\text{ Å}$; cutoff **$r_c = 2.65\text{–}2.85\text{ Å}$** (canonical default: **$2.80\text{ Å}$**).
- **$\text{Li–O}_{\text{anion}}$ ($\text{FSI}^-$, $\text{TFSI}^-$, $\text{DFOB}^-$, $\text{NO}_3^-$):** Peak $1.95\text{–}2.10\text{ Å}$; cutoff **$r_c = 2.75\text{–}2.95\text{ Å}$** (canonical default: **$2.85\text{ Å}$**).
- **$\text{Li–F}_{\text{anion/solv}}$ ($\text{PF}_6^-$, $\text{BF}_4^-$, $\text{FSI}^-$, fluorinated ethers):** Peak $1.85\text{–}2.05\text{ Å}$; donor-atom cutoff **$r_c = 2.60\text{–}2.90\text{ Å}$** (canonical default: **$2.75\text{ Å}$**; when using polyatomic center-atom proxies, $r_c(\text{Li–P}) \approx 4.1\text{ Å}$, $r_c(\text{Li–B}) \approx 3.5\text{ Å}$).
- **$\text{Li–N}_{\text{anion/nitrile}}$ ($\text{FSI}^-/\text{TFSI}^-$ imide N, AN $-\text{C}\equiv\text{N}$):** Peak $2.05\text{–}2.20\text{ Å}$; cutoff **$r_c = 2.85\text{–}3.20\text{ Å}$** (canonical default: **$3.00\text{ Å}$**).

#### 2. Bipartite Graph Speciation (`SSIP` / `CIP` / `AGG`) & Additive Enrichment
Define the instantaneous cation–anion coordination adjacency matrix $A_{ia}(t) \in \{0,1\}$, where $A_{ia}(t) = 1$ iff $\min_{k \in \mathcal{D}(a)} \|\mathbf{r}_{\text{Li}_i}(t) - \mathbf{r}_{a,k}(t)\|_{\text{PBC}} \le r_c(\text{Li}, \text{type}(k))$ over donor atoms $\mathcal{D}(a)$ of anion $a$. Let $n_a(i,t) = \sum_a A_{ia}(t)$ be the anion count in $\text{Li}_i^+$'s primary shell and $n_c(a,t) = \sum_i A_{ia}(t)$ be the cation-bridging degree of anion $a$. Let $\mathcal{C}(i,t) = (V_+, V_-)$ be the connected component of $\text{Li}_i^+$ in the bipartite graph $A(t)$:
$$\begin{aligned}
\text{SSIP}(\text{Li}_i, t) &\iff n_a(i,t) = 0 \quad (|V_+| = 1,\; |V_-| = 0) \\
\text{CIP}(\text{Li}_i, t) &\iff |V_+| = 1 \;\land\; |V_-| = 1 \quad \bigl(\text{single } \text{Li}^+\text{–anion pair: } n_a(i,t)=1 \text{ and } n_c(a^*,t)=1\bigr) \\
\text{AGG}(\text{Li}_i, t) &\iff |V_+| + |V_-| \ge 3 \quad \bigl(n_a(i,t) \ge 2 \text{ or } \exists a:\, A_{ia}(t)=1 \land n_c(a,t) \ge 2\bigr)
\end{aligned}$$
*(Note: Some high-throughput packages such as `MISPR` also report a local cation-only variant where `CIP` is $n_a(i)=1$ and `AGG` is $n_a(i)\ge 2$; WignerFlow should compute both the exact bipartite connected-component partition $\chi_{\text{SSIP}} + \chi_{\text{CIP}} + \chi_{\text{AGG}} = 1$ and the anion-centric Raman proxy $\text{Free} / \text{CIP}(n_c=1) / \text{AGG-I}(n_c=2) / \text{AGG-II}(n_c \ge 3)$.)*

For a minority additive $A$ (bulk solvent mole fraction $x_A^{\text{bulk}} = N_A / \sum_s N_s$), the **primary-sheath competitive enrichment factor** $\alpha_{\text{enr}}(A)$ and preferential solvation free energy $\Delta G_{\text{pref}}$ relative to co-solvent $B$ are:
$$\alpha_{\text{enr}}(A) = \frac{CN_{\text{Li},A} / \sum_{s\in\text{solv}} CN_{\text{Li},s}}{N_A / \sum_{s\in\text{solv}} N_s}, \qquad \Delta G_{\text{pref}}(A\text{ vs }B) = -k_B T \ln\!\left(\frac{CN_{\text{Li},A}/N_A}{CN_{\text{Li},B}/N_B}\right)$$

#### 3. Cage / Ligand Residence Autocorrelation & Exchange Lifetime $\tau_{\text{ex}}$
With pair indicator $h_{ij}(t) = \mathbb{I}[\|\mathbf{r}_{\text{Li}_i}(t) - \mathbf{r}_j(t)\|_{\text{PBC}} \le r_c]$:
$$C_{\text{cage}}^{\text{inter}}(t) = \frac{\langle h_{ij}(t_0)\, h_{ij}(t_0 + t) \rangle}{\langle h_{ij}(t_0) \rangle}, \qquad S_{\text{cage}}^{\text{cont}}(t) = \frac{\left\langle h_{ij}(t_0) \prod_{\tau=t_0}^{t_0+t} h_{ij}(\tau) \right\rangle}{\langle h_{ij}(t_0) \rangle}$$
Fitting $C_{\text{cage}}(t) = A \exp[-(t/\tau_0)^\beta]$ (KWW stretched exponential) yields the mean exchange residence time and vehicular-vs-structural exchange length:
$$\tau_{\text{ex}} = \int_0^\infty C_{\text{cage}}(t)\,dt = \frac{A\,\tau_0}{\beta}\,\Gamma\!\left(\frac{1}{\beta}\right), \qquad L_{\text{ex}} = \sqrt{6 D_{\text{Li}}\, \tau_{\text{ex}}}$$

---

### Annotated Survey of Papers 1–20

#### 1. Electrolytes and Interphases in Li-Ion Batteries and Beyond
- **Citation:** Kang Xu, *Chemical Reviews* **114**(23), 11503–11618 (2014). DOI: `10.1021/cr500003w`.
- **Core Problem & Method:** Comprehensive synthesis of nonaqueous $\text{Li}^+$ electrolyte thermodynamics, ion solvation sheaths, transport, and SEI/CEI formation mechanisms across carbonates, ethers, and fluorinated salts, contrasting bulk dielectric permittivity ($\varepsilon_r$) against local Gutmann donor number ($DN$).
- **Key Metrics / Formulas:** Establishes the canonical tetrahedral $\text{Li}^+$ primary coordination number $CN_{\text{tot}} \approx 4.0\text{–}4.5$ in cyclic/acyclic carbonate mixtures (e.g., EC preferentially displacing linear DMC/EMC due to higher dipole/binding energy) and the classical Nernst–Einstein conductivity $\sigma_{\text{NE}} = \frac{F^2}{RT}\sum_i c_i z_i^2 D_i$ with ionicity (Haven-like ratio) $\mathcal{I} = \sigma_{\text{exp}}/\sigma_{\text{NE}} < 1$.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's solvation module must never treat mixed solvents as homogeneous media; it must resolve species-specific $CN_{\text{Li},s}(r_c)$ and primary-sheath composition ratios because SEI chemistry is dictated by the inner-sheath stoichiometric composition carried by migrating $\text{Li}^+$, not bulk mole fractions.

#### 2. The Coupling between Stability and Ion Pair Formation in Magnesium Electrolytes from First-Principles Quantum Mechanics and Classical Molecular Dynamics
- **Citation:** Nav Nidhi Rajput, Xiaohui Qu, Niya Sa, Anthony K. Burrell, and Kristin A. Persson, *Journal of the American Chemical Society* **137**(9), 3411–3420 (2015). DOI: `10.1021/jacs.5b01004` *(foundational JCESR Persson/Curtiss Electrolyte Genome work alongside Cheng, Rajput, Curtiss & Persson, Chem. Mater. 2014)*.
- **Core Problem & Method:** Combines classical MD solvation sampling across glymes/ethers with high-throughput DFT reduction potential calculations on extracted explicit cation–anion ion pairs (`SSIP` vs `CIP`).
- **Key Metrics / Formulas:** Computes thermodynamic reduction potential $E_{\text{red}} = -\frac{\Delta G_{\text{red}}}{nF} - E_{\text{ref}}$ across explicit solvation motifs; proves that direct cation–anion contact (`CIP`) lowers the LUMO and weakens intramolecular $\text{S–C}$ / $\text{S–N}$ bonds in paired $\text{TFSI}^-$, shifting reduction onset positively by $>0.5\text{ V}$ relative to isolated anions.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow should export representative 3D cluster geometries (and motif weights $\chi_{\text{SSIP}}, \chi_{\text{CIP}}, \chi_{\text{AGG}}$) directly from trajectories so downstream electronic-structure / redox predictors evaluate realistic ion-paired complexes rather than unphysical bare molecules.

#### 3. Direct Correlation between Ionic Liquid Transport Properties and Ion Pair Lifetimes: A Molecular Dynamics Study
- **Citation:** Yong Zhang and Edward J. Maginn, *The Journal of Physical Chemistry Letters* **6**(4), 700–705 (2015). DOI: `10.1021/acs.jpclett.5b00003`.
- **Core Problem & Method:** All-atom MD trajectories across 29 ionic liquids over multiple temperatures to establish a universal microscopic dynamical predictor of macroscopic diffusivity, viscosity, and conductivity.
- **Key Metrics / Formulas:** Defines two distinct population correlation functions using step indicator $h_{ij}(t) \in \{0,1\}$ within first-minimum RDF cutoff $r_c$: the **ion-pair (IP)** correlation (single specific counterion $j$) and **ion-cage (IC)** correlation (complete surrounding counterion cage), fitted to stretched exponentials $C(t) = \sum_k a_k \exp[-(t/\tau_k)^{\beta_k}]$; proves universal power-law scaling $D_\pm \propto \tau_{\text{IC}}^{-\gamma}$ ($\gamma \approx 0.9\text{–}1.0$, $R^2 > 0.96$).
- **Concrete Design Takeaway for WignerFlow:** Implement vectorized FFT-based intermittent autocorrelation $C_{\text{cage}}(t) = \langle h_{ij}(0)h_{ij}(t)\rangle / \langle h_{ij}\rangle$ and KWW stretched-exponential integration $\tau_{\text{ex}} = \frac{\tau_0}{\beta}\Gamma(1/\beta)$ in WignerFlow's solvation module as a fast, low-variance proxy for transport sluggishness even on nanosecond-scale MLIP rollouts.

#### 4. Modeling Insight into Battery Electrolyte Electrochemical Stability and Interfacial Structure
- **Citation:** Oleg Borodin, Xiaoming Ren, Jenel Vatamanu, Arthur von Wald Cresce, Jaroslaw Knap, and Kang Xu, *Accounts of Chemical Research* **50**(12), 2886–2894 (2017). DOI: `10.1021/acs.accounts.7b00486` *(paired with Seo, Borodin, Han, Ly & Henderson, J. Electrochem. Soc. 159, A1489, 2012 & 167, 167111, 2020)*.
- **Core Problem & Method:** Synthesizes polarizable MD (`APPLE&P`), DFT redox clusters, and Raman/X-ray crystallographic benchmarks to map how bulk `SSIP`/`CIP`/`AGG` speciation reshapes under electric double-layer polarization at anode and cathode surfaces.
- **Key Metrics / Formulas:** Classifies anion coordination by bridging count $n_c(a) \in \{0\text{ (SSIP/free)}, 1\text{ (CIP)}, 2\text{ (AGG-I)}, \ge 3\text{ (AGG-II)}\}$ using $\text{Li–O}/\text{Li–F}/\text{Li–N}$ cutoffs $r_c \approx 2.6\text{–}3.0\text{ Å}$; shows `AGG` formation shifts anion reduction potentials above carbonate reduction, switching SEI from organic alkoxides to inorganic $\text{LiF}/\text{Li}_2\text{O}$.
- **Concrete Design Takeaway for WignerFlow:** Report both cation-centric (`SSIP`/`CIP`/`AGG`) and anion-centric (`Free`/`CIP`/`AGG-I`/`AGG-II`) histograms in WignerFlow CLI outputs so computational predictions map 1-to-1 onto experimental Raman S–N–S / P–F band deconvolutions.

#### 5. Advances and Issues in Developing Salt-Concentrated Battery Electrolytes
- **Citation:** Yuki Yamada, Jianhui Wang, Seitaro Ko, Eriko Watanabe, and Atsuo Yamada, *Nature Energy* **4**(4), 269–280 (2019). DOI: `10.1038/s41560-019-0336-z`.
- **Core Problem & Method:** Authoritative mechanistic analysis of superconcentrated electrolytes (>3–5 M, solvent-to-salt molar ratio $N_{\text{solv}}/N_{\text{Li}} \lesssim 2$) where uncoordinated ("free") solvent vanishes and 3D `AGG` networks dominate.
- **Key Metrics / Formulas:** Tracks fraction of uncoordinated solvent $f_{\text{free}} = 1 - \frac{N_{\text{Li}} \, CN_{\text{Li,solv}}}{N_{\text{solv}} \, \nu_{\text{dent}}}$ ($\to 0$ in superconcentrated regimes), showing LUMO shifts from solvent to anion as $\chi_{\text{CIP}} + \chi_{\text{AGG}} \to 90\text{–}100\%$, at the cost of $5\text{–}20\times$ higher viscosity $\eta$ and poor separator wetting.
- **Concrete Design Takeaway for WignerFlow:** Include **free-solvent fraction ($f_{\text{free}}$)** alongside $\chi_{\text{SSIP}}/\chi_{\text{CIP}}/\chi_{\text{AGG}}$ as a first-class metric in WignerFlow: $f_{\text{free}} < 0.15$ flags high anodic oxidative stability (>4.4 V), whereas $\chi_{\text{AGG}} > 0.50$ without diluent flags severe viscosity penalty.

#### 6. Monolithic Solid–Electrolyte Interphases Formed in Fluorinated Orthoformate-Based Electrolytes Minimize Li Depletion and Pulverization
- **Citation:** Xia Cao, Xiaodi Ren, Lianfeng Zou, Mark H. Engelhard, William Huang, Haiping Wang, Bethany E. Matthews, Hongkyung Lee, Chongmin Wang, Bruce W. Arey, Yi Cui, Wu Xu, and Ji-Guang Zhang, *Nature Energy* **4**(9), 796–805 (2019). DOI: `10.1038/s41560-019-0464-5` *(with Ren, Zou, Cao, Xu & Zhang, Joule 3, 1662–1676, 2019)*.
- **Core Problem & Method:** Pioneers Localized High-Concentration Electrolytes (`LHCE`, e.g., LiFSI–DME diluted with non-coordinating fluorinated orthoformate TFEO or TTE) analyzed via AIMD/MD RDFs, cryo-TEM, and XPS.
- **Key Metrics / Formulas:** Demonstrates bimodal spatial separation in partial RDFs: solvating solvent DME + $\text{FSI}^-$ populate $r < 2.8\text{ Å}$ ($CN_{\text{Li,O(FSI)}} \approx 1.8\text{–}2.5$, preserving local `CIP`/`AGG` clusters), while hydrofluoroether/orthoformate diluent shows $g_{\text{Li,O(diluent)}}(r) \approx 0$ for $r < 3.5\text{ Å}$ ($CN_{\text{diluent}}(2.8\text{ Å}) < 0.05$), cutting viscosity from $>25\text{ cP}$ to $<4\text{ cP}$ and achieving $>99.5\%$ Li Coulombic efficiency via monolithic $\text{LiF}$-rich SEI.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's multi-component analyzer must automatically classify components by $CN_{\text{Li},s}(r_c)$ into **primary solvators** ($CN > 0.2$), **competitive additives**, and **true non-coordinating diluents** ($CN < 0.05$ inside $r_c = 2.8\text{ Å}$ but nonzero in the $4.5\text{–}7.0\text{ Å}$ secondary shell), enabling automated `LHCE` detection.

#### 7. The Influence of FEC on the Solvation Structure and Reduction Reaction of $\text{LiPF}_6$/EC Electrolytes and Its Implication for Solid Electrolyte Interphase Formation
- **Citation:** Tingzheng Hou, Guang Yang, Nav Nidhi Rajput, Julian Self, Seung-Wan Park, Jagjit Nanda, and Kristin A. Persson, *Nano Energy* **64**, 103881 (2019). DOI: `10.1016/j.nanoen.2019.103881`.
- **Core Problem & Method:** Classical MD + quantum chemical free-energy cluster sampling of $1.0\text{ M }\text{LiPF}_6$ in EC with $10\%$ fluoroethylene carbonate (FEC) additive to resolve how a minority fluorinated additive restructures $\text{Li}^+$ speciation and SEI onset.
- **Key Metrics / Formulas:** Using first-shell cutoffs ($r_c(\text{Li–O}_{\text{carb}}) \approx 2.8\text{ Å}$, $r_c(\text{Li–P}) \approx 4.1\text{ Å}$ / $r_c(\text{Li–F}) \approx 2.7\text{ Å}$), shows that $10\%$ FEC participates in $\sim 20\%$ of $\text{Li}^+$ primary solvation sheaths and weakens average shell binding enough to **more than double the `CIP` fraction from $\sim 6\%$ (pure EC) to $\sim 14\%$**, while coordinated $\text{Li}^+\text{–FEC}$ reduces at $+0.3\text{ V}$ higher potential than EC to generate $\text{LiF}$.
- **Concrete Design Takeaway for WignerFlow:** Track **joint shell-motif distributions** $P(n_{\text{solv}}, n_{\text{add}}, n_{\text{anion}})$ rather than marginal averages alone: WignerFlow must quantify how additive coordination shifts $\chi_{\text{CIP}}$ via allosteric inductive weakening of the primary $\text{Li}^+$ shell.

#### 8. Transport in Superconcentrated $\text{LiPF}_6$ and $\text{LiBF}_4$/Propylene Carbonate Electrolytes
- **Citation:** Julian Self, Kara D. Fong, and Kristin A. Persson, *ACS Energy Letters* **4**(12), 2843–2849 (2019). DOI: `10.1021/acsenergylett.9b02118`.
- **Core Problem & Method:** All-atom MD simulations across $1\text{–}3\text{ M}$ $\text{LiPF}_6/\text{PC}$ and $\text{LiBF}_4/\text{PC}$ dissecting the mechanistic crossover between **vehicular** (intact shell co-diffusion) and **structural** (ligand-exchange hopping) $\text{Li}^+$ transport.
- **Key Metrics / Formulas:** Computes ligand exchange residence times $\tau_{\text{ex}}$ from $C(t) = \langle h(0)h(t)\rangle$ and the characteristic diffusion distance prior to shell exchange $L_{\text{ex}} = \sqrt{6 D_{\text{Li}} \tau_{\text{ex}}}$; reveals anion divergence at high concentration (`CIP`/`AGG`): $\text{PF}_6^-$ shifts $\text{Li}^+$ toward vehicular motion, whereas strongly associating $\text{BF}_4^-$ promotes structural hopping across anion bridges.
- **Concrete Design Takeaway for WignerFlow:** Compute the dimensionless **vehicularity index** $\Lambda_{\text{veh}} = \sqrt{6 D_{\text{Li}} \tau_{\text{ex}}} / r_c$ for both solvent and anion shells in WignerFlow ($\Lambda_{\text{veh}} \gg 1$: vehicular; $\Lambda_{\text{veh}} \lesssim 1$: structural/hopping exchange).

#### 9. Correlations from Ion Pairing and the Nernst–Einstein Equation
- **Citation:** Arthur France-Lanord and Jeffrey C. Grossman, *Physical Review Letters* **122**(13), 136001 (2019). DOI: `10.1103/PhysRevLett.122.136001`.
- **Core Problem & Method:** Derives the **cluster Nernst–Einstein (`cNE`)** theory to rescue transport prediction from the noisy $O(1)$ variance of collective Green–Kubo current integrals while correcting the massive overestimation of standard `NE` in ion-paired electrolytes.
- **Key Metrics / Formulas:** Partitions ions into connected coordination clusters $(i\text{ cations}, j\text{ anions})$ with population $\langle \alpha_{ij} \rangle$, net charge $z_{ij} = i z_+ + j z_-$, and cluster diffusivity $D_{ij}$:
  $$\sigma_{\text{cNE}} = \frac{e^2}{V k_B T} \sum_{i,j} z_{ij}^2 \langle \alpha_{ij} \rangle D_{ij}, \qquad t_+^{\text{cNE}} = \frac{\sum_{i,j} i\, z_{ij} \langle \alpha_{ij} \rangle D_{ij}}{\sum_{i,j} z_{ij}^2 \langle \alpha_{ij} \rangle D_{ij}}$$
  Neutral `CIP`s ($i=1, j=1 \implies z_{11}=0$) contribute **zero** to $\sigma_{\text{cNE}}$, and negatively charged `AGG`s ($i=1, j=2 \implies z_{12}=-1$) naturally explain **negative transference numbers** ($t_+^{\text{cNE}} < 0$).
- **Concrete Design Takeaway for WignerFlow:** Implement `cNE` directly atop the `SSIP`/`CIP`/`AGG` connected-component graph! Because `cNE` uses single-cluster MSDs ($1/\sqrt{N_{\text{clusters}}}$ statistical convergence vs non-self-averaging Green–Kubo), WignerFlow can accurately estimate $\sigma_{\text{cNE}}$, ionicity $\sigma_{\text{cNE}}/\sigma_{\text{NE}}$, and $t_+^{\text{cNE}}$ from short MD/WignerFlow rollouts.

#### 10. Uncharted Waters: Super-Concentrated Electrolytes
- **Citation:** Oleg Borodin, Julian Self, Kristin A. Persson, Chunsheng Wang, and Kang Xu, *Joule* **4**(1), 69–100 (2020). DOI: `10.1016/j.joule.2019.12.007`.
- **Core Problem & Method:** Definitive structural-dynamical taxonomy bridging dilute ("salt-in-solvent"), concentrated, "water-in-salt" (`WiSE`), and localized high-concentration electrolytes via MD RDFs, cluster size distributions, and exchange kinetics.
- **Key Metrics / Formulas:** Formalizes percolating `AGG` cluster thresholds (largest cluster fraction $P_\infty = N_{\max}/N_{\text{ion}} > 0.5$ signaling gel-like 3D ionic network percolation) and relates decoupling of $\text{Li}^+$ conductivity from structural relaxation time ($\tau_\alpha / \tau_{\text{Li}} \gg 1$) to asymmetrical anion–solvent nanodomains.
- **Concrete Design Takeaway for WignerFlow:** Add **percolation order parameter** $P_\infty = \max_k |V_k| / (N_+ + N_-)$ over the ion-pair graph: WignerFlow should penalize candidate fast-charging electrolytes when $P_\infty > 0.35$, as percolated `AGG` networks choke high-rate mass transport.

#### 11. Onsager Transport Coefficients and Transference Numbers in Polyelectrolyte Solutions and Polymerized Ionic Liquids
- **Citation:** Kara D. Fong, Julian Self, Bryan D. McCloskey, and Kristin A. Persson, *Macromolecules* **53**(21), 9503–9512 (2020). DOI: `10.1021/acs.macromol.0c02001` *(with Fong, Self, McCloskey & Persson, Macromolecules 54, 2575–2591, 2021 on liquid/polymer ion correlations)*.
- **Core Problem & Method:** Uses linear irreversible thermodynamics and MD displacement cross-correlations to decompose total ionic transport into ideal self-terms and distinct cation–cation, anion–anion, and cation–anion (`CIP`/`AGG`) Onsager transport coefficients $L^{ij}$.
- **Key Metrics / Formulas:**
  $$L^{ij} = \frac{1}{6 V k_B T} \lim_{t\to\infty} \frac{d}{dt}\left\langle \sum_{\alpha \in i}\Delta\mathbf{r}_\alpha(t) \cdot \sum_{\beta \in j}\Delta\mathbf{r}_\beta(t) \right\rangle, \qquad \sigma = F^2\left(z_+^2 L^{++} + 2 z_+ z_- L^{+-} + z_-^2 L^{--}\right)$$
  Shows that when `CIP`/`AGG` lifetime $\tau_{\text{IP}}$ exceeds cage escape time, positive $L^{+-} > 0$ (since $z_+ z_- < 0$) severely suppresses $\sigma$ and drives true transference number $t_+ = \frac{z_+^2 L^{++} + z_+ z_- L^{+-}}{\sigma / F^2}$ far below $t_+^{\text{NE}}$.
- **Concrete Design Takeaway for WignerFlow:** Expose optional Onsager $L^{ij}$ cross-correlation diagnostics alongside `cNE` when multi-nanosecond trajectories are available, cross-validating `cNE` cluster cutoffs against cutoff-free $L^{+-}$ anti-conductivity.

#### 12. Molecular Design for Electrolyte Solvents Enabling Energy-Dense and Long-Cycling Lithium Metal Batteries
- **Citation:** Zhiao Yu, Hansen Wang, Xian Kong, William Huang, Yuchi Tsao, David G. Mackanic, Kecheng Wang, Xinchang Wang, Wenxiao Huang, Snehashis Choudhury, Yu Zheng, Chibueze V. Amanchukwu, Samantha T. Hung, Yuting Ma, Eder G. Lomeli, Jian Qin, Yi Cui, and Zhenan Bao, *Nature Energy* **5**(7), 526–533 (2020). DOI: `10.1038/s41560-020-0634-5`.
- **Core Problem & Method:** Synthesizes fluorinated 1,4-dimethoxylbutane (`FDMB`) to achieve `CIP`/`AGG`-rich solvation in a **single-solvent 1.0 M LiFSI** formulation without high salt concentration or cosolvent diluents, characterized via MD RDFs, Raman, and $^7\text{Li}/^{19}\text{F}$ NMR.
- **Key Metrics / Formulas:** MD radial analysis reveals weak $\text{Li–F}_{\text{FDMB}}$ secondary interaction ($r \approx 2.6\text{–}3.0\text{ Å}$) alongside $\text{Li–O}_{\text{FDMB}}$ ($r_c \approx 2.8\text{ Å}$) and elevated anion coordination ($CN_{\text{Li,O(FSI)}} > 2.0$, dominated by `CIP`/`AGG`), yielding $99.52\%$ Li CE and $>4.2\text{ V}$ oxidation stability, though moderate conductivity ($\sim 0.35\text{ mS/cm}$) highlights the excess-`AGG` transport trade-off.
- **Concrete Design Takeaway for WignerFlow:** Explicitly include **organofluorine $\text{Li–F}_{\text{solv}}$** in partial RDF and running $CN(r)$ channels ($r_c = 2.9\text{ Å}$), since $-\text{CF}_2-$ / $-\text{CHF}_2$ groups in fluorinated ethers actively participate in weak secondary $\text{Li}^+$ coordination.

#### 13. Regulating Interfacial Chemistry in Lithium-Ion Batteries by a Weakly Solvating Electrolyte
- **Citation:** Yu-Xing Yao, Xiang Chen, Chong Yan, Xue-Qiang Zhang, Wen-Long Cai, Jia-Qi Huang, and Qiang Zhang, *Angewandte Chemie International Edition* **60**(8), 4090–4097 (2021). DOI: `10.1002/anie.202011482`.
- **Core Problem & Method:** Formalizes the **Weakly Solvating Electrolyte (`WSE`)** paradigm ($1.0\text{ M LiFSI}$ in low-permittivity 1,4-dioxane, $\varepsilon_r = 2.2$) to suppress solvent co-intercalation into graphite and enable fast charging via pure solvation-power tuning.
- **Key Metrics / Formulas:** Deconvolutes MD coordination shells ($r_c = 2.8\text{ Å}$) and Raman S–N–S modes ($717\text{ cm}^{-1}$ `SSIP`, $730\text{ cm}^{-1}$ `CIP`, $745\text{ cm}^{-1}$ `AGG`), showing `SSIP` drops from $>85\%$ (in DME) to $<15\%$ in `WSE`, while `CIP` ($\sim 40\%$) + `AGG` ($\sim 45\%$) dominate at $1.0\text{ M}$, lowering $\text{Li}^+$ desolvation energy barrier to $<45\text{ kJ/mol}$.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow can predict experimental Raman S–N–S speciation areas directly from MD `SSIP`/`CIP`/`AGG` fractions, providing an immediate zero-parameter experimental validation hook for newly screened `WSE` solvents.

#### 14. Potentiometric Measurement to Probe Solvation Energy and Its Correlation to Lithium Battery Cyclability
- **Citation:** Sang Cheol Kim, Xinyi Kong, Rafael A. Vilá, William Huang, Yuchi Tsao, David G. Mackanic, Jian Qin, Yi Cui, and Zhenan Bao, *Journal of the American Chemical Society* **143**(27), 10301–10308 (2021). DOI: `10.1021/jacs.1c03868`.
- **Core Problem & Method:** Measures relative $\text{Li}^+$ solvation free energy $\Delta G_{\text{solv}}$ potentiometrically across 15+ ether/carbonate/fluorinated electrolytes and correlates it directly with MD `CIP`/`AGG` populations and Li metal Coulombic efficiency.
- **Key Metrics / Formulas:** Establishes Nernstian cell EMF relation $E_{\text{cell}} = -\frac{\Delta G_{\text{solv}}^{\text{test}} - \Delta G_{\text{solv}}^{\text{ref}}}{F}$; demonstrates monotonic correlation where more positive $\Delta G_{\text{solv}}$ (weaker solvation, $-50$ to $+80\text{ mV}$ shift vs $1\text{ M LiFSI/DEC}$) increases anion coordination number $ACN$ from $<0.3$ to $>1.8$ and raises Li CE from $<90\%$ to $>99\%$.
- **Concrete Design Takeaway for WignerFlow:** Combine WignerFlow's structural $ACN = CN_{\text{Li,anion}}(r_c)$ and $\chi_{\text{CIP}}+\chi_{\text{AGG}}$ metrics into an empirical proxy for $\Delta G_{\text{solv}}$, allowing WignerFlow to rank anode SEI passivating capability directly from pure-MD coordinates.

#### 15. Moving Beyond 99.9% Coulombic Efficiency for Lithium Anodes in Liquid Electrolytes
- **Citation:** Gustavo M. Hobold, Jeffrey Lopez, Rui Guo, Nicolò Minafra, Abhik Banerjee, Y. Shirley Meng, Yang Shao-Horn, and Betar M. Gallant, *Nature Energy* **6**(10), 951–960 (2021). DOI: `10.1038/s41560-021-00910-w` *(with Leverick, Shao-Horn et al. on donor-number / solvation enthalpy $\Delta H_{\text{solv}}$ vs entropy $\Delta S_{\text{solv}}$ scaling)*.
- **Core Problem & Method:** Quantitative thermodynamic and kinetic meta-analysis across liquid electrolytes linking $\text{Li}^+$ solvation enthalpy $\Delta H_{\text{solv}}$, Gutmann donor number ($DN$), exchange current density $j_0$, and SEI $\text{LiF}/\text{Li}_2\text{O}$ nanostructure to $>99.9\%$ CE.
- **Key Metrics / Formulas:** Shows $\Delta G_{\text{solv}} = \Delta H_{\text{solv}} - T\Delta S_{\text{solv}}$ is enthalpy-dominated and scales linearly with effective shell donor number; weaker solvation ($DN \lesssim 14\text{ kcal/mol}$ or high `CIP` activity) lowers charge-transfer reorganization barrier and promotes compact $\text{Li}_2\text{O}/\text{LiF}$ SEI with fast interfacial $\text{Li}^+$ exchange.
- **Concrete Design Takeaway for WignerFlow:** Pair structural speciation (`CIP`/`AGG`) with dynamical exchange rate $k_{\text{ex}} = 1/\tau_{\text{ex}}$ in WignerFlow's scoring function: ultra-high CE + fast plating requires **both** anion presence in the primary shell ($\chi_{\text{CIP}} \ge 0.15$) **and** rapid ligand turnover ($\tau_{\text{ex}} \lesssim 300\text{ ps}$).

#### 16. Rational Solvent Molecule Tuning for High-Performance Lithium Metal Battery Electrolytes
- **Citation:** Zhiao Yu, Paul E. Rudnicki, Zewen Zhang, Ziying Huang, Hasan Celik, Shannon T. Oyakhire, Yuelang Chen, Xinyi Kong, Sang Cheol Kim, Xin Xiao, Hansen Wang, Yu Zheng, Gaurav A. Kamat, Mun Sek Kim, Stacey F. Bent, Jian Qin, Yi Cui, and Zhenan Bao, *Nature Energy* **7**(1), 94–106 (2022). DOI: `10.1038/s41560-021-00962-y`.
- **Core Problem & Method:** Resolves the conductivity–SEI trade-off of `FDMB` by asymmetrically tuning $-\text{CHF}_2$ vs $-\text{CF}_3$ terminal groups on 1,2-diethoxyethane (`F3DEE`, `F4DEE`, `F5DEE`), combining MD solvation trajectories, pulsed-field-gradient NMR, and fast-charging pouch cells.
- **Key Metrics / Formulas:** Identifies the **"Goldilocks" solvation regime** in $1.2\text{ M LiFSI/F4DEE}$ and `F5DEE`: partial chelation yields balanced `SSIP` ($\sim 35\text{–}55\%$), `CIP` ($\sim 30\text{–}45\%$), and controlled `AGG` ($\sim 10\text{–}25\%$), simultaneously achieving bulk ionic conductivity $\sigma \approx 5.0\text{ mS/cm}$ ($10\times$ higher than `FDMB`), fast $\text{Li}^+$ exchange, and $99.5\%$ CE under high current density.
- **Concrete Design Takeaway for WignerFlow:** Directly encodes WignerFlow's quantitative **"Ideal Fast-Charging / Stable-SEI Window"**: target $\chi_{\text{CIP}} \in [0.15, 0.45]$ and $\chi_{\text{AGG}} \le 0.25$ so anions populate the inner Helmholtz plane without collapsing $\sigma_{\text{cNE}}$.

#### 17. MISPR: An Open-Source Package for High-Throughput Multiscale Molecular Simulations (with HiPRGen SEI Network Coupling)
- **Citation:** Rasha Atwi, Matthew Bliss, Maxim Makeev, and Nav Nidhi Rajput, *Scientific Reports* **12**, 15760 (2022). DOI: `10.1038/s41598-022-20009-w` *(paired with Evan Walter Clark Spotte-Smith, Ronald L. Kam, Daniel Barter, Xiaowei Xie, Tingzheng Hou, Shyam Dwaraknath, Samuel M. Blau, and Kristin A. Persson, `HiPRGen`, ACS Energy Letters 7(4), 1446–1453, 2022, DOI: `10.1021/acsenergylett.2c00517`)*.
- **Core Problem & Method:** Automates high-throughput MD/DFT workflows (`MISPR`) extracting partial RDFs, running $CN(r)$, `SSIP`/`CIP`/`AGG` speciation, diffusivity, and residence times $\tau_{\text{res}}$, whose dominant solvation clusters seed automated Monte Carlo electrochemical reaction networks (`HiPRGen`) of SEI formation.
- **Key Metrics / Formulas:** Automatically detects $r_c$ as the first minimum of $g_{\text{Li},\beta}(r)$ in $[2.0, 3.5]\text{ Å}$ via spline derivative roots ($g'(r_c)=0, g''(r_c)>0$), classifies every $\text{Li}^+(t)$ frame-by-frame into `SSIP`/`CIP`/`AGG`, and fits exponential decay to $C(t) = \langle h(0)h(t)\rangle$ for ligand-specific $\tau_{\text{res}}$.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's solvation module should implement **automated first-minimum $r_c$ detection** (Savitzky–Golay smoothed $g(r)$ minimum in $[2.4, 3.3]\text{ Å}$ with fallback defaults $r_c(\text{O})=2.80\text{ Å}, r_c(\text{F})=2.75\text{ Å}, r_c(\text{N})=3.00\text{ Å}$) so users do not have to hand-tune cutoffs per chemistry.

#### 18. Electrolyte Design for Li-Ion Batteries under Extreme Operating Conditions
- **Citation:** Jijian Xu, Jiaxun Zhang, Travis P. Pollard, Qingdong Li, Sha Tan, Singyuk Hou, Hongli Wan, Fu Chen, Huixin He, Enyuan Hu, Kang Xu, Xiao-Qing Yang, Oleg Borodin, and Chunsheng Wang, *Nature* **614**, 694–700 (2023). DOI: `10.1038/s41586-022-05627-8`.
- **Core Problem & Method:** Establishes the **"soft solvation"** design principle (e.g., fluoroacetonitrile / weak-binding low-viscosity nitriles & esters analyzed via MD + DFT binding energy descriptors) enabling simultaneous fast charging, $4.5\text{ V}$ NMC811 stability, and $-60\,^\circ\text{C}$ operation without Li plating.
- **Key Metrics / Formulas:** Maps electrolytes in the 2D descriptor plane of $(\text{Li}^+\text{–solvent binding energy } E_b,\; \text{bulk dielectric/fluidity})$: shows optimal fast-charging electrolytes maintain **moderate `CIP` ($\sim 25\text{–}40\%$)** and small finite `AGG` dimers (`AGG-I`) with rapid exchange $\tau_{\text{ex}} < 100\text{ ps}$, delivering $>40\text{ mS/cm}$ at $25\,^\circ\text{C}$ and $>10\text{ mS/cm}$ at $-40\,^\circ\text{C}$.
- **Concrete Design Takeaway for WignerFlow:** Distinguish small dimeric aggregates (`AGG-I`, cluster size $\le 4$ ions) from extended polymeric aggregates (`AGG-net`, cluster size $\ge 5$ ions) in WignerFlow: finite `AGG-I` pairs aid $\text{LiF}$ SEI formation and structural hopping, whereas `AGG-net` freezes low-temperature kinetics.

#### 19. Machine Learning Force Fields for Molecular Liquids: Ethylene Carbonate/Ethyl Methyl Carbonate Binary Solvent
- **Citation:** Ioan-Bogdan Magdău, David J. Arismendi-Arrieta, Helen E. Smith, Clare P. Grey, Kersti Hermansson, and Gábor Csányi, *npj Computational Materials* **9**, 146 (2023). DOI: `10.1038/s41524-023-01100-w`.
- **Core Problem & Method:** Develops and rigorously validates machine-learned interatomic potentials (`GAP` / local many-body descriptors) for condensed-phase battery carbonate mixtures (`EC:EMC`), diagnosing why static force RMSE fails when intramolecular covalent forces ($>1\text{ eV/Å}$) dwarf intermolecular van der Waals packing forces ($\sim 0.02\text{ eV/Å}$).
- **Key Metrics / Formulas:** Introduces **intra- vs. intermolecular force decomposition** ($\mathbf{F}_i = \mathbf{F}_i^{\text{intra}} + \mathbf{F}_i^{\text{inter}}$, torque/center-of-mass molecular force RMSE) and validates NPT liquid density $\rho$, intermolecular O–O / C–C RDFs, and molecular self-diffusivities $D_{\text{EC}}, D_{\text{EMC}}$.
- **Concrete Design Takeaway for WignerFlow:** When rolling out molecular liquid electrolytes in WignerFlow, always validate **center-of-mass intermolecular RDFs** and NPT/NVT packing stability separately from stiff intramolecular bond vibrations; errors in weak intermolecular dispersion distort $CN_{\text{Li,solv}}(r_c)$ even when total force MAE looks small.

#### 20. A Predictive Machine Learning Force-Field Framework for Liquid Electrolyte Development (`BAMBOO`)
- **Citation:** Sheng Gong et al. (ByteDance Research), *Nature Machine Intelligence* **7**, 543–555 (2025; arXiv:2404.07181). DOI: `10.1038/s42256-025-01009-7` *(complemented by Y. Wu, H. Xu, L. Wang, X.-M. He, Adv. Energy Mater. 2023 ESP solvation descriptors & J. Zhang et al., Angew. Chem. Int. Ed. 2025 Hofmeister anion selectivity)*.
- **Core Problem & Method:** Introduces `BAMBOO` (**B**yteDance **A**I **M**olecular Simulation **Boo**ster), an equivariant graph transformer MLIP with ensemble knowledge distillation and experimental density alignment spanning $>15$ chemical elements for multi-component lithium battery salts, carbonates, ethers, and fluorinated additives.
- **Key Metrics / Formulas:** Achieves liquid density MAE of $0.01\text{ g/cm}^3$, reproduces DFT/experimental partial $\text{Li}^+$ RDFs, `SSIP`/`CIP`/`AGG` fractions, Green–Kubo shear viscosity $\eta = \frac{V}{k_B T}\int_0^\infty \langle P_{\alpha\beta}(0)P_{\alpha\beta}(t)\rangle\,dt$, and ionic conductivity across fast-charging multi-additive formulations.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's solvation and CLI modules should support seamless plug-in of universal/electrolyte MLIP calculators (`MACE-MP`, `BAMBOO`-style equivariant checkpoints) to run end-to-end **formulation $\to$ rollout $\to$ (`SSIP`/`CIP`/`AGG`, $\alpha_{\text{enr}}$, $\tau_{\text{ex}}$, $\sigma_{\text{cNE}}$) fast-charging screening** from a single CLI command.

---

### Summary Table: Working Target Ranges Used in WignerFlow's Solvation Code

> These ranges are our own rough heuristics, pieced together from the papers above. They are not values reported by any single paper, and we have not validated them against experiments.

| Solvation / Kinetic Metric | Mathematical Definition | Ideal Fast-Charging + Stable-SEI Target Window | Failure Mode Outside Window |
| :--- | :--- | :--- | :--- |
| **`CIP` Fraction ($\chi_{\text{CIP}}$)** | $\frac{1}{N_{\text{Li}}}\sum_i \mathbb{I}[|V_+(i)|=1 \land |V_-(i)|=1]$ | **$15\% \text{ – } 45\%$** | $<10\%$: organic solvent-derived SEI, Li dendrites / graphite exfoliation |
| **`AGG` Fraction ($\chi_{\text{AGG}}$)** | $\frac{1}{N_{\text{Li}}}\sum_i \mathbb{I}[|V_+(i)|+|V_-(i)| \ge 3]$ | **$\le 20\%\text{–}25\%$** *(single-solvent/WSE)*; local $30\text{–}50\%$ OK **only** in `LHCE` ($CN_{\text{dil}} < 0.05$) | $>30\%$ without diluent: 3D `AGG` percolation ($P_\infty > 0.35$), high $\eta$, low $t_+^{\text{cNE}}$ |
| **`SSIP` Fraction ($\chi_{\text{SSIP}}$)** | $\frac{1}{N_{\text{Li}}}\sum_i \mathbb{I}[n_a(i)=0]$ | **$35\% \text{ – } 75\%$** | $<25\%$: severe loss of free charge carriers ($\sigma_{\text{cNE}}/\sigma_{\text{NE}} < 0.35$) |
| **Anion Coord. Number ($ACN$)** | $4\pi\rho_{\text{anion}}\int_0^{r_c} r^2 g_{\text{Li,anion}}(r)\,dr$ | **$0.20 \text{ – } 0.85$** *(out of $CN_{\text{tot}} \approx 4.0\text{–}4.5$)* | $>1.5$: sluggish desolvation/cluster drag unless diluted (`LHCE`) |
| **Additive Enrichment ($\alpha_{\text{enr}}$)** | $(CN_{\text{Li},A}/CN_{\text{Li,solv}}) / (N_A/N_{\text{solv}})$ | **$1.2 \text{ – } 3.0$** for SEI formers (e.g. FEC, $\text{NO}_3^-$) | $\ll 0.2$: non-coordinating diluent; $>5$: traps $\text{Li}^+$ in deep solvation well |
| **Cage Exchange Time ($\tau_{\text{ex}}$)** | $\int_0^\infty \frac{\langle h_{ij}(0)h_{ij}(t)\rangle}{\langle h_{ij}(0)\rangle}\,dt$ | **$20 \text{ – } 300\text{ ps}$** at $298\text{ K}$ | $>1\text{ ns}$: rigid vehicular shell, high interfacial desolvation barrier ($>60\text{ kJ/mol}$) |
| **`cNE` Ionicity ($\mathcal{I}_{\text{cNE}}$)** | $\sigma_{\text{cNE}} / \sigma_{\text{NE}}$ | **$0.50 \text{ – } 0.85$** with $t_+^{\text{cNE}} \ge 0.35$ | $<0.30$ or $t_+^{\text{cNE}} < 0.15$: concentration polarization under $>4\text{C}$ charging |

---

## Part II: Pure-MD Trajectory Rollout Benchmarks (`MD17`/`rMD17`/`MD22`/`Fu et al.`) & Fast Scientific MD CLI Tooling (Papers 21–40)

### Canonical Benchmark Equations from `Fu et al. (ICLR / TMLR 2023)` & Dynamic Observable Suite

1. **Bond-Length Trajectory Stability Horizon ($T_{\text{stab}}$):**
   Let $\mathcal{B}$ be the set of covalent bonded atom pairs with reference equilibrium bond lengths $b_{ij}^0$. A rollout of length $T_{\max}$ becomes **unstable** at the earliest frame $T_{\text{stab}} \le T_{\max}$ satisfying:
   $$\max_{(i,j) \in \mathcal{B}} \left| \|\mathbf{r}_i(t) - \mathbf{r}_j(t)\|_{\text{PBC}} - b_{ij}^0 \right| > \Delta_{\text{stab}}, \qquad \Delta_{\text{stab}} = 0.5\text{ Å}$$
   *(For flexible polypeptides like `Ac-Ala3-NHMe` or periodic liquids, covalent bond rupture $>0.5\text{ Å}$ and non-bonded steric clash $\min_{i\neq j} r_{ij}(t) < r_{\text{clash}} \approx 0.7\text{ Å}$ are monitored jointly.)*

2. **All-Pair Interatomic Distance Distribution $h(r)$ MAE (Isolated Molecules: `MD17` / `rMD17` / `MD22`):**
   Because single molecules in vacuum have no bulk density $\rho$, structural fidelity over a rollout of $K$ frames is evaluated via the normalized pairwise distance density $h(r)$ ($\int_0^\infty h(r)\,dr = 1$):
   $$h(r) = \frac{2}{N(N-1)} \frac{1}{K} \sum_{k=1}^K \sum_{1 \le i < j \le N} K_\sigma\!\left(r - \|\mathbf{r}_i(t_k) - \mathbf{r}_j(t_k)\|\right), \qquad \text{MAE}_{h(r)} = \int_0^\infty \left|\hat{h}_{\text{model}}(r) - h_{\text{ref}}(r)\right| dr$$
   where $K_\sigma$ is a Gaussian KDE kernel (typically $\sigma \approx 0.02\text{–}0.05\text{ Å}$) or fine histogram bin $\Delta r$.

3. **Periodic Radial Distribution Function (`RDF`) MAE (Condensed Phase Liquids / Electrolytes):**
   $$\text{MAE}_{\text{RDF}}^{(\alpha,\beta)} = \int_0^{R_{\max}} \left|\hat{g}_{\alpha\beta}(r) - g_{\alpha\beta}^{\text{ref}}(r)\right| dr \quad \text{(or mean } \frac{1}{R_{\max}}\int_0^{R_{\max}} |\hat{g}_{\alpha\beta} - g_{\alpha\beta}^{\text{ref}}|\,dr\text{, } R_{\max}=6.0\text{ Å)}$$

4. **Vibrational Density of States (`VDOS`) & Self-Diffusivity from Velocity Autocorrelation (`VACF`):**
   From mass-weighted velocities $\mathbf{v}_i(t)$, define the normalized `VACF` $C_v(t)$ and its windowed cosine Fourier transform (`VDOS` power spectrum):
   $$C_v(t) = \frac{\left\langle \sum_{i=1}^N m_i \mathbf{v}_i(t_0)\cdot \mathbf{v}_i(t_0+t) \right\rangle_{t_0}}{\left\langle \sum_{i=1}^N m_i \|\mathbf{v}_i(t_0)\|^2 \right\rangle_{t_0}}, \qquad S_{\text{VDOS}}(\omega) = \frac{1}{\pi} \int_0^{t_{\max}} C_v(t)\, w_{\text{Hann}}(t) \cos(\omega t)\,dt$$
   with wavenumber $\tilde{\nu} = \omega / (2\pi c)$ in $\text{cm}^{-1}$ ($0\text{–}4000\text{ cm}^{-1}$), spectral $L_1$/log-spectral error $\text{MAE}_{\text{VDOS}} = \int_0^{\omega_{\text{Nyq}}} |\hat{S}_{\text{VDOS}}(\omega) - S_{\text{VDOS}}^{\text{ref}}(\omega)|\,d\omega$, and Green–Kubo diffusivity $D = \frac{1}{3N}\sum_{i=1}^N \int_0^\infty \langle \mathbf{v}_i(0)\cdot\mathbf{v}_i(t)\rangle\,dt$. *(Crucial Nyquist constraint for coarse-stride models with step $\Delta t$: only modes below $\tilde{\nu}_{\text{Nyq}} = \frac{1}{2 c \Delta t} \approx \frac{16678}{\Delta t[\text{fs}]}\text{ cm}^{-1}$ are unaliased!)*

---

### Annotated Survey of Papers 21–40

#### 21. Machine Learning of Accurate Energy-Conserving Molecular Force Fields (`GDML` / `MD17`)
- **Citation:** Stefan Chmiela, Alexandre Tkatchenko, Huziel E. Sauceda, Igor Poltavsky, Kristof T. Schütt, and Klaus-Robert Müller, *Science Advances* **3**(5), e1603015 (2017). DOI: `10.1126/sciadv.1603015`.
- **Core Problem & Method:** Introduces **Gradient Domain Machine Learning (`GDML`)**—training curl-free Hessian-kernel Gaussian processes directly in the force domain ($\hat{\mathbf{F}}(\mathbf{x}) = -\nabla \hat{E}(\mathbf{x})$)—and releases the canonical **`MD17`** AIMD trajectory dataset (8 small organic molecules: aspirin, benzene, ethanol, malonaldehyde, naphthalene, salicylic acid, toluene, uracil at $500\text{ K}$, $0.5\text{ fs}$ resolution).
- **Key Metrics / Formulas:** Proves non-conservative vector-regression force fields violate $\nabla \times \mathbf{F} = \mathbf{0}$ and cause catastrophic NVE total-energy drift; establishes $1{,}000$-sample training splits with Force MAE $< 1.0\text{ kcal/(mol}\cdot\text{Å)}$ and Energy MAE $< 0.3\text{ kcal/mol}$.
- **Concrete Design Takeaway for WignerFlow:** Use `MD17` (specifically aspirin, ethanol, malonaldehyde, salicylic acid) for continuous time-ordered trajectory rollout evaluation (which preserves consecutive $0.5\text{ fs}$ phase-space momentum $(\mathbf{q}(t), \mathbf{p}(t))$ needed by WignerFlow!).

#### 22. Towards Exact Molecular Dynamics Simulations with Machine-Learned Force Fields (`sGDML`)
- **Citation:** Stefan Chmiela, Huziel E. Sauceda, Klaus-Robert Müller, and Alexandre Tkatchenko, *Nature Communications* **9**, 3887 (2018). DOI: `10.1038/s41467-018-06169-2`.
- **Core Problem & Method:** Embeds fluxional point-group and methyl-rotor permutation symmetries into `GDML` (`sGDML`), training on CCSD(T) trajectories (`MD17-CCSD(T)`) to run quantum-accurate NVE/NVT/path-integral MD rollouts.
- **Key Metrics / Formulas:** Evaluates rollout quality not just via static Force MAE ($0.1\text{–}0.7\text{ kcal/(mol}\cdot\text{Å)}$), but via **free-energy dihedral surfaces** $F(\phi_1, \phi_2) = -k_B T \ln P(\phi_1, \phi_2)$ and **`VACF`-derived `VDOS` anharmonic peak shifts** ($\Delta \tilde{\nu} < 5\text{ cm}^{-1}$).
- **Concrete Design Takeaway for WignerFlow:** Include **2D Ramachandran / dihedral free-energy surface (`FES`) KL/MAE** (e.g., malonaldehyde proton-transfer / aspirin ester–carboxyl torsion angles) in WignerFlow's `MD17` benchmark alongside $h(r)$ MAE to verify barrier-crossing Boltzmann fidelity.

#### 23. On the Role of Gradients for Machine Learning of Molecular Energies and Forces (`rMD17`)
- **Citation:** Anders S. Christensen and O. Anatole von Lilienfeld, *Machine Learning: Science and Technology* **1**(4), 045018 (2020). DOI: `10.1088/2632-2153/abba6f`.
- **Core Problem & Method:** Discovers that original `MD17` trajectories suffer from loose DFT SCF convergence noise ($\sim 0.1\text{–}0.5\text{ kcal/mol}$ energy jitter and non-conservative force mismatches), and recomputes $100{,}000$ frames per molecule at tight PBE/def2-SVP `ORCA` Grid5 convergence to create **revised MD17 (`rMD17`)**.
- **Key Metrics / Formulas:** Quantifies intrinsic label noise floor of original `MD17` vs `rMD17` ($<0.001\text{ kcal/mol}$) and demonstrates $7\times$ sample-efficiency gain from force-gradient supervision, providing standardized 5-split $1{,}000$-frame benchmarks.
- **Concrete Design Takeaway for WignerFlow:** Note the critical distinction for WignerFlow's data loader: `rMD17` provides ultra-clean static $(\mathbf{r}, E, \mathbf{F})$ frames (subsampled/reordered from `MD17`), whereas contiguous dynamical rollouts $(\mathbf{r}(t), \mathbf{v}(t) \to \mathbf{r}(t+\Delta t), \mathbf{v}(t+\Delta t))$ require mapping `rMD17` indices back to continuous `MD17` timestamps or running fresh velocity-Verlet reference trajectories!

#### 24. Accurate Global Machine Learning Force Fields for Molecules with Hundreds of Atoms (`MD22`)
- **Citation:** Stefan Chmiela, Valentin Vassilev-Galindo, Oliver T. Unke, Adil Kabylda, Huziel E. Sauceda, Alexandre Tkatchenko, and Klaus-Robert Müller, *Science Advances* **9**(2), eadf0873 (2023). DOI: `10.1126/sciadv.adf0873`.
- **Core Problem & Method:** Introduces the **`MD22` supramolecular AIMD trajectory benchmark** (PBE+MBD at $400\text{–}500\text{ K}$) spanning 7 systems from 42 to 370 atoms: `Ac-Ala3-NHMe` (42), `DHA` docosahexaenoic acid (56), `stachyose` (87), `AT-AT` DNA base pair (60), `AT-AT-CG-CG` (118), `buckyball catcher` (148), and `double-walled nanotube` (370).
- **Key Metrics / Formulas:** Demonstrates that finite-cutoff MPNNs miss collective dispersion/electrostatic modes ($>6\text{ Å}$) essential for supramolecular breathing vibrations (`VDOS` $< 200\text{ cm}^{-1}$) and host–guest stability in `buckyball catcher`.
- **Concrete Design Takeaway for WignerFlow:** Adopt **`Ac-Ala3-NHMe` (42 atoms)** and **`AT-AT` (60 atoms)** from `MD22` as WignerFlow's mid-scale contiguous trajectory benchmarks: they retain exact time-ordered AIMD frames while stress-testing non-covalent hydrogen bonding and slow conformational modes where coarse-stride Wigner phase-space flows shine.

#### 25. Forces Are Not Enough: Benchmark and Critical Evaluation for Machine Learning Force Fields with Molecular Simulations
- **Citation:** Xiang Fu, Zhenghao Wu, Wujie Wang, Tian Xie, Sinan Keten, Rafael Gómez-Bombarelli, and Tommi Jaakkola, *Transactions on Machine Learning Research (TMLR)* (2023; arXiv:2210.07237, ICLR 2023 Physics4ML).
- **Core Problem & Method:** Landmark benchmark across `MD17`, `Ac-Ala3-NHMe`, periodic liquid water, and $\text{Li}$ thiophosphate solid electrolytes proving that low test Force MAE correlates poorly with actual MD rollout stability and thermodynamic observables.
- **Key Metrics / Formulas:** Defines the canonical 4-pillar rollout evaluation protocol: (1) **Stability horizon** $T_{\text{stab}}$ under $\max_{(i,j)\in\mathcal{B}} |r_{ij}(t) - b_{ij}^0| > 0.5\text{ Å}$; (2) **Interatomic distance distribution error** $\text{MAE}_{h(r)} = \int |\hat{h}(r) - h_{\text{ref}}(r)|\,dr$ for molecules; (3) **Element-pair `RDF` MAE** for periodic liquids; and (4) **Diffusivity $D$ / `VACF` spectrum** fidelity, alongside inference throughput ($\text{steps/s}$).
- **Concrete Design Takeaway for WignerFlow:** Make the **Fu et al. 4-pillar metric suite (`Stability` %, $T_{\text{stab}}$, $\text{MAE}_{h(r)}$ / `RDF` MAE, `VDOS` MAE, and wall-clock $\text{ns/day}$)** the exact core output schema of WignerFlow's pure-MD benchmark harness.

#### 26. How to Validate Machine-Learned Interatomic Potentials
- **Citation:** Joe D. Morrow, John L. A. Gardner, and Volker L. Deringer, *The Journal of Chemical Physics* **158**(12), 121501 (2023). DOI: `10.1063/5.0139611`.
- **Core Problem & Method:** Establishes a hierarchical, physics-guided validation protocol for ML potentials separating numerical point-wise errors from configurational space coverage and emergent dynamical/thermodynamic observables.
- **Key Metrics / Formulas:** Pairs locally resolved force errors (conditioned on coordination environment / `SOAP` kernel distance $d_{\text{SOAP}}(\chi, \chi_{\text{train}}) = \sqrt{2 - 2 K(\chi, \chi_{\text{train}})}$) with NVE energy conservation drift $|\Delta E_{\text{tot}}| / (N \cdot \Delta T_{\text{sim}})$ ($\text{meV/atom/ps}$), RDFs, coordination distributions, and `VDOS`.
- **Concrete Design Takeaway for WignerFlow:** Track **kinetic/Hamiltonian temperature drift** $\Delta T / T_0 = |\langle T_{\text{kin}}\rangle - T_0| / T_0$ alongside Fu et al.'s bond stability in WignerFlow rollouts: coarse-stride generative integrators can stay geometrically intact ($<0.5\text{ Å}$ bond error) while silently overheating or freezing velocities if Wigner phase-space volume is not preserved!

#### 27. EGraFFBench: Evaluation of Equivariant Graph Neural Network Force Fields for Atomistic Simulations
- **Citation:** Vaibhav Bihani, Sajid Mannan, Utkarsh Pratiush, Tao Du, Zhimin Chen, Santiago Miret, Matthieu Micoulaut, Morten M. Smedskjaer, Sayan Ranu, and N. M. Anoop Krishnan, *Digital Discovery* **3**(4), 759–768 (2024). DOI: `10.1039/D4DD00027G`.
- **Core Problem & Method:** Benchmarks 6 equivariant GNN architectures (`NequIP`, `Allegro`, `MACE`, `BOTNet`, `Equiformer`, `TorchMD-Net`) under out-of-distribution (`OOD`) temperatures, strained glasses/liquids, and long-horizon NVE/NVT rollouts.
- **Key Metrics / Formulas:** Introduces **Energy Violation / Conservation Drift**, **Structure Factor $S(q)$ & RDF error**, **`VACF`/`VDOS` fidelity**, and **OOD Rollout Survival Rate** across elevated temperature ramps ($300\text{ K} \to 3000\text{ K}$).
- **Concrete Design Takeaway for WignerFlow:** Include an **OOD temperature generalization test** (train at $300\text{–}500\text{ K}$, evaluate rollout stability and $h(r)$ widening at $1.25\times\text{–}1.5\times T_{\text{train}}$) in WignerFlow's benchmark suite.

#### 28. MLIP Arena & CHIPS-FF: Open Fair Leaderboards for Universal Machine Learning Potentials
- **Citation:** Yuan Chiang, Tobias Kreiman, Christine Zhang, Matthew C. Kuner, Elizabeth Jin Weaver, Ishan Amin, Hyunsoo Park, Yunsung Lim, Jihan Kim, Daryl Chrzan, Aron Walsh, Samuel M. Blau, Mark Asta, and Aditi S. Krishnapriyan, *MLIP Arena*, *NeurIPS 2025 Datasets and Benchmarks Track* (arXiv:2509.20630) *(paired with Daniel Wines and Kamal Choudhary, `CHIPS-FF: Evaluating Universal Machine Learning Force Fields for Material Properties`, ACS Materials Letters / arXiv:2412.10516, 2025)*.
- **Core Problem & Method:** Eliminates test-set data leakage across foundation MLIPs (`MACE-MP`, `SevenNet`, `MatterSim`, `ORB`, `CHGNet`, `EquiformerV2`) by benchmarking physical equations of state, diatomic short-range repulsion curves, high-$T/P$ MD stability, elastic/phonon spectra (`CHIPS-FF`), and condensed-phase liquid transport (`MLIP Arena`).
- **Key Metrics / Formulas:** Evaluates **short-range nuclear repulsion monotonicity** ($F(r) > 0$ as $r \to 0$, preventing pathological "atomic fusion holes" during MD), center-of-mass momentum drift $\|\sum_i \mathbf{p}_i(t)\|$, and enthalpy/density convergence.
- **Concrete Design Takeaway for WignerFlow:** Add a lightweight **short-range steric repulsion guard / diagnostic** ($\min_{i<j} r_{ij}(t)$ check) in WignerFlow's rollout evaluator, as unphysical core collapse ($r_{ij} < 0.6\text{ Å}$) is the #1 failure mode of learned neural propagators and foundation MLIPs.

#### 29. TorchSim: An Efficient Atomistic Simulation Engine in PyTorch
- **Citation:** Orion Cohen, Janosh Riebesell, Rhys Goodall, Adeesh Kolluru, Stefano Falletta, Joseph Krause, Jorge Colindres, Gerbrand Ceder, and Abhijeet S. Gangan, *arXiv:2508.06628* / *AI for Science* (2025).
- **Core Problem & Method:** Replaces single-system CPU-bound Python loops (`ASE`) with a pure-PyTorch GPU-native atomistic integrator supporting **automatic graph batching** of heterogeneous multi-system MD rollouts, NeighbourList caching, and memory-aware autobatching.
- **Key Metrics / Formulas:** Achieves **$20\times\text{–}100\times$ wall-clock speedup** over `ASE` MD loops on modern GPUs by packing $B$ independent systems of sizes $\{N_b\}$ into block-diagonal sparse graphs $(\sum_b N_b, |E_{\text{tot}}|)$ without CPU–GPU synchronization per step.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's rollout engine and benchmark runner must execute **batched parallel rollouts** ($B = 16\text{–}128$ replicas packed into one sparse PyTorch graph tensor) on GPU—never sequential Python `for`-loops over single trajectories—turning stability/$h(r)$ benchmarking from hours into seconds!

#### 30. Timewarp: Transferable Acceleration of Molecular Dynamics by Learning Time-Coarsened Dynamics
- **Citation:** Leon Klein, Andrew Y. K. Foong, Tor Erlend Fjelde, Bruno Mlodozeniec, Marc Brockschmidt, Sebastian Nowozin, Frank Noé, and Ryota Tomioka, *Advances in Neural Information Processing Systems (NeurIPS)* **36**, 52863–52883 (2023).
- **Core Problem & Method:** Trains a conditional normalizing flow that proposes large time jumps for small peptides (2 and 4 residues), using auxiliary velocity-like variables. Proposals can be accepted or rejected with a Metropolis-Hastings test so that sampling stays exact.
- **Key Metrics / Formulas:** Measures **Time-lagged Independent Component Analysis (`TICA`)** free-energy landscapes, Ramachandran Jensen–Shannon divergence (`JSD`), state transition matrices, and effective sample size per wall-clock second (`ESS/s`).
- **Concrete Design Takeaway for WignerFlow:** A key coarse-step baseline (`x -> x'` over a large time lag). We have not compared WignerFlow with Timewarp in this repo.

#### 31. Implicit Transfer Operator Learning: Multiple Time-Resolution Models for Molecular Dynamics (`ITO`)
- **Citation:** Mathias Schreiner, Ole Winther, and Simon Olsson, *Advances in Neural Information Processing Systems (NeurIPS)* **36**, 36449–36462 (2023).
- **Core Problem & Method:** Learns multi-timescale Markovian transition densities $p(\mathbf{x}_{t + N\Delta t} \mid \mathbf{x}_t, N)$ conditioned on variable lag integer $N$ using $\text{SE}(3)$-equivariant score-based diffusion (`ITO` and coarse-grained `CG-SE3-ITO`).
- **Key Metrics / Formulas:** Evaluates **Chapman–Kolmogorov self-consistency** ($T(2\tau) \approx T(\tau)^2$ across Markov State Model implied timescales $t_i = -\tau / \ln |\lambda_i(\tau)|$) and equilibrium VAMP-2 scores on alanine dipeptide and fast-folding Chignolin.
- **Concrete Design Takeaway for WignerFlow:** Add **variable-stride Chapman–Kolmogorov consistency error** (comparing $k$ rollouts of stride $\Delta t$ vs $1$ rollout of stride $k\Delta t$ in $h(r)$ and kinetic energy distribution) to WignerFlow's diagnostic suite!

#### 32. EquiJump: Protein Dynamics Simulation via $\text{SO}(3)$-Equivariant Stochastic Interpolants
- **Citation:** Allan dos Santos Costa, Ilan Mitnikov, Franco Pellegrini, Ameya Daigavane, Mario Geiger, Zhonglin Cao, Karsten Kreis, Tess Smidt, Emine Kucukbenli, and Joseph Jacobson, *arXiv:2410.09667* (2024).
- **Core Problem & Method:** Formulates coarse-stride all-atom trajectory stepping (large time jumps on fast-folding proteins) via **two-sided $\text{SO}(3)$-equivariant stochastic interpolants** bridging consecutive trajectory frames $(\mathbf{X}_t, \mathbf{X}_{t+\tau})$ directly in 3D coordinates and backbone frames without Gaussian prior destruction.
- **Key Metrics / Formulas:** Learns drift $\mathbf{b}_\theta(s, \mathbf{I}_s, \mathbf{X}_t)$ and score $\mathbf{s}_\theta(s, \mathbf{I}_s, \mathbf{X}_t)$ along $\mathbf{I}_s = (1-s)\mathbf{X}_t + s\mathbf{X}_{t+\tau} + \gamma(s)\mathbf{Z}$ ($s \in [0,1]$), recovering `TICA` free-energy basins and folding/unfolding flux rates from coarse frames.
- **Concrete Design Takeaway for WignerFlow:** Confirms that **frame-to-frame conditional flow/interpolant bridges** vastly outperform unconditional noise-to-structure diffusion for stable autoregressive rollouts because local displacement $\Delta\mathbf{r} = \mathbf{r}_{t+\Delta t} - \mathbf{r}_t$ has compact, low-variance support.

#### 33. Generative Modeling of Molecular Dynamics Trajectories (`MDGen`)
- **Citation:** Bowen Jing, Hannes Stärk, Tommi Jaakkola, and Bonnie Berger, *Advances in Neural Information Processing Systems (NeurIPS)* **37** (2024; arXiv:2409.17808).
- **Core Problem & Method:** Tokenizes entire multi-frame trajectories ($K = 100\text{–}1000$ frames) into $\text{SE}(3)$-invariant/equivariant keyframe-anchored internal roto-translational tokens and trains a scalable **SiT/Flow-Matching Transformer** over space–time simultaneously for forward simulation, transition-path inpainting, and temporal upsampling.
- **Key Metrics / Formulas:** Benchmarks torsional JSD, flux/decorrelation timescales, dynamical MSM transition matrices, and **trajectory upsampling (`1 ns -> 10 ps`)** reconstructing high-frequency sidechain dynamics from coarse frames.
- **Concrete Design Takeaway for WignerFlow:** Highlights an important multi-resolution capability for WignerFlow: if coarse-stride rollouts ($\Delta t = 10\text{–}50\text{ fs}$) filter out ultra-fast $\text{C–H}$/$\text{O–H}$ bond stretches above $\omega_{\text{Nyq}}$, short bursts of fine-stride rollout can reconstruct full-bandwidth `VDOS` spectra on demand.

#### 34. Scalable Emulation of Protein Equilibrium Ensembles with Generative Deep Learning (`BioEmu`)
- **Citation:** Sarah Lewis, Timur Hempel, José Jiménez-Luna, Michael Gastegger, Yu Xie, Andrew Y. K. Foong, Victor García Satorras, Osama Abdin, et al., and Frank Noé, *Science* **389** (2025; bioRxiv `2024.12.05.626885`). DOI: `10.1126/science.adv9817`.
- **Core Problem & Method:** Trains a generative score/flow foundation emulator on $>200\text{ ms}$ of aggregate MD trajectories plus experimental folding free energies ($\Delta G_{\text{fold}}$ via property-alignment fine-tuning) to sample independent equilibrium conformations at $10^3\text{–}10^4$ structures/GPU-hour.
- **Key Metrics / Formulas:** Achieves folding stability free-energy MAE $\approx 0.8\text{–}1.0\text{ kcal/mol}$ and cryptic-pocket/local-unfolding ensemble recall, using **free-energy preference/reweighting losses** to correct generative distribution bias.
- **Concrete Design Takeaway for WignerFlow:** Separates **equilibrium ensemble samplers** (`BioEmu`, which lose time-correlation kinetics $\tau_{\text{ex}}, D, C_v(t)$) from **true dynamical phase-space propagators** (`WignerFlow`, `TrajCast`): WignerFlow's unique value proposition on electrolytes and `MD17`/`MD22` is preserving *both* Boltzmann structure ($g(r), \chi_{\text{CIP}}$) *and* real-time kinetics ($\tau_{\text{ex}}, D, S_{\text{VDOS}}(\omega)$)!

#### 35. Force-Free Molecular Dynamics Through Autoregressive Equivariant Networks (`TrajCast`) & Energy-Consistent Diffusion (`ScoreMD`)
- **Citation:** Fabian L. Thiemann, Thiago Reschützegger, Massimiliano Esposito, Tseden Taddese, Juan D. Olarte-Plata, and Fausto Martelli, *TrajCast*, *arXiv:2503.23794* (2025) *(paired with Michael Plainer, Hao Wu, Leon Klein, Stephan Günnemann, and Frank Noé, `ScoreMD: Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models`, arXiv:2506.17139, 2025)*.
- **Core Problem & Method:** `TrajCast` uses an $\text{E}(3)$-equivariant autoregressive message-passing network (`MACE`/`NequIP`-style spherical tensors taking *both* positions $\mathbf{r}(t)$ and velocities $\mathbf{v}(t)$) to directly forecast $(\Delta\mathbf{r}, \Delta\mathbf{v})$ at $10\times\text{–}30\times$ larger timesteps ($\Delta t = 5\text{–}30\text{ fs}$) on `rMD17`/paracetamol, bulk liquid water, and quartz without force evaluations, enforcing exact zero linear/angular momentum drift and optional thermostatting; `ScoreMD` regularizes diffusion scores via Fokker–Planck consistency so score-driven Langevin rollouts match equilibrium Boltzmann densities.
- **Key Metrics / Formulas:** `TrajCast` predicts equivariant $L=1$ displacement vectors $(\hat{\Delta\mathbf{r}}_i, \hat{\Delta\mathbf{v}}_i)$, removes net COM drift $\Delta\mathbf{v}_i \leftarrow \Delta\mathbf{v}_i - \frac{1}{M}\sum_j m_j \Delta\mathbf{v}_j$, and validates **RDFs, `VACF`/`VDOS` up to Nyquist, and self-diffusivity $D$** at $10\times\text{–}30\times$ coarse stride!
- **Concrete Design Takeaway for WignerFlow:** **`TrajCast` is the closest direct architectural cousin and primary benchmark comparator for WignerFlow!** Adopting `TrajCast`'s exact conservation post-projection (removing net COM translation/momentum drift and supporting CSVR/Bussi velocity rescaling in NVT rollouts) guarantees long-horizon rollout stability in WignerFlow.

#### 36. The Atomic Simulation Environment—A Python Library for Working with Atoms (`ASE`)
- **Citation:** Ask Hjorth Larsen et al., *Journal of Physics: Condensed Matter* **29**(27), 273002 (2017). DOI: `10.1088/1361-648X/aa680e`.
- **Core Problem & Method:** Establishes the universal Python `Atoms` data model (`positions`, `momenta`/`velocities`, `cell`, `pbc`, `numbers`) and decoupled `Calculator` interface (`get_potential_energy`, `get_forces`, `get_stress`), plus binary `.traj` / extended XYZ (`.extxyz`) serialization and `VelocityVerlet`/`Langevin` integrators.
- **Key Metrics / Formulas:** Standardizes internal units (`eV` energy, `Å` length, `eV/Å` force, `fs = 0.09822694788...` internal ASE time units via `ase.units.fs`) across $>40$ quantum/classical codes.
- **Concrete Design Takeaway for WignerFlow:** WignerFlow's CLI must treat **`.extxyz` and `ASE .traj`** as native zero-config input/output formats and expose both a standard `ASE Calculator` wrapper (when forces/energies are queried) and a **coarse-stride `WignerDynamics` driver** mirroring `ase.md.md.MolecularDynamics` (`dyn.run(steps)`), careful to convert `ase.units.fs` velocities accurately!

#### 37. MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations
- **Citation:** Naveen Michaud-Agrawal, Elizabeth J. Denning, Thomas B. Woolf, and Oliver Beckstein, *Journal of Computational Chemistry* **32**(10), 2319–2327 (2011), DOI: `10.1002/jcc.21787`; and Richard J. Gowers et al., *Proceedings of the 15th Python in Science Conference (SciPy)*, 98–105 (2016), DOI: `10.25080/Majora-629e541a-00e`.
- **Core Problem & Method:** Object-oriented `Universe(topology, trajectory)` streaming engine with Cython/OpenMP cell-list neighbor searches (`capped_distance`), `InterRDF`, `EinsteinMSD`, and ` SolvationAnalysis`-compatible `AtomGroup` boolean algebra (`around 2.8 group Li`).
- **Key Metrics / Formulas:** Computes periodic minimum-image cell-grid neighbor pairs in $O(N)$ time (`lib.distances.capped_distance`) rather than $O(N^2)$ all-pairs distance matrices, scaling `RDF` and `SSIP`/`CIP`/`AGG` extraction to $10^4\text{–}10^5$ atoms.
- **Concrete Design Takeaway for WignerFlow:** Use cell-list / `torch_cluster.radius_graph` or `scipy.spatial.cKDTree(..., boxsize=L)` $O(N)$ periodic neighbor search in WignerFlow's solvation analyzer—never dense $N \times N \times 3$ broadcasting—so CLI analysis of $10{,}000$-frame electrolyte trajectories completes in $<2\text{ s}$.

#### 38. MDTraj: A Modern Open Library for the Analysis of Molecular Dynamics Trajectories
- **Citation:** Robert T. McGibbon, Kyle A. Beauchamp, Matthew P. Harrigan, Christoph Klein, Jason M. Swails, Carlos X. Hernández, Christian R. Schwantes, Lee-Ping Wang, Thomas J. Lane, and Vijay S. Pande, *Biophysical Journal* **109**(8), 1528–1532 (2015). DOI: `10.1016/j.bpj.2015.08.015`.
- **Core Problem & Method:** SIMD/SSE-vectorized C/Python library (`mdtraj.Trajectory`) supporting fast `compute_rdf`, `compute_distances`, `rmsd` (Theobald QCP quaternion alignment in $O(N)$), dihedral angles, and compact HDF5/DCD/XTC binary I/O.
- **Key Metrics / Formulas:** Vectorized minimum-image distances and quaternion Kabsch/QCP alignment $\min_{\mathbf{R}\in\text{SO}(3)} \frac{1}{N}\sum_i \|\mathbf{R}\mathbf{x}_i - \mathbf{y}_i\|^2$ at $>10\times$ higher single-core throughput than legacy Python scripts.
- **Concrete Design Takeaway for WignerFlow:** Adopt `MDTraj`'s philosophy for WignerFlow's CLI: support streaming/chunked `.xyz`/`.extxyz`/`.npz`/`.h5` trajectory I/O with structured JSON + Markdown summary reports so users can pipe `wignerflow rollout` directly into `wignerflow analyze-solvation` or `wignerflow bench`.

#### 39. OpenMM 8: Molecular Dynamics Simulation with Machine Learning Potentials
- **Citation:** Peter Eastman et al., and Thomas E. Markland, *The Journal of Physical Chemistry B* **128**(1), 109–116 (2024). DOI: `10.1021/acs.jpcb.3c06662`.
- **Core Problem & Method:** Embeds TorchScript / `OpenMM-Torch` / `OpenMM-ML` directly inside `OpenMM`'s CUDA/OpenCL integrator loop, supporting pure-ML potentials and hybrid `ML/MM` solvation zones without PCIe host–device roundtrips.
- **Key Metrics / Formulas:** Eliminates Python-interpreter and CPU–GPU copy overhead per step ($\sim 1\text{–}5\text{ ms/step}$ saved), keeping positions, velocities, neighbor lists, and thermostats (`LangevinMiddleIntegrator`, `MonteCarloBarostat`) 100% resident in GPU VRAM.
- **Concrete Design Takeaway for WignerFlow:** Keep WignerFlow's entire autoregressive rollout loop (`step = 1 ... N_steps`, PBC wrapping, COM zeroing, stability check, and strided trajectory buffer) **100% on-device in PyTorch (`torch.no_grad()` / `torch.compile`)**, transferring trajectory chunks to CPU disk only every `save_every` frames.

#### 40. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields (& `MACE-MP-0` CLI Ecosystem)
- **Citation:** Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, and Gábor Csányi, *Advances in Neural Information Processing Systems (NeurIPS)* **35**, 11423–11436 (2022) *(paired with Ilyes Batatia et al., `A foundation model for atomistic materials chemistry` [`MACE-MP-0`], arXiv:2401.00096, 2024)*.
- **Core Problem & Method:** Combines Atomic Cluster Expansion (`ACE`) higher-body-order equivariant tensor products ($\nu = 3$, 4-body messages in just 2 MPNN layers) with clean, composable CLI entrypoints (`mace_run_train`, `mace_eval_configs`, `MACECalculator`) and zero-shot pretrained foundation checkpoints (`small`, `medium`, `large`).
- **Key Metrics / Formulas:** Achieves SOTA accuracy on `rMD17` (e.g., aspirin Force MAE $\approx 6.6\text{ meV/Å} \approx 0.15\text{ kcal/(mol}\cdot\text{Å)}$), `MD22`, and condensed-phase liquids with shallow 2-layer receptive fields that parallelize cleanly across GPUs.
- **Concrete Design Takeaway for WignerFlow:** Model WignerFlow's CLI ergonomics directly on `MACE`'s clean subcommand design: provide ergonomic, declarative subcommands (`wignerflow rollout`, `wignerflow eval-md`, `wignerflow solvation`) with sensible defaults, JSON/CSV artifact emission, deterministic `--seed`, `--device`, and pluggable reference calculators (`MACE` / `ASE`) for instant baseline comparisons.

---

## Consolidated Engineering Blueprint for WignerFlow's Next Three Modules

1. **Module A — Battery Electrolyte Solvation & Exchange Kinetics (`solvation`):**
   - **Inputs:** Periodic trajectory (`(T, N, 3)` positions, `(3, 3)` cell, atomic numbers `Z`, molecule/ion residue IDs or automated covalent bond graph clustering via covalent radii + $0.4\text{ Å}$ tolerance to auto-detect $\text{Li}^+$, anions, primary solvents, and diluents/additives!).
   - **Core Computations:**
     - Periodic minimum-image partial RDFs $g_{\text{Li},\beta}(r)$ ($\Delta r = 0.02\text{ Å}$, $r \in [0, 8.0]\text{ Å}$) + running $CN_{\text{Li},\beta}(r)$.
     - Auto-detected first-minimum cutoffs $r_c \in [2.4, 3.3]\text{ Å}$ (defaults: $r_c(\text{O})=2.80\text{ Å}$, $r_c(\text{F})=2.75\text{ Å}$, $r_c(\text{N})=3.00\text{ Å}$).
     - Exact bipartite connected-component speciation ($\chi_{\text{SSIP}}, \chi_{\text{CIP}}, \chi_{\text{AGG}}$, dimer `AGG-I` vs network `AGG-net`, percolation fraction $P_\infty$, free-solvent fraction $f_{\text{free}}$) + additive enrichment $\alpha_{\text{enr}}$ & $\Delta G_{\text{pref}}$.
     - FFT-accelerated cage residence autocorrelation $C_{\text{cage}}(t) = \langle h_{ij}(0)h_{ij}(t)\rangle / \langle h_{ij}\rangle$, KWW stretched-exponential lifetime $\tau_{\text{ex}}$, vehicularity index $\Lambda_{\text{veh}} = \sqrt{6 D_{\text{Li}}\tau_{\text{ex}}}/r_c$, and `cNE` cluster conductivity $\sigma_{\text{cNE}}$ & transference number $t_+^{\text{cNE}}$.
     - Automated **"Fast-Charging / Stable-SEI Readiness Score"** checking $\chi_{\text{CIP}} \in [0.15, 0.45]$, $\chi_{\text{AGG}} \le 0.25$ (unless `LHCE` diluent detected), $P_\infty < 0.35$, and $\tau_{\text{ex}} \in [20, 300]\text{ ps}$.

2. **Module B — Pure-MD Rollout Benchmark Suite (`MD17` / `rMD17` / `MD22` / `Fu et al.`):**
   - **Metrics Implemented Exactly per `Fu et al. (2023)`:**
     - Covalent bond-rupture stability horizon $T_{\text{stab}}$ ($\max_{(i,j)\in\mathcal{B}} |r_{ij}(t) - b_{ij}^0| > 0.5\text{ Å}$) + steric-clash guard ($\min_{i<j} r_{ij}(t) < 0.7\text{ Å}$).
     - Normalized all-pair interatomic distance density $\text{MAE}_{h(r)} = \int_0^\infty |\hat{h}(r) - h_{\text{ref}}(r)|\,dr$ (for finite molecules) and periodic $\text{MAE}_{\text{RDF}}$ (for bulk liquids).
     - Windowed `VACF` $\to$ `VDOS` power spectrum $S_{\text{VDOS}}(\tilde{\nu})$ MAE (explicitly bounded at Nyquist $\tilde{\nu}_{\text{Nyq}} = \frac{1}{2c\Delta t}$ for coarse-stride rollouts!), kinetic temperature drift $|\langle T_{\text{kin}}\rangle - T_0|/T_0$, COM momentum drift, and throughput (`steps/s`, `ns/day`).

3. **Module C — Fast Composable Scientific MD CLI (`wignerflow`):**
   - **Subcommands:** `wignerflow rollout` (on-GPU batched trajectory generation with COM-drift removal & `.extxyz`/`.npz` output), `wignerflow benchmark` (`Fu et al.` evaluation against reference trajectory), and `wignerflow solvation` (full battery electrolyte speciation, RDF, $\tau_{\text{ex}}$, `cNE`, and fast-charging diagnostic report in JSON/Markdown).
