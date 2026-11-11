# Fine-Grained 3D Solvation Shell & Exchange Kinetics of Top 15 `1,000k`-Funnel Electrolytes

Building on our 40-paper solvation survey (`Xu 2014`; `Rajput & Persson JACS 2015`; `Zhang & Maginn JPCL 2015`; `Borodin et al. 2017/2020`; `Yamada et al. Nat. Energy 2019`; `Cao et al. Nat. Energy 2019`; `France-Lanord & Grossman PRL 2019`; `Yu, Cui & Bao Nat. Energy 2020/2022`; `Xu, Borodin & Wang Nature 2023`), [`equitraj/solvation.py`](../equitraj/solvation.py) performs explicit multi-component 3D periodic BAOAB solvation rollouts across all **Top 15 novel formulations** discovered by the `1,000,000`-candidate funnel alongside a **`1.0 M LiPF6 in EC:DMC (1:1)` commercial baseline**.

## 1. Microscopic Observables Computed

- **Partial RDFs & Auto-Cutoffs**: Minimum-image $g_{\text{Li},\text{Backbone}}(r)$, $g_{\text{Li},\text{Additive}}(r)$, $g_{\text{Li},\text{Anion}}(r)$, running $CN(r)$, and automated first-minimum cutoff $r_c$.
- **Exact Bipartite Connected-Component Speciation**:
  - **Cation-centric**: `SSIP`, `CIP` ($|V_+|=1, |V_-|=1$), finite dimers `AGG-I` ($\le 4$ ions), extended networks `AGG-net` ($\ge 5$ ions), and percolation order parameter $P_\infty$.
  - **Anion-centric Raman Proxy**: `Free`, `CIP`, `AGG-I`, `AGG-II` + `cNE` cluster ionicity $\mathcal{I}_{\text{cNE}}$.
- **Competitive Additive Enrichment & Cage Exchange**: Enrichment index $\alpha_{\text{enr}}$, $\Delta G_{\text{pref}}$ ($\text{kJ/mol}$), Yamada free-solvent fraction $f_{\text{free}}$, and FFT intermittent cage lifetime $\tau_{\text{ex}}$.

## 2. Top 15 Solvation Audit vs. Commercial `EC:DMC` Baseline

| Rank | Formulation | $\sigma_{298\text{K}}$ (mS/cm) | $\sigma_{253\text{K}}$ (mS/cm) | `SSIP` | `CIP` | `AGG-I` | `AGG-net` | $\tau_{\text{ex}}$ (ps) | Readiness (/100) | Regime |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Base** | `1.0M LiPF6 in EC:DMC (1:1)` | `11.20` | `2.85` | `0.90` | `0.10` | `0.00` | `0.00` | `2.37` | `58.9` | `SSIP_SOLVENT_DOMINATED` |
| **#2** | `1.0M LiFSI in AN:DMC + 5wt% F-Ester-1` | **`30.85`** | **`15.01`** | `0.64` | **`0.21`** | `0.10` | `0.04` | `0.80` | **`73.8`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |
| **#3** | `1.0M LiFSI in EC:MA + 35wt% F-Ester-2` | `24.59` | `12.05` | `0.38` | **`0.38`** | `0.23` | `0.00` | `1.03` | **`72.3`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |
| **#4** | `1.0M LiFSI in FEC:MA + 35wt% F-Ester-2` | `20.19` | `9.84` | `0.52` | **`0.25`** | `0.23` | `0.00` | `1.68` | **`83.7`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |
| **#5** | `1.0M LiFSI in EC:DME + 35wt% F-Ester-3` | `20.48` | `9.52` | `0.41` | **`0.34`** | `0.25` | `0.00` | `0.57` | **`76.1`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |
| **#6** | `1.0M LiFSI in 2-MeTHF:DME + 35wt% F-Ester-3` | `18.70` | `8.71` | `0.26` | `0.31` | `0.09` | **`0.33`** | `1.90` | `70.2` | `AGGREGATE_ENRICHED_WSE` *(Flagged)* |
| **#11** | `1.0M LiFSI in EC:DMC + 35wt% F-Ester-5` | `14.88` | `6.52` | `0.75` | **`0.25`** | `0.00` | `0.00` | `3.33` | **`89.1`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |
| **#13** | `1.0M LiFSI in GBL:DMC + 35wt% F-Ester-7` | `14.12` | `6.18` | `0.47` | **`0.34`** | `0.18` | `0.00` | `2.78` | **`78.9`** | **`GOLDILOCKS_FAST_CHARGE_ANION_SEI`** |

- **10 of 15 (`66.7%`)** top funnel hits satisfy the **Goldilocks Fast-Charge + Anion-Derived SEI window** ($\chi_{\text{CIP}} \in [15\%, 45\%]$, $\chi_{\text{AGG-net}} \le 15\%$, $P_\infty < 0.35$), tripling primary-shell anion participation relative to commercial `EC:DMC` (`32.6%` mean `CIP` vs `9.8%`) while accelerating solvation-cage turnover by `31%`.
- **Actionable De-Risking**: Candidate `#6` (`2-MeTHF:DME`) is automatically flagged for `33%` extended `AGG-net` formation due to low cyclic-ether dielectric screening, whereas `#2` (`AN:DMC + 5wt% F-Ester-1`), `#4` (`FEC:MA + 35wt% F-Ester-2`), and `#11` (`EC:DMC + 35wt% F-Ester-5`) balance bulk rate capability and inorganic $\text{LiF}$ SEI formation.

## 3. Reproducing the Analysis

```bash
python scripts/analyze_top15_solvation.py
pytest tests/test_solvation.py -q
```
