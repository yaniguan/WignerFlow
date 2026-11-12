# Li+ solvation shells of the top 15 funnel recipes (experimental)

`equitraj/solvation.py` and `scripts/analyze_top15_solvation.py` take the top 15 recipes
from the 1,000,000-recipe funnel (`results/funnel_1m/top15_novel_electrolytes.json`) plus a
`1.0 M LiPF6 in EC:DMC (1:1)` baseline, build a small periodic box for each, run a short
Langevin (BAOAB) simulation with a simple toy force field, and look at which molecules sit
next to each Li+ ion.

This is a demo of the analysis tools. The force field is a coarse toy model, the boxes are
small, and the runs are short. None of these numbers has been checked against full MD or
experiment. WignerFlow's public generation results are for argon only.

## What is computed

- **Partial RDFs and coordination numbers** for Li+ with solvent, additive, and anion, with the
  cutoff set at the first minimum.
- **Ion-pair types**, from the connected clusters of Li+ and anions:
  - `SSIP`: Li+ with only solvent around it (free ion).
  - `CIP`: Li+ touching exactly one anion (simple ion pair).
  - `AGG-I`: small clusters (up to 4 ions). `AGG-net`: larger connected networks (5 or more ions).
- **Additive enrichment**: how much more often the additive sits next to Li+ than its share in
  the liquid would suggest.
- **Shell exchange time** `tau_ex`: how long a neighbor stays in a Li+ shell.
- **Readiness score** (0 to 100) and **regime** label: our own heuristic rules, not a validated
  metric. The working ranges are listed in
  `docs/literature/SOLVATION_AND_PURE_MD_40_PAPER_SURVEY.md`.

## Results from the saved run

All numbers below come from `results/solvation_top15/top15_solvation_report.json`.
Conductivities are copied from the funnel output (model predictions). The baseline
conductivities (11.20 and 2.85 mS/cm) are hand-entered approximate values; CALiSol-23 lists
12.09 mS/cm at 297.9 K and 3.31 mS/cm at 253.4 K for this recipe (doi 10.1016/j.jpowsour.2011.06.030).

| Rank | Recipe | sigma 298 K (mS/cm) | sigma 253 K (mS/cm) | SSIP | CIP | AGG-I | AGG-net | tau_ex (ps) | Readiness | Regime |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| Baseline | 1.0M LiPF6 in EC:DMC (1:1) | 11.20 | 2.85 | 0.90 | 0.10 | 0.00 | 0.00 | 2.37 | 58.9 | `SSIP_SOLVENT_DOMINATED` |
| #1 | 1.0M LiFSI in AN:EA + 5wt% F-Ester-1 | 35.65 | 18.35 | 0.75 | 0.00 | 0.25 | 0.00 | 1.73 | 46.6 | `AGGREGATE_ENRICHED_WSE` |
| #2 | 1.0M LiFSI in AN:DMC + 5wt% F-Ester-1 | 30.85 | 15.01 | 0.64 | 0.21 | 0.10 | 0.04 | 0.80 | 73.8 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #3 | 1.0M LiFSI in EC:MA + 35wt% F-Ester-2 | 24.59 | 12.05 | 0.38 | 0.38 | 0.23 | 0.00 | 1.03 | 72.3 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #4 | 1.0M LiFSI in FEC:MA + 35wt% F-Ester-2 | 20.19 | 9.84 | 0.52 | 0.25 | 0.23 | 0.00 | 1.68 | 83.7 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #5 | 1.0M LiFSI in EC:DME + 35wt% F-Ester-3 | 20.48 | 9.52 | 0.41 | 0.34 | 0.25 | 0.00 | 0.57 | 76.1 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #6 | 1.0M LiFSI in 2-MeTHF:DME + 35wt% F-Ester-3 | 16.85 | 10.41 | 0.26 | 0.31 | 0.09 | 0.33 | 1.90 | 70.2 | `AGGREGATE_ENRICHED_WSE` |
| #7 | 1.0M LiFSI in FEC:EA + 35wt% F-Ester-4 | 17.82 | 8.39 | 0.88 | 0.12 | 0.00 | 0.00 | 1.33 | 70.8 | `SSIP_SOLVENT_DOMINATED` |
| #8 | 1.0M LiFSI in PC:EA + 35wt% F-Ester-4 | 17.18 | 8.24 | 0.38 | 0.62 | 0.00 | 0.00 | 1.08 | 44.6 | `AGGREGATE_ENRICHED_WSE` |
| #9 | 1.0M LiFSI in DME:DOL + 20wt% F-Ester-5 | 13.93 | 8.17 | 0.30 | 0.45 | 0.25 | 0.00 | 1.23 | 61.1 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #10 | 1.0M LiFSI in PC:DMC + 35wt% F-Ester-6 | 15.00 | 6.66 | 0.50 | 0.38 | 0.05 | 0.07 | 2.03 | 72.7 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #11 | 1.0M LiFSI in EC:DMC + 35wt% F-Ester-5 | 15.11 | 6.42 | 0.75 | 0.25 | 0.00 | 0.00 | 3.33 | 89.1 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #12 | 1.0M LiFSI in EC:EMC + 35wt% F-Ester-6 | 14.57 | 6.11 | 0.36 | 0.64 | 0.00 | 0.00 | 1.10 | 44.0 | `AGGREGATE_ENRICHED_WSE` |
| #13 | 1.0M LiFSI in GBL:DMC + 35wt% F-Ester-7 | 12.95 | 6.38 | 0.47 | 0.34 | 0.18 | 0.00 | 2.78 | 78.9 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #14 | 1.0M LiFSI in FEC:EMC + 35wt% Cyc-Acetal-1 | 13.26 | 5.82 | 0.50 | 0.14 | 0.36 | 0.00 | 0.60 | 65.3 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |
| #15 | 1.0M LiFSI in PC:EMC + 35wt% F-Ester-7 | 13.21 | 5.75 | 0.52 | 0.44 | 0.04 | 0.00 | 3.22 | 64.6 | `GOLDILOCKS_FAST_CHARGE_ANION_SEI` |

Summary over the top 15 (from `top15_summary`):

- 10 of 15 recipes fall in our working target range (`GOLDILOCKS_FAST_CHARGE_ANION_SEI`).
- Mean CIP fraction is 32.6% vs. 9.8% for the baseline.
- Mean shell exchange time is 1.63 ps vs. 2.37 ps for the baseline.
- Recipe #6 (`2-MeTHF:DME`) has the largest AGG-net fraction (33%).

These trends come from a toy model and may not hold in real electrolytes.

## Reproduce

```bash
python scripts/analyze_top15_solvation.py
pytest tests/test_solvation.py -q
```
