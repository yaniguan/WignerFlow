# CALiSol-23 box-size study and WignerFlow-Align (experimental)

`equitraj/align.py` and `scripts/calisol_align.py` are tools for linking short
WignerFlow-style rollouts of small periodic Li+ electrolyte boxes to measured ionic
conductivity in the public `CALiSol-23` dataset.

This is experimental tooling. WignerFlow's public generation results are for argon only.
Nothing here shows that WignerFlow has been validated on Li+ electrolytes.

## Data

`CALiSol-23` (de Blasio, Elsborg, Vegge, Flores & Bhowmik, *Sci. Data* 11, 750, 2024;
CC BY 4.0) has 13,825 measured conductivities from 27 papers, with 14 lithium salts and
38 solvents. Download `CALiSol-23 Dataset.csv` from DTU Data
(https://doi.org/10.11583/DTU.c.6929599) or https://github.com/Pele0599/CALiSol-23.
The CSV is not stored in this repo. `load_calisol23` drops rows it cannot use, so the row
count after loading is a little lower than 13,825.

## What the code does

1. **Box-size study** (`analyze_electrolyte_box`, `--mode box-study`): for a given recipe and
   number of ion pairs, estimates the box length, atom count, viscosity, and the Yeh-Hummer
   finite-size correction `D_inf = D_box + 2.837297 kB T / (6 pi eta L)`. The CPU time field is
   a rough estimate.
2. **Measured CPU speed**: `results/calisol/box_scaling_cpu.json` holds a measured CPU sweep of
   two model sizes from 256 to 10,080 atoms (ms per 50 fs step, steps/s, ns/day).
3. **Stage 1 post-training** (`posttrain_equitraj_transport`): short unrolled rollouts with loss
   terms for temperature, velocity autocorrelation, mean-squared displacement, and atom
   overlap, plus EMA weights.
4. **Stage 2 aligner** (`CALiSolTransportAligner`): a small model that maps recipe descriptors
   and rollout observables to conductivity, using charge scaling, a viscosity term, the
   Yeh-Hummer correction, and a cluster Nernst-Einstein ionicity. Train and test rows are split
   by recipe group, so test recipes are not seen in training.

## Results

One run of `--mode align-eval` (seed 42, CPU, about 5.5 minutes) is saved in
`results/calisol/align_eval_seed42.json`. After loading, 13,228 rows remain; they are split
by recipe group into 10,476 train and 2,752 test rows.

| Model | Test set | R² | MAE (mS/cm) | Spearman |
| :--- | :--- | :---: | :---: | :---: |
| Raw rollout, Nernst-Einstein | 72-row rollout panel | `-0.4143` | `4.181` | `0.0871` |
| Stage 1 post-training + Yeh-Hummer | 72-row rollout panel | `-15.8541` | `18.873` | `0.1565` |
| Stage 1 + Stage 2 aligner | 72-row rollout panel | `0.5739` | `2.076` | `0.8296` |
| Stage 2 aligner, recipe descriptors only | all 2,752 test rows | `0.7218` | `1.275` | `0.9058` |

How to read this:

- Raw rollouts of these small toy boxes do not predict conductivity (R² below zero).
- Most of the accuracy comes from the Stage 2 aligner. On the full test set it uses only recipe
  descriptors (no rollout input), so the `0.7218` row is a descriptor model result, not an MD
  result.
- The rollout panel numbers change from run to run because the rollouts are random.

## Usage

```bash
python scripts/calisol_align.py --mode box-study
python scripts/calisol_align.py --mode align-eval --calisol-csv "CALiSol-23 Dataset.csv" --json-out results/calisol/align_eval.json
```
