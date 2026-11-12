# 1,000,000-recipe electrolyte screening funnel (experimental)

`equitraj/funnel.py` and `scripts/electrolyte_funnel_1m.py` build a list of 1,000,000
made-up Li+ electrolyte recipes and rank them with a 3-step funnel on CPU:

`10 salts x 20 solvent pairs x 100 additives x 10 blend ratios x 5 salt concentrations = 1,000,000`

The 100 additives are hand-made SMILES strings. `tests/test_funnel_1m.py` checks that they
parse in RDKit and are unique (the test is skipped if RDKit is not installed). None of them has
been made or tested in a lab.

## The three steps

1. **Step 1 (1,000,000 -> 180)**: score every recipe at `298.15 K` and `253.15 K` with the
   Stage 2 aligner from `equitraj/align.py` (trained on CALiSol-23 in the same run), and keep a
   diverse set across solvents and additive families.
2. **Step 2 (180 -> 20)**: build a small periodic box for each recipe, check for atom overlap,
   and run a short `FastRollout` burst to check it stays stable.
3. **Step 3 (20 -> top 15)**: run a longer rollout and compute cluster Nernst-Einstein
   ionicity, a hopping index, a transference number estimate, and corrected conductivity.

## Saved run

`results/funnel_1m/top15_novel_electrolytes.json` stores one run: the top 15 recipes, and
CPU wall times of `0.992 s` (step 1), `22.136 s` (step 2), `20.347 s` (step 3), and
`43.66 s` in total.

All conductivities in that file are **model predictions**. The aligner's accuracy on
CALiSol-23 is not yet reproduced in this repo (see `docs/CALISOL_ALIGNMENT.md`), and none of
the top recipes has been checked by full MD or by experiment. Treat the list as a demo of the
screening pipeline, not as a list of discovered electrolytes. WignerFlow's public generation
results are for argon only.

## Usage

```bash
python scripts/electrolyte_funnel_1m.py --calisol-csv "CALiSol-23 Dataset.csv" --space-size 1000000 --json-out results/funnel_1m/top15_novel_electrolytes.json
```
