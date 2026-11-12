# Large time step test on MD17 and MD22 molecules

`equitraj/mdbench.py` and `scripts/bench_pure_md17.py` test whether a simple
coarse-step integrator stays stable at `10 fs` and `20 fs` time steps on 7 public
quantum MD datasets (`9` to `60` atoms). Normal MD uses about `0.5 fs`.

## What this benchmark is (and is not)

- The force model is `EquivariantPairFlowModel`: a small pair model (harmonic bonds and
  angles plus a fitted radial residual) fit on the training split of each dataset.
- The coarse step (`rollout_wigner_coarse`) evaluates the slow forces once per big step,
  then keeps bond lengths in range with a bond projection and holds the temperature with
  a velocity-rescaling thermostat.
- It is **not** the trained WignerFlow neural network. It is a baseline for the
  stability checks we want any coarse-step model to pass.
- The reference is a `0.5 fs` Velocity-Verlet run with the same pair model, not a
  quantum-level MD run. Speedups are measured against that reference on one CPU.

Data comes from the public sGDML / MD22 archives (Chmiela et al., Sci. Adv. 2017,
Nat. Commun. 2018, Sci. Adv. 2023). The scripts download them on first use.

## Metrics (following Fu et al., TMLR 2023)

1. **Stability**: a run fails if any bond moves more than `0.5 Å` from its reference
   length, or any two atoms come closer than `0.7 Å`.
2. **h(r) error**: mean absolute error between the all-pair distance histogram of the
   rollout and that of the held-out quantum test frames.
3. **VDOS cosine**: cosine similarity of the vibrational spectrum, up to the Nyquist limit
   of the step size.
4. **Step-size consistency**: h(r) difference between the `10 fs` and `20 fs` runs.
5. **Drift and speed**: center-of-mass drift, temperature drift, and CPU `ns/day`.

## Results (`1.0 ps` runs at `500 K`)

All numbers below come from `results/pure_md_md17/md17_md22_benchmark.json`.

| System | Theory | Atoms | Plain Verlet `10 fs` | Coarse `10 fs` stable | Coarse `20 fs` stable | h(r) MAE (`10 fs`) | VDOS cosine (`10 fs`) | Speedup `10 fs` | Speedup `20 fs` | CPU ns/day (`10 fs`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ethanol | CCSD(T) | 9 | fails at `0.01 ps` | yes | yes | `0.1997` | `0.6161` | `2.38x` | `4.79x` | `1,379` |
| Malonaldehyde | CCSD(T) | 9 | fails at `0.01 ps` | yes | yes | `0.1388` | `0.5074` | `1.99x` | `4.62x` | `1,173` |
| Benzene | CCSD(T) | 12 | fails at `0.01 ps` | yes | yes | `0.5152` | `0.5388` | `1.88x` | `3.70x` | `1,142` |
| Uracil | DFT | 12 | fails at `0.01 ps` | yes | yes | `0.1653` | `0.7227` | `1.73x` | `3.95x` | `1,139` |
| Toluene | CCSD(T) | 15 | fails at `0.01 ps` | yes | yes | `0.2615` | `0.7759` | `3.72x` | `5.68x` | `2,013` |
| Aspirin | CCSD | 21 | fails at `0.01 ps` | yes | yes | `0.1311` | `0.7220` | `3.67x` | `5.23x` | `1,420` |
| MD22 AT-AT | DFT-PBE+MBD | 60 | fails at `0.01 ps` | yes | yes | `0.1076` | `0.7470` | `6.35x` | `9.76x` | `633` |
| Mean | | | 0 of 7 stable | 7 of 7 | 7 of 7 | `0.2170` | `0.6614` | `3.10x` | `5.39x` | `1,271` |

For comparison, the `0.5 fs` reference run has a mean h(r) MAE of `0.1388`, so the big
step adds error. The `20 fs` runs have a mean h(r) MAE of `0.3191`. The largest
center-of-mass drift is about `2e-18 Å/fs`, and temperature drift is `0.01%` to `0.84%`.

These are 1 ps runs on single small molecules. They show the coarse step does not blow
up; they do not show that it gives correct long-time dynamics.

## Reproduce

```bash
python scripts/bench_pure_md17.py --horizon-ps 1.0   # also writes results/pure_md_md17/md17_coarse_step_benchmark.png
pytest tests/test_mdbench.py -q
```
