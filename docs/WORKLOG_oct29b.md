# WignerFlow packaging and full-retrain preparation — October 29, 2026

No git commands were run. No full data generation or GPU training was attempted locally. The original `configs/argon/B_S3a_seed1.yaml` remains the requested 40,000-step recipe, unchanged.

## Changes

- `pyproject.toml`: explicit setuptools discovery limited to `equitraj` and its subpackages; declared core dependencies (PyTorch, e3nn, NumPy, SciPy, h5py, matplotlib, PyYAML) and optional simulation/OpenMM, train/TensorBoard and test/pytest dependencies. The supported e3nn range includes the actually installed/tested 0.4.4. OpenFF system rebuilding still needs the conda environment described in README.
- `configs/data/argon_full.yaml`: ten 500 ps production trajectories, 20 ps equilibration each, 5 fs MD steps, saved every step, CUDA mixed precision. Explicit reconstructed split: trajectories 0–7 train, 8 validation, 9 test. These settings restore the production volume, not an exact historical config. Seeds and other reconstruction choices follow `docs/RETRAIN_NOTES_oct29.md`. Output is `data/argon_full`, deliberately separate from completed reduced data: the generator skips completed files without checking their requested duration.
- `equitraj/evaluation.py`: moved the existing checkpoint loader and first-zero helper here; `scripts/evaluate.py` imports them. Both training and released checkpoints remain supported.
- `equitraj/multiseed.py` and `scripts/eval_multiseed.py`: evaluate one or more checkpoints with explicit paired RNG seeds and evenly spaced starts. One rollout is held in memory at a time. Compute temperature relative error, RDF L1 (Å), VACF first-zero relative error, diffusion relative error, crash fraction, and steps survived. JSON includes checkpoint SHA-256, protocol, ground-truth references, per-rollout observations/timing, missing counts, means and bootstrap intervals. Markdown summarizes the same statistics. Results are saved after each completed checkpoint.
- `tests/test_multiseed.py`: synthetic reference/identity metrics, reproducible cluster-bootstrap/missing values, start/seed pairing, crash exclusion and JSON/Markdown output.
- `colab/argon_full_retrain.ipynb`: complete GPU workflow, described below. README includes installation extras and evaluator usage.

## Statistical protocol and limits

Each physical metric is computed per rollout, then averaged. The formulas and reference sampling follow `scripts/evaluate.py`: test-reference temperature sampled every tenth model-stride frame; second-half model temperature/RDF/VACF; RDF 200 bins and the same range; VACF lag min(steps/4, 2 ps); full-rollout MSD with lag up to half the requested horizon and a slope fit over its final 75%. The new evaluator reports a mean of per-rollout absolute errors, rather than an error of a pooled curve. Those are different estimands.

Crashes use the existing rollout checks (nonfinite values, separation below 0.5 Å, or temperature above three times the test reference; checked every 10 steps and at the end). Steps survived follows the original evaluator's convention: first detected crash step, or the full requested budget. Physical metrics exclude crashed rollouts entirely; a missing VACF zero or undefined diffusion error is null, never fabricated or replaced by zero. All missing counts are reported.

The 95% CIs are percentile intervals from 2,000 bootstrap resamples of starting windows, retaining each window's seeds together. Bootstrap RNG seed is 1234; rollout seeds are 0 and 1 for every start, paired across checkpoints. Thus the intervals describe variation over windows conditional on the specified seed set, checkpoints, and test data; they do not measure training-seed uncertainty. Three starts are few, their windows can overlap, and all starts here come from a single held-out trajectory. Zero observed crashes yields a degenerate bootstrap interval, not proof of zero crash probability. A 50-step rollout is only 2.5 ps; the resulting short-time diffusion fit is not an asymptotic transport measurement. Neither checkpoint's current results certify the original 10,000-step release criteria. v0 is run as-is under the current in-softmax-envelope implementation.

## Installation and validation

Requested Python: `/workspace/venv311/bin/python` (3.11.16), with no pip module. An initial `uv pip install -p /workspace/venv311/bin/python -e .` failed because build isolation could not fetch setuptools (PyPI DNS unavailable). Installing the local `/usr/share/python-wheels/setuptools-78.1.1-py3-none-any.whl` into that environment then confirmed it is read-only. No permissions were escalated.

A writable temporary Python 3.11 environment was used instead, reusing the provided scientific libraries through a `.pth` entry (not `PYTHONPATH`). These commands succeeded:

```bash
UV_CACHE_DIR=/tmp/wigner_uv uv venv --python /workspace/venv311/bin/python /tmp/wigner_installcheck
UV_CACHE_DIR=/tmp/wigner_uv uv pip install -p /tmp/wigner_installcheck/bin/python \
  /usr/share/python-wheels/setuptools-78.1.1-py3-none-any.whl --offline
# Write /tmp/wigner_installcheck/lib/python3.11/site-packages/scientific_environment.pth:
# /workspace/venv311/lib/python3.11/site-packages
UV_CACHE_DIR=/tmp/wigner_uv uv pip install -p /tmp/wigner_installcheck/bin/python \
  -e . --offline --no-build-isolation --no-deps
UV_CACHE_DIR=/tmp/wigner_uv uv pip install -p /tmp/wigner_installcheck/bin/python \
  /usr/share/python-wheels/pip-25.1.1-py3-none-any.whl --offline
env -u PYTHONPATH /tmp/wigner_installcheck/bin/python -m pip install \
  -e . --no-build-isolation --no-index
```

The final **pip** command resolved all core dependencies (no `--no-deps` flag) and installed the editable package. `results/argon/multiseed_oct29b/pip_install.log` records success. `uv pip check` does not discover the libraries supplied by this `.pth` fallback and reported them missing; Python/pip do discover them. A direct importlib.metadata/packaging check validated every core version against the installed package metadata, recorded in `packaging_validation.json`. Versions: torch 2.10.0+cpu, e3nn 0.4.4, NumPy 1.26.4, SciPy 1.17.1, h5py 3.16.0, matplotlib 3.11.2, PyYAML 6.0.3.

An optional ordinary-wheel build with the distribution-provided setuptools wheel failed with its Debian `install_layout` patch (`build.log`); the successful editable builds above exercise the requested installation path. No repository workaround for that external setuptools patch was introduced. A clean networked environment can use ordinary `pip install -e '.[simulation,train,test]'` with isolated build dependencies; that fresh network installation was not executable in this sandbox.

From the repo root, without `PYTHONPATH`, both evaluator help commands and a real 8-step rollout succeeded:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl \
  /tmp/wigner_installcheck/bin/python scripts/eval_multiseed.py \
  --checkpoints checkpoints/argon_flow_v1_cpu_reduced.pt --data_dir data/argon \
  --starts 1 --seeds 1 --steps 8 --sampling_steps 1 --device cpu --threads 2 \
  --out results/argon/multiseed_oct29b/install_smoke
```

This installation smoke run took 5.706 s overall, including 4.58 s in rollout, and survived all 8 steps. Its very short-horizon metrics are retained separately and are not the comparison below.

Full suite:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENMM_CPU_THREADS=2 \
  MPLCONFIGDIR=/tmp/wigner_mpl /tmp/wigner_installcheck/bin/python -m pytest -q
```

**50 passed**, 5 deprecation warnings, 5.50 s; full output in `results/argon/multiseed_oct29b/pytest.log`. The notebook parses as nbformat-4 JSON and all seven Python code cells compile. `nbformat` is unavailable, so JSON parsing plus compilation was used, as permitted. Validation record: `notebook_validation.json`. Notebook CUDA execution has not been performed locally.

## CPU checkpoint comparison

Run from the repo root, two PyTorch/OMP/MKL threads. The main comparison began before the writable installation fallback was ready, so this command used `PYTHONPATH=.`; the separate actual rollout above verifies execution after installation without that variable.

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/eval_multiseed.py \
  --checkpoints checkpoints/argon_flow_v0.pt checkpoints/argon_flow_v1_cpu_reduced.pt \
  --data_dir data/argon --starts 3 --seeds 2 --steps 50 --sampling_steps 10 \
  --device cpu --threads 2 --out results/argon/multiseed_oct29b
```

Completed successfully: measured total wall time **719.681 s** (11.995 minutes), exit 0. Both checkpoints have six valid observations for every requested metric. All 12 rollouts survived all 50 steps.

| Metric | v0 mean [95% CI] | v1 CPU reduced mean [95% CI] |
|---|---:|---:|
| temperature_rel_err | 0.155856104 [0.154534595, 0.156826152] | 0.0742899661 [0.0597782901, 0.0833401927] |
| rdf_l1 | 0.532867189 [0.464519958, 0.613807998] | 0.214529791 [0.210474355, 0.218663774] |
| vacf_zero_rel_err | 0.0735785832 [0.040960953, 0.13469149] | 0.0928537901 [0.0313046721, 0.185594717] |
| D_rel_err | 0.0218144845 [0.00607052748, 0.0508791947] | 0.0576146792 [0.0423353062, 0.0702932989] |
| crash_fraction | 0 [0, 0] | 0 [0, 0] |
| steps_survived | 50 [50, 50] | 50 [50, 50] |

The three test starts are trajectory 3, model-frame indices 0, 132, 264 (saved-frame indices 0, 1320, 2640), each with RNG seeds 0 and 1. Ground-truth temperature, RDF and dynamics references are recomputed from `data/argon` at the checkpoint stride of 50 fs. The JSON includes checkpoint hashes and test-reference values.

On this short test, v1 has lower mean temperature and RDF errors, but higher VACF-zero and diffusion relative errors. This is not an isolated test of the attention change: training budgets and data differ. No physical metric was missing in this comparison.

[Full JSON](../results/argon/multiseed_oct29b/metrics.json), [Markdown report](../results/argon/multiseed_oct29b/metrics.md), [rollout log](../results/argon/multiseed_oct29b/run.log). Per-rollout wall times include shared-machine contention; they are not controlled hardware benchmarks.

## Exactly what remains to run on Colab

Open `colab/argon_full_retrain.ipynb`, select a GPU runtime (T4/L4/A100), and run all cells in order. No manual shell commands or local data upload are needed:

1. Mount Drive and use `/content/drive/MyDrive/WignerFlow_runs/argon_full_oct29` as the persistent run directory. Clone `https://github.com/yaniguan/WignerFlow.git`, checkout `feat/colab-oct29`, and install `.[simulation,train,test]`. Record package versions, source commit, GPU and test results on Drive.
2. Generate ten 500 ps production trajectories with OpenMM CUDA from `configs/data/argon_full.yaml`, directly onto Drive; validate completion and 100,000 saved frames per trajectory. Copy completed HDF5/config files locally for training and rebuild local NPY caches as needed. Do not reuse reduced `data/argon` files.
3. Load the **original** `configs/argon/B_S3a_seed1.yaml`; save a runtime copy changing only data/output paths and checkpoint interval (10 minutes). Keep original architecture, seed 1, batch 8, 40,000 steps, optimizer and unroll curriculum. Train on CUDA. Checkpoints (`ckpt.pt`, `best.pt`, every evaluated snapshot), TensorBoard and logs go directly to Drive. On interruption rerun the notebook to resume from `ckpt.pt`; RNG/DataLoader state is not restored bit-for-bit.
4. Optionally run the included GPU calibration cell (separate disposable model) for batch 8, unroll lengths 1/2/4 and rollout batches 1/4. It saves actual hardware timing projections. Markdown runtime estimates are explicitly assumptions grounded in the earlier CPU measurements, not measured Colab performance. If batch 8 runs out of memory, use a larger GPU rather than changing the requested training recipe.
5. After step 40,000, export `snapshots/step040000.pt` to `checkpoints/argon_flow_full_step40000.pt` under the Drive run. Copy v0 and checksum alongside it. Run `scripts/eval_multiseed.py` on v0 and the new checkpoint with full held-out data, 8 starts × 3 seeds × 1,000 steps, 10 flow steps, CUDA. Retain JSON, Markdown and evaluation log on Drive.
6. The final cell zips configs, checkpoints, metrics, timings and logs to `WignerFlow_runs/argon_full_oct29_results.zip` and offers a download. Large HDF5 data remain separately on Drive; local NPY caches are reproducible and excluded from the zip.

The notebook runtime estimate is 8–36 h T4, 4–20 h L4, or 2–12 h A100 plus data/evaluation, with substantial uncertainty and possible multiple sessions. Its justification is the earlier measured batch-1 CPU update timings (1.76–2.24 s), a crude eightfold batch extrapolation to 40,000 updates (156–200 CPU hours before m=4/validation overhead), and an explicitly assumed GPU speedup range. The calibration cell supplies a better estimate on the actual chosen GPU. No full-budget training result is claimed here.
