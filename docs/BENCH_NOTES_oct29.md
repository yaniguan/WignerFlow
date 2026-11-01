# Benchmark notes, 2026-10-29 (branch `feat/bench-oct29`)

> Status: run on 2026-10-29 (CPU, torch 2.10.0+cpu, 8 threads). Scripts need `PYTHONPATH=.` (or `pip install -e .`). Test suite: **45 passed, 1 skipped** (openmm absent).

## What was added

| File | Purpose |
|---|---|
| `equitraj/bench.py` | Shared helpers: random O(3) matrices, synthetic argon-like inputs, an untrained B_S3a-size model, equivariance error, timing and peak-RSS measurement (with subprocess isolation), and the cutoff-continuity scan. |
| `scripts/bench_equivariance.py` | Error of f(Rx+t) vs R f(x) for Δx and v_new: both heads × float32/float64 × rotation/reflection × vacuum/periodic. Can also run on the v0 release checkpoint (`--ckpt`). |
| `scripts/bench_speed_memory.py` | CPU forward (`model.loss`), backward and one prediction step: wall time and peak RSS vs number of atoms and number of context frames. Each setting runs in a fresh subprocess. |
| `scripts/bench_cutoff_continuity.py` | The model predicts no energies or forces, so there is no force-consistency or NVE check to run. Instead this scans one neighbor through the cutoff and tests whether the outputs are continuous there. |
| `tests/test_bench.py` | Tests for every helper. Tests that need the model skip without e3nn. |

Test-collection fix: `tests/test_data.py` now calls `pytest.importorskip` for `openmm` and `h5py`, and `tests/test_model.py` does the same for `e3nn`. Before, both files errored at collection. The pytest cache recorded both as failed, which points to the venv lacking openmm and probably e3nn. `tests/test_release.py` also skips without `e3nn` or `matplotlib`.

e3nn is pure Python. If it is missing, `pip install e3nn` brings back the model tests and makes the benchmarks runnable.

## Commands to run

```bash
source /workspace/venv311/bin/activate
pip install e3nn                 # only if missing
pytest -q tests/
python scripts/bench_equivariance.py --out results/bench/equivariance.json
python scripts/bench_equivariance.py --ckpt checkpoints/argon_flow_v0.pt --trials 3 --out results/bench/equivariance_ckpt.json
python scripts/bench_speed_memory.py --out results/bench/speed_memory.json
python scripts/bench_cutoff_continuity.py --out results/bench/cutoff_continuity.json
```

Defaults are sized for 8 CPU cores and about 3 GB of free RAM. The model is B_S3a size (64 channels, cutoff 6 Å, l_max 2). The speed sweep covers 64/128/256 atoms at k=3 and 1/2/3/5/8 frames at 64 atoms, batch 1. If a setting runs out of memory, it is reported as an error and the sweep continues.

## Results (measured)

**Equivariance** (`results/bench/equivariance*.json`, max abs error of dx / v under random O(3) ops, 128-atom liquid):
- Untrained, float64: ≤1.6e-13 (rotation and reflection, vacuum and periodic, both heads).
- Untrained, float32: ≤7.4e-05 abs (≤4.3e-05 rel).
- `argon_flow_v0.pt`, float32: dx max 1.75e-06, v max 5.0e-08; float64: ≤3e-15.

**Speed / memory** (flow head, 64 ch, cutoff 6 Å, batch 1):

| atoms | frames | edges/frame | fwd s | bwd s | predict s | peak MiB |
|---|---|---|---|---|---|---|
| 64 | 3 | 1212 | 0.075 | 0.112 | 0.121 | 535 |
| 128 | 3 | 2845 | 0.197 | 0.315 | 0.257 | 1153 |
| 256 | 3 | 5401 | 0.315 | 0.468 | 0.443 | 1867 |
| 64 | 1 | 1210 | 0.042 | 0.067 | 0.098 | 347 |
| 64 | 8 | 1216 | 0.167 | 0.264 | 0.177 | 1108 |

**Cutoff continuity** (float64, an atom pair crossing r_c; "max crossing" = max output change in the step that crosses the cutoff):
- On `main`: crossing jump stays ~8e-02 to 4.2e-01 as h goes 1e-2 → 1e-4 (relative jump 1e-2 to 2e-1), while control steps shrink linearly → **outputs are discontinuous at the cutoff**, as predicted below (envelope applied after softmax).
- Same script with `equitraj/layers.py` from the local `fix/cutoff-envelope` branch (envelope as log-bias inside softmax + zero-logit sink): crossing jump falls as ~h^3 (e.g. 5.7e-03 → 5.7e-06 → 5.7e-09), i.e. continuous. That fix is not on `main` yet.

## Prediction from reading the code (confirmed by the scan above)

`GraphAttention` (equitraj/layers.py) computes the softmax over all edges into a node and only then multiplies by the cutoff envelope. When an edge enters the cutoff, its own value is multiplied by envelope ≈ 0, so it adds almost nothing. But its logit is not suppressed, so it joins the softmax denominator with O(1) weight. That rescales the attention on every other edge into that node by a finite amount. **The outputs are therefore expected to jump when an atom crosses the cutoff**, despite the comment on `envelope` that says otherwise.

`bench_cutoff_continuity.py` is built to confirm or refute this. With a jump, the "max crossing" column stays the same size as h shrinks from 1e-2 to 1e-4 Å, while the other steps shrink like h. A common fix is to put the envelope inside the normalisation: α = env·exp(logit) / Σ env·exp(logit). That would change the trained model's behaviour, so it was not made here.
