# WignerFlow Colab Results and Process Notes

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository and Branch State

- Repository: `yaniguan/WignerFlow`
- `main` commit: `3d650126703ba78556e2fc9ed8c1f9da5b0cf7d2`
- Open pull requests: none
- Branches kept because Colab notebooks reference them:
  - `cursor/resilient-drive-ckpt-save-0ccc`
  - `cursor/attn-sup-colab-nov4-c49b`
- Future restart rule (only if training is ever restarted later, not now):
  - GPU: A100 only
  - `BRANCH`: `main` or `cursor/resilient-drive-ckpt-save-0ccc`
  - `INIT_SNAPSHOT`: `step008000.pt`

## Run Location and Training Process

- Google Drive run directory: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/`
- Runtime config recorded in profiling summary (`MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`): `B_S3a_stable_ft_runtime.yaml`
- Initialization checkpoint (`wignerflow_training.log`): `/content/drive/MyDrive/WignerFlow_runs/argon_full_oct29/train/snapshots/step008000.pt` (`init_source_step: 8000`, `missing_keys: ["prior.rmin"]`)
- Primary training log: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (modified 2026-11-05 at 10:54 AM PST; copied locally as `wignerflow_training.log`)
- System setup recorded in `wignerflow_training.log`:
  - Element: `Ar` (256 Ar atoms per `profiling/summary.json`)
  - Model parameters (`n_params`): `1,945,520` (`wignerflow_training.log` and `profiling/summary.json`)
  - Training windows (`n_train_windows`): `799,520` (`wignerflow_training.log`)
  - Target: `residual` (`wignerflow_training.log`)
  - Repulsion prior: `repulsion_prior_rmin = 3.1533050537109375`, `r0 = 2.9956398010253906`, `overlap_penalty_r0 = 3.1533050537109375` (`wignerflow_training.log`)

### Unroll Curriculum and Session Resumes

From `wignerflow_training.log` (`MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log`):

1. Steps `0` to `3000`: trained with unroll length `m = 2` (resumed at steps `590` and `1710`).
2. Steps `3050` to `8000`: trained with unroll length `m = 3` (resumed at steps `3939`, `4395`, and `5225`).
3. At step `8000`: when transitioning to `m = 4` without micro-batching on a 39.49 GiB GPU, training hit two CUDA out-of-memory errors (`Tried to allocate 26.00 MiB` and `Tried to allocate 222.00 MiB` on `39.49 GiB` total capacity).
4. Steps `8000` to `14250`: resumed with `{"unroll_m": 4, "at_step": 8000, "micro_batch_size": 4}` and continued across multiple Colab sessions (resumed at steps `8757`, `9517`, `10255`, `10780`, `11798`, `12000`, `12788`, `13313`, and `13821`).
5. Last recorded training step: step `14250` with training `loss = 0.1335` (`0.1335360575467348`), `m = 4`, `lr = 3.0378420717485258e-06`, `grad_norm = 0.2348608708381653` (`wignerflow_training.log`).

## Best Validation Metrics (Step 14000)

Source: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`)

Step `14000` is the best validation checkpoint so far:

| Metric | Value at Step 14000 | Source File |
| --- | --- | --- |
| `val_loss` | `0.05332` (`0.053320590709336105`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_crash_frac` | `0.0` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_T_rel_err` | `0.0223435096013214` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_rdf_l1` | `0.13151098594868632` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_T_end` | `91.90` (`91.89904698665008`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_abs_T_drift_K_per_ns` | `59.73` (`59.73464322976534`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `val_min_pair_dist` | `3.049` (`3.0491205252109816`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_T_rel_err` | `0.04897` (`0.04896719243683184`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_rdf_l1` | `0.14243` (`0.1424267756437849`) | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_crash_frac` | `0.0` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_T_end` | `86.93469406854837` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_abs_T_drift_K_per_ns` | `64.27373374558736` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |
| `sel_val_min_pair_dist` | `3.071331234133949` | `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`) |

## Validation History Across Steps 0 to 14000

Source: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`)

- **Step 0**: `val_crash_frac = 0.0`, `val_T_rel_err = 0.248` (`0.2483274536040686`), `val_rdf_l1 = 0.231` (`0.23051022147454175`), `val_loss = 0.033998754364438354`.
- **Steps 2000, 4000, 6000**: worse than the later recovery (`val_crash_frac = 0.0`, with elevated temperature drift and RDF error).
- **Steps 8000 and 10000**: `val_crash_frac = 1.0` (temperature crashes, `val_crash_temperature = 8.0`).
- **Step 12000 (recovered)**: `val_crash_frac = 0.0`, `val_T_rel_err = 0.0663` (`0.06631851617837924`), `val_rdf_l1 = 0.1393` (`0.13925397951888302`), `val_loss = 0.07229247498326004`.
- **Step 14000 (best so far)**: `val_crash_frac = 0.0`, `val_T_rel_err = 0.0223435096013214`, `val_rdf_l1 = 0.13151098594868632`, `val_loss = 0.05332` (`0.053320590709336105`).

Full table from `wignerflow_training.log`:

| Step | `val_loss` | `val_crash_frac` | `val_T_rel_err` | `val_rdf_l1` | `val_T_end` | `val_abs_T_drift_K_per_ns` | `val_min_pair_dist` | `sel_val_T_rel_err` | `sel_val_rdf_l1` | Source |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `0` | `0.03400` | `0.0` | `0.24833` | `0.23051` | `118.91` | `523.33` | `3.013` | `0.34397` | `0.28263` | `wignerflow_training.log` |
| `2000` | `0.08642` | `0.0` | `1.13765` | `0.53383` | `217.80` | `2675.44` | `2.929` | `1.47960` | `0.69278` | `wignerflow_training.log` |
| `4000` | `0.08399` | `0.0` | `0.45239` | `0.47182` | `41.91` | `986.69` | `3.082` | `0.65362` | `0.85499` | `wignerflow_training.log` |
| `6000` | `0.08617` | `0.0` | `0.69546` | `0.38911` | `172.55` | `1654.21` | `2.961` | `1.06487` | `0.54611` | `wignerflow_training.log` |
| `8000` | `0.08378` | `1.0` | crashed (`302.5` steps) | crashed | crashed | crashed | `2.885` | crashed (`302.5` steps) | crashed | `wignerflow_training.log` |
| `10000` | `0.09599` | `1.0` | crashed (`268.75` steps) | crashed | crashed | crashed | `2.883` | crashed (`262.5` steps) | crashed | `wignerflow_training.log` |
| `12000` | `0.07229` | `0.0` | `0.06632` | `0.13925` | `100.60` | `124.81` | `3.032` | `0.08471` | `0.14369` | `wignerflow_training.log` |
| `14000` | `0.05332` | `0.0` | `0.02234` | `0.13151` | `91.90` | `59.73` | `3.049` | `0.04897` | `0.14243` | `wignerflow_training.log` |

## CUDA Out-of-Memory (OOM) on NVIDIA L4

After logging train step `14250` (`loss = 0.1335`), a later session resumed from snapshot step `14000` (`{"unroll_m": 4, "at_step": 14000, "micro_batch_size": 4}`) on an **NVIDIA L4** GPU (`torch 2.11.0+cu130`, recorded in the failing session `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/gpu.json`).

From `wignerflow_training.log` (`MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log`):

- Error: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 74.00 MiB. GPU 0 has a total capacity of 22.03 GiB of which 75.12 MiB is free. Including non-PyTorch memory, this process has 21.95 GiB memory in use. Of the allocated memory 21.62 GiB is allocated by PyTorch, and 101.10 MiB is reserved by PyTorch but unallocated.`
- Stack trace location: inside `accumulate_gradients` -> `training_loss` -> `model.predict_with_grad` -> `self.denoiser` -> `self.attn` -> `self.so2_1(per_l_scale(xe, rad, self.lmax))` at `/content/WignerFlow/equitraj/layers.py`, line 199 (`return x * torch.repeat_interleave(scale, reps, dim=1)`).
- Why it failed: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json` shows `peak_mem_GB = 50.20` during standard profiling and `40430 MiB` max memory in the longest training session (`training_session_1793811383`). An NVIDIA L4 only has `22.03 GiB` of GPU memory, which cannot fit the `unroll_m = 4`, `micro_batch_size = 4` equivariant attention activation graph.

## Artifacts Not to Push to GitHub

Do not upload or commit large binary or trace artifacts to GitHub:

- `.pt` model checkpoints on Drive under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/` and `.../train/snapshots/` (including `best.pt`, `best_stable.pt`, `ckpt.pt`, `step008000.pt`, and `step014000.pt`).
- `trace.json` (~290 MB) under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/trace.json`.
