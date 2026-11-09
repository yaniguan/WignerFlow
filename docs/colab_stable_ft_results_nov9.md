# WignerFlow Colab Results and Process Notes

Inventory date: 2026-11-09 (America/Los_Angeles)

## Repository and Branch State

- Repository: `yaniguan/WignerFlow`
- `main` commit: `3d650126703ba78556e2fc9ed8c1f9da5b0cf7d2`
- Open pull requests: this note is in open pull request #22. No other pull request is open.
- Branches kept because Colab notebooks reference them:
  - `cursor/resilient-drive-ckpt-save-0ccc`
  - `cursor/attn-sup-colab-nov4-c49b`
- Future restart rule (only if training is ever restarted later, not now):
  - GPU: A100 only. The Drive longest-session max of `40430` MiB is close to a 40 GB A100, so "A100 only" may require an 80 GB card, or a confirmed fit with `micro_batch_size` 4. This note does not add a new memory measurement, and it does not start training.
  - `BRANCH`: `main` or `cursor/resilient-drive-ckpt-save-0ccc`
  - `INIT_SNAPSHOT`: `step008000.pt` is the written choice for a new fine-tune from the original train snapshot. The notebook does not always use it. See the conflict below.

### Restart-rule conflict (notebook unchanged)

`colab/argon_stable_ft.ipynb` sets the form default `INIT_SNAPSHOT = 'auto'`. If `FT_NAME` already contains `B_S3a_stable_ft_runtime.yaml`, the notebook keeps the original `init_from` from that file and does not apply `INIT_SNAPSHOT`. Training then resumes from `ckpt.pt` when that file exists (`equitraj/train.py`). For the existing Drive run `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/`, `INIT_SNAPSHOT=step008000.pt` is ignored.

Yani needs to decide whether a later restart should continue the existing fine-tune (`ckpt.pt` around step 14000) or start a new `FT_NAME` from the original `MyDrive/WignerFlow_runs/argon_full_oct29/train/snapshots/step008000.pt`. Do not start training from this note. This pull request does not change the notebook.

"Step 8000" names two different checkpoints. One is the original run's `train/snapshots/step008000.pt`. The other is step 8000 of this fine-tune, where the Drive log reports `val_crash_frac = 1.0`.

## Run Location and Training Process

- Google Drive run directory: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/`
- Runtime config recorded in profiling summary (`MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`): `B_S3a_stable_ft_runtime.yaml`
- Initialization checkpoint (`wignerflow_training.log`): `/content/drive/MyDrive/WignerFlow_runs/argon_full_oct29/train/snapshots/step008000.pt` (`init_source_step: 8000`, `missing_keys: ["prior.rmin"]`)
- Primary training log: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (modified 2026-11-05 at 10:54 AM PST; copied locally as `wignerflow_training.log`)
- System setup recorded in `wignerflow_training.log`:
  - Element: `Ar` (256 Ar atoms per `profiling/summary.json`)
  - Model parameters (`n_params`): `1,945,520` (`docs/profiling/data.json` and `docs/profiling/README.md`; also the Drive log)
  - Training windows (`n_train_windows`): `799,520` (Drive `wignerflow_training.log` only; not in this repository)
  - Target: `residual` (`wignerflow_training.log`)
  - Repulsion prior: `repulsion_prior_rmin = 3.1533050537109375`, `r0 = 2.9956398010253906`, `overlap_penalty_r0 = 3.1533050537109375` (`wignerflow_training.log`)

### Unroll Curriculum and Session Resumes

From `wignerflow_training.log` (`MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log`):

1. Steps `0` to `3000`: trained with unroll length `m = 2` (resumed at steps `590` and `1710`).
2. Steps `3050` to `8000`: trained with unroll length `m = 3` (resumed at steps `3939`, `4395`, and `5225`).
3. At step `8000`: when transitioning to `m = 4` without micro-batching on a 39.49 GiB GPU, training hit two CUDA out-of-memory errors (`Tried to allocate 26.00 MiB` and `Tried to allocate 222.00 MiB` on `39.49 GiB` total capacity).
4. Steps `8000` to `14250`: resumed with `{"unroll_m": 4, "at_step": 8000, "micro_batch_size": 4}` and continued across multiple Colab sessions (resumed at steps `8757`, `9517`, `10255`, `10780`, `11798`, `12000`, `12788`, `13313`, and `13821`).
5. Last recorded training step: step `14250` with training `loss = 0.1335` (`0.1335360575467348`), `m = 4`, `lr = 3.0378420717485258e-06`, `grad_norm = 0.2348608708381653` (`wignerflow_training.log`).

## Best Checkpoint by selection_score (Step 14000)

Source: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (`wignerflow_training.log`). The metric values below are from that Drive log. They are not stored in this repository.

Step `14000` is the best checkpoint so far by `selection_score`, which is what writes `best_stable.pt`. It is not the best by `val_loss`. The table below shows step `0` has a lower `val_loss` (`0.03400` versus `0.05332` at step `14000`). In `equitraj/train.py`, `selection_score` ranks crash fraction first, then steps survived, then RDF L1 plus relative temperature error. When `sel_` keys are present, it uses those longer rollouts.

These Nov 9 numbers are validation rollouts from `configs/argon/B_S3a_stable_ft.yaml`: 8 x 1000 steps (`rollout_starts` / `rollout_steps`), and 8 x 2000 steps for selection (`rollout_starts_select` / `rollout_steps_select`). They are not the `RESULTS.md` test protocol of 15 x 10^4 steps.

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
- **Step 14000 (best by `selection_score`, not by `val_loss`)**: `val_crash_frac = 0.0`, `val_T_rel_err = 0.0223435096013214`, `val_rdf_l1 = 0.13151098594868632`, `val_loss = 0.05332` (`0.053320590709336105`). Step `0` still has the lower `val_loss`.

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
- Why it failed: the L4 has `22.03 GiB` (Drive `gpu.json` and `training.log`). The two larger memory figures below are different measures. Do not treat them as the memory of one micro-batched step.
  - Full-batch profiling, torch allocated: `peak_mem_GB = 50.20` (`50.195597312` in `results/profiling/a100_oct29/profile.log`). `scripts/profile_run.py` calls `profile_training` (`equitraj/profiling.py`), which runs `training_loss` on the full batch (`batch_size` 8, `m` 4) with no micro-batching. Training at `m = 4` uses `micro_batch_size` 4 (`configs/argon/B_S3a_stable_ft.yaml`). Do not read 50.20 GB as the memory of a micro-batched training step.
  - Longest training session on Drive: `40430` MiB max for `training_session_1793811383` in `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`. That figure is not in this repository. `40430` MiB is close to a 40 GB A100, so "A100 only" may require an 80 GB card, or a confirmed fit with micro-batching. This note does not invent a measured peak for `micro_batch_size` 4.

## Artifacts Not to Push to GitHub

Do not upload or commit large binary or trace artifacts to GitHub:

- `.pt` model checkpoints on Drive under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/` and `.../train/snapshots/` (including `best.pt`, `best_stable.pt`, `ckpt.pt`, `step008000.pt`, and `step014000.pt`).
- `trace.json` (~290 MB) under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/trace.json`.
