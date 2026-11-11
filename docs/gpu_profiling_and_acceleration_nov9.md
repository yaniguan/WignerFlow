# GPU Profiling and Acceleration Analysis (WignerFlow, YLM, ouroboros-ocsr)

Inventory date: 2026-11-09 (America/Los_Angeles)

Reference presentation: [GPU Profiling, Explained (WignerFlow, OCSR, YLM)](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)

Note: No training, profiling jobs, or Colab sessions were started for this note. Numbers that appear in this repository are cited from `results/profiling/a100_oct29/profile.log` and `docs/profiling/data.json`. Numbers that appear only on Google Drive stay labeled as Drive.

---

## 1. WignerFlow GPU Profiling Evidence

Sources:
- In this repository: `results/profiling/a100_oct29/profile.log`, `docs/profiling/data.json`, `equitraj/profiling.py`, `scripts/profile_run.py`
- Drive only: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`
- Drive only: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (local copy name: `wignerflow_training.log`)
- Drive only: failing session `gpu.json` under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/`

### Verified Profiling Metrics

| Metric | Value | Source |
| --- | --- | --- |
| Runtime config | `B_S3a_stable_ft_runtime.yaml` | Notebook writes this filename (`colab/argon_stable_ft.ipynb`). Drive `profiling/summary.json` records it. That summary file is not in this repository. |
| System size | `256` Ar atoms | `results/profiling/a100_oct29/profile.log` and `docs/profiling/data.json` (`atoms` 256) |
| Model parameters (`n_params`) | `1,945,520` | `docs/profiling/data.json` and `docs/profiling/README.md` (`1945520`) |
| Training windows (`n_train_windows`) | `799,520` | Drive `wignerflow_training.log` only. Not in this repository. |
| Mean training step time (`step_ms_mean`) | `1906.6` ms | `results/profiling/a100_oct29/profile.log` (`1906.6029046333217`). Same run as Drive `profiling/summary.json`. Full batch, no micro-batching. |
| Training time share — `forward` | `49.1%` | `results/profiling/a100_oct29/profile.log` (fraction `0.4910540196073563`) |
| Training time share — `backward` | `36.2%` | `results/profiling/a100_oct29/profile.log` (fraction `0.36181834858037987`) |
| Training time share — `neighbor_list` | `14.1%` | `results/profiling/a100_oct29/profile.log` (fraction `0.14072576365579598`) |
| Peak memory (`peak_mem_GB`) | `50.20` GB | Full-batch profiling, torch allocated (`torch.cuda.max_memory_allocated` / 1e9). `results/profiling/a100_oct29/profile.log` has `50.195597312`. `scripts/profile_run.py` calls `profile_training` on the full batch (`batch_size` 8, `m` 4) with no micro-batching. Not a micro-batched training step. |
| Short profile GPU utilization (`util_mean`) | `35.3%` | `docs/profiling/data.json` nvidia-smi `util_mean` `35.325` |
| Short profile max GPU memory (`mem max`) | `60988` MiB | nvidia-smi `memory.used` max from the short profile (`docs/profiling/data.json` `mem_used_max_MiB` `60988.0`). This is not equal to `peak_mem_GB` 50.20. |
| Longest session (`training_session_1793811383`) `util_mean` | `78.58%` | Drive `profiling/summary.json` only. Not in this repository. |
| Longest session (`training_session_1793811383`) `mem max` | `40430` MiB | Drive `profiling/summary.json` only. Not in this repository. Close to a 40 GB A100. Not equal to `50.20` GB or `60988` MiB. |
| Rollout throughput | `5.97` steps/s (`25.81` ns/day) | `results/profiling/a100_oct29/profile.log` (`5.974730340945067` steps/s, `25.81083507288269` ns/day) |
| Rollout step latency (`step_ms`) | `167.4` ms | `results/profiling/a100_oct29/profile.log` (`167.37157042000035`) |
| Failing resume GPU & software | `NVIDIA L4` (`22.03` GiB total), `torch 2.11.0+cu130` | Drive failing-session `gpu.json` and `wignerflow_training.log`. Not in this repository. |

*Artifact rule*: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/trace.json` (~290 MB) must not be downloaded or pushed to GitHub. Keep only `summary.json` and markdown notes.

---

## 2. Why NVIDIA L4 Is the Wrong GPU for WignerFlow

The profiling data and training logs directly explain why the training resume at step `14000` crashed with `torch.OutOfMemoryError` on an NVIDIA L4:

1. **Memory demand vs. hardware capacity**:
   - Full-batch profiling, torch allocated: `peak_mem_GB = 50.20` (`50.195597312` in `results/profiling/a100_oct29/profile.log`). `profile_training` runs `training_loss` on the whole batch (`batch_size` 8, `m` 4). It does not call `accumulate_gradients`, so this number has no micro-batching. Training at `m = 4` uses `micro_batch_size` 4 (`configs/argon/B_S3a_stable_ft.yaml`). The 50.20 GB figure is only that full-batch profile.
   - Short-profile nvidia-smi max: `60988` MiB (`docs/profiling/data.json`, `mem_used_max_MiB`). This is `memory.used` from nvidia-smi. It is a different measure from torch `peak_mem_GB` 50.20. Do not write them as if they were equal.
   - Drive longest training session: `40430` MiB max for `training_session_1793811383` in `profiling/summary.json`. That figure is not in this repository. `40430` MiB is close to a 40 GB A100, so "A100 only" may require an 80 GB card, or a confirmed fit with micro-batching. This note does not invent a measured peak for `micro_batch_size` 4.
   - Earlier in the Drive log `wignerflow_training.log` (at step `8000`), attempting `unroll_m = 4` before `micro_batch_size = 4` exhausted a `39.49 GiB` GPU (`39.46 GiB` and `39.43 GiB` in use). That log is not in this repository. `configs/argon/B_S3a_stable_ft.yaml` says `m=4` with the full batch of 8 does not fit on an A100-40GB. `docs/profiling/README.md` says the 50.2 GB torch peak was on an A100-80GB and would not fit on a 40 GB card.
   - The failing session was assigned an **NVIDIA L4** (Drive `gpu.json`), which only has `22.03 GiB` total memory (`21.95 GiB` in use, `21.62 GiB` allocated by PyTorch, `101.10 MiB` reserved unallocated when trying to allocate `74.00 MiB`). Those L4 figures are from the Drive log.
2. **Why parameter count is small but activation memory is huge**:
   - The model has `1,945,520` parameters (`docs/profiling/data.json` and `docs/profiling/README.md`). Training unrolls the denoiser over `m = 4` trajectory steps for `256` Ar atoms and keeps intermediate equivariant tensors across attention blocks (`self.attn` -> `self.so2_1(per_l_scale(xe, rad, self.lmax))` in `equitraj/layers.py`) for backpropagation. At `m = 4`, training uses `micro_batch_size` 4. The 50.20 GB profile did not.
   - Activation memory, not the parameter count, is what makes the full-batch profile large. The three memory figures stay separate: torch allocated `50.20` GB (full batch, no micro-batching), nvidia-smi short-profile max `60988` MiB, and Drive longest-session max `40430` MiB. An L4 (`22.03 GiB`, Drive) cannot fit the full-batch graph. This repository does not contain a measured peak for the micro-batched path.

---

## 3. Where Training and Rollout Time Goes

The step-time shares below are from the full-batch profile in `results/profiling/a100_oct29/profile.log` (`step_ms_mean` `1906.6029046333217`). That profile does not use micro-batching.

1. **Forward (`49.1%`) and Backward (`36.2%`) of `1906.6` ms/step**:
   - `forward` takes `49.1%` and `backward` takes `36.2%` of that full-batch profile step (`step_ms_mean = 1906.6` ms).
   - The Drive OOM stack trace in `wignerflow_training.log` shows the inner loop runs `self.denoiser(cond, graph, y, tau)` over `n_steps` inside `predict_with_grad`, passing edge features through `SO(2)` equivariant attention layers (`equitraj/layers.py`, `per_l_scale` and `self.so2_1`). That log is not in this repository. The same call order is in `equitraj/model.py`.
2. **Neighbor List Construction (`14.1%` of `1906.6` ms/step)**:
   - `neighbor_list` accounts for `14.1%` of the full-batch profile step (`results/profiling/a100_oct29/profile.log`). This is not a separate measurement of every micro-batched training step.
   - While smaller than the equivariant forward and backward passes, `14.1%` is a substantial secondary cost and can cause host-device synchronization stalls if graph edges are recomputed or filtered frequently.
3. **Startup / Short-Run Overhead vs. Steady-State Utilization**:
   - In the short profile, nvidia-smi `util_mean` was `35.3%` (`35.325` in `docs/profiling/data.json`). The Drive longest session (`training_session_1793811383`) reports `util_mean` `78.58%` in `profiling/summary.json`. That `78.58%` figure is not in this repository.
4. **Inference / Rollout Speed (`167.4` ms/step, `5.97` steps/s, `25.81` ns/day)**:
   - Without backward passes or multi-step unroll gradient accumulation, a single rollout step takes `167.4` ms (`results/profiling/a100_oct29/profile.log`). During validation at step `14000` (Drive `wignerflow_training.log`), `val_rollout_wall_s` is `337.40551018714905` s (for `1000.0` steps) and `sel_val_rollout_wall_s` is `673.5191538333893` s (for `2000.0` steps). Those Nov 9 validation numbers are validation rollouts (8 x 1000 / 8 x 2000 steps in `configs/argon/B_S3a_stable_ft.yaml`), not the `RESULTS.md` 15 x 10^4-step test protocol. Step `14000` is best by `selection_score` (`best_stable.pt`), not by `val_loss`.

---

## 4. Practical Next Steps for Acceleration and Profiling

Do not run any of these changes now. When work resumes later, prioritize the following concrete, conservative steps.

### A. WignerFlow (`yaniguan/WignerFlow`)

1. **Hardware Requirement (Mandatory if training restarts later)**:
   - Use an NVIDIA A100 if training is ever restarted later. Do not start training now. The full-batch profile (`peak_mem_GB` 50.20, torch allocated; short-profile nvidia-smi max `60988` MiB) was recorded on an A100-80GB (`docs/profiling/data.json`). Those two numbers are different measures. `docs/profiling/README.md` says that profile would not fit on a 40 GB card. The Drive longest-session max of `40430` MiB is close to a 40 GB A100, so "A100 only" may require an 80 GB card, or a confirmed fit with `micro_batch_size` 4.
   - Add a pre-flight check at the top of the Colab launcher/training script that checks `torch.cuda.get_device_properties(0).total_memory` and aborts immediately with a clear message if an NVIDIA L4 (`22.03 GiB`, Drive log) is attached. This pull request does not change the notebook.
   - Restart rule, only if training is ever restarted later: A100 only (see the memory caveat above), `BRANCH` set to `main` or `cursor/resilient-drive-ckpt-save-0ccc`. The notebook default is `INIT_SNAPSHOT = 'auto'`. If `FT_NAME` already has `B_S3a_stable_ft_runtime.yaml`, `colab/argon_stable_ft.ipynb` keeps the original `init_from` and resumes from `ckpt.pt`, so `INIT_SNAPSHOT=step008000.pt` is ignored. Yani needs to decide whether a later restart should continue the existing fine-tune (`ckpt.pt` around step 14000) or start a new `FT_NAME` from the original `train/snapshots/step008000.pt`. Do not change the notebook in this docs pull request.
2. **Eliminate Unnecessary Intermediate Allocations in `per_l_scale` (`equitraj/layers.py`)**:
   - The L4 OOM occurred at `/content/WignerFlow/equitraj/layers.py`, line 199: `return x * torch.repeat_interleave(scale, reps, dim=1)`.
   - `torch.repeat_interleave` allocates a brand-new edge-by-channel tensor in both forward and backward passes for every attention block and denoiser step.
   - Replace `repeat_interleave` with precomputed index gather (`scale[:, l_index_map]`) or a fused broadcast multiply so `scale` is applied without allocating an extra copy of `x`-sized edge activations.
3. **Gradient / Activation Checkpointing Across Unroll or Denoiser Steps**:
   - The full-batch profile reaches `peak_mem_GB` 50.20 with no micro-batching. The Drive longest session reports `40430` MiB. Those are different measures. If memory is still tight on a later A100 run, wrap individual transformer blocks (`blk(x, graph)` in `equitraj/model.py`, line 91) or denoiser sub-steps in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`. Do not run that change now.
   - This reduces activation memory across unroll steps at the cost of recomputing forward activations during backward, allowing safer execution and larger micro-batch headroom on A100.
   - Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recommended in the PyTorch OOM log in `wignerflow_training.log`) to prevent reserved-memory fragmentation.
4. **Neighbor List Caching / Verlet Skin (`14.1%` step time)**:
   - `neighbor_list` is `14.1%` of the full-batch profile step (`results/profiling/a100_oct29/profile.log`). Check whether any other path rebuilds the neighbor graph on every denoiser sub-step (`tau`) when atomic positions are fixed.
   - `EquiTraj.predict_with_grad` already calls `encode` once, before the `n_steps` denoiser loop (`equitraj/model.py`). The conditioning graph is already built outside that loop. Do not add a second hoist. A GPU Verlet list with a skin buffer during validation rollouts is still an open idea. Do not run it now.
5. **Conservative Mixed Precision (`bfloat16` / `TF32` on A100)**:
   - Enable `torch.set_float32_matmul_precision("high")` (TF32 on A100 Tensor Cores) or test `bfloat16` autocast strictly inside the `SO(2)` linear/attention layers (`forward` `49.1%` and `backward` `36.2%`), while keeping coordinate updates, pair distances (`val_min_pair_dist`), and loss reductions in `float32` so rollout stability (`val_crash_frac = 0.0`) is preserved.

### B. YLM (`yaniguan/YLM`)

This section is out of scope for the WignerFlow repository. It is kept only so the earlier wrong equivariance sentence is corrected. Do not treat it as a WignerFlow result.

1. **Current State and Missing Profiling Numbers**:
   - Fine-grained kernel timing and peak VRAM numbers for YLM are not in this repository (high-level comparisons appear in the [GPU Profiling Google Slides deck](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)).
   - Equivariance error and force MAE are different metrics. `yaniguan/YLM` `docs/results_options_abc.md` says mean force equivariance error is about `2e-6`. The same file lists Option C force MAE as `0.00487`. That file is not in this WignerFlow checkout. Do not quote equivariance as `1e-6`.
   - The Drive file `YLM_runs/options_bc_test/runs/full_C/results.json` is not in this repository. An earlier draft of this note cited `test force MAE = 0.0048709` from that file and called it an equivariance error. Keep `0.0048709` labeled as that Drive force MAE only. It is not the equivariance error.
2. **Concrete Next Steps for YLM Profiling and Acceleration**:
   - **Preserve about `2e-6` equivariance**: Before enabling `TF32` or `bfloat16` on spherical harmonic / tensor product operations (including the optional `0e+2e` tensor readout in YLM pull request #20 after Yani reviews it), benchmark equivariance error directly. Keep spherical harmonics and Clebsch-Gordan / tensor readout contractions in `float32` if reduced precision raises equivariance error above about `2e-6`. Do not use force MAE `0.00487` as that threshold.
   - **Precompute and cache frozen language embeddings**: For runs using `minilm` (`full_C`) or `matscibert` (`pol_s2` polarizability track), if text descriptions are static per sample and the text encoder is frozen, precompute and cache embeddings so GPU time goes to the geometric backbone.
   - **Lightweight `summary.json` profiler**: Add the same lightweight step-timer and peak memory summary logger used in WignerFlow (`summary.json` only, without saving large `trace.json` files) to measure forward, backward, and data-loader time shares on `full_C`.

### C. ouroboros-ocsr (`yaniguan/ouroboros-ocsr`)

This section is out of scope for the WignerFlow repository. It is kept as a pointer only. The numbers below are not WignerFlow results.

1. **Current State and Missing Profiling Numbers**:
   - Detailed step-time and memory numbers for `ouroboros-ocsr` are not yet confirmed from the official file in this folder (experiments were run in the "Ouroboros — A100 experiments" Colab notebook and summarized in the [GPU Profiling Google Slides deck](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)).
2. **Concrete Next Steps for OCSR Profiling and Acceleration**:
   - **Pin the small official metric file first**: Before running any new GPU benchmarks, locate the small official metric file on Drive for the `0.540` exact-match result so accuracy comparisons have a pinned baseline without downloading `ouroboros-real-aug-results.zip` (~833 MB).
   - **Separate training vs. decoding profiling**: In OCSR, CPU-heavy image augmentation ("real-aug") often bottlenecks training GPU utilization, while autoregressive SMILES/structure token generation bottlenecks validation. Log data-loading wait time (`data_ms`), encoder/decoder `forward`/`backward` time, and beam/greedy decoding throughput (molecules/sec) in a small `summary.json`.
   - **Batched evaluation and `bfloat16` on A100**: Use `bfloat16` mixed precision and KV-caching during evaluation decoding to reduce validation wall time on A100.
