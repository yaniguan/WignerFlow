# GPU Profiling and Acceleration Analysis (WignerFlow, YLM, ouroboros-ocsr)

Inventory date: 2026-11-09 (America/Los_Angeles)

Reference presentation: [GPU Profiling, Explained (WignerFlow, OCSR, YLM)](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)

Note: No training, profiling jobs, or Colab sessions were started for this analysis. All figures below come strictly from the verified profiling files and logs on Google Drive and in this folder.

---

## 1. WignerFlow GPU Profiling Evidence

Sources:
- `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`
- `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/training.log` (local copy: `wignerflow_training.log`)
- Failing session `gpu.json` under `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/`

### Verified Profiling Metrics

| Metric | Value | Source |
| --- | --- | --- |
| Runtime config | `B_S3a_stable_ft_runtime.yaml` | `profiling/summary.json` |
| System size | `256` Ar atoms | `profiling/summary.json` |
| Model parameters (`n_params`) | `1,945,520` | `profiling/summary.json`, `wignerflow_training.log` |
| Training windows (`n_train_windows`) | `799,520` | `wignerflow_training.log` |
| Mean training step time (`step_ms_mean`) | `1906.6` ms | `profiling/summary.json` |
| Training time share — `forward` | `49.1%` | `profiling/summary.json` |
| Training time share — `backward` | `36.2%` | `profiling/summary.json` |
| Training time share — `neighbor_list` | `14.1%` | `profiling/summary.json` |
| Peak memory (`peak_mem_GB`) | `50.20` GB | `profiling/summary.json` |
| Short profile GPU utilization (`util_mean`) | `35.3%` | `profiling/summary.json` |
| Short profile max GPU memory (`mem max`) | `60988` MiB | `profiling/summary.json` |
| Longest session (`training_session_1793811383`) `util_mean` | `78.58%` | `profiling/summary.json` |
| Longest session (`training_session_1793811383`) `mem max` | `40430` MiB | `profiling/summary.json` |
| Rollout throughput | `5.97` steps/s (`25.81` ns/day) | `profiling/summary.json` |
| Rollout step latency (`step_ms`) | `167.4` ms | `profiling/summary.json` |
| Failing resume GPU & software | `NVIDIA L4` (`22.03` GiB total), `torch 2.11.0+cu130` | Failing session `gpu.json`, `wignerflow_training.log` |

*Artifact rule*: `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/trace.json` (~290 MB) must not be downloaded or pushed to GitHub. Keep only `summary.json` and markdown notes.

---

## 2. Why NVIDIA L4 Is the Wrong GPU for WignerFlow

The profiling data and training logs directly explain why the training resume at step `14000` crashed with `torch.OutOfMemoryError` on an NVIDIA L4:

1. **Memory demand vs. hardware capacity**:
   - Even with micro-batching enabled (`unroll_m = 4`, `micro_batch_size = 4`), the longest stable training session (`training_session_1793811383` in `profiling/summary.json`) required `40430 MiB` max GPU memory, and standard profiling recorded `peak_mem_GB = 50.20` (`60988 MiB` max in the short profile).
   - Earlier in `wignerflow_training.log` (at step `8000`), attempting `unroll_m = 4` prior to setting `micro_batch_size = 4` even exhausted a `39.49 GiB` GPU (`39.46 GiB` and `39.43 GiB` in use).
   - The failing session was assigned an **NVIDIA L4** (`gpu.json`), which only has `22.03 GiB` total memory (`21.95 GiB` in use, `21.62 GiB` allocated by PyTorch, `101.10 MiB` reserved unallocated when trying to allocate `74.00 MiB`).
2. **Why parameter count is small but activation memory is huge**:
   - Although the model has `1,945,520` parameters (`profiling/summary.json`), training unrolls the denoiser over `m = 4` trajectory steps (`unroll_m = 4`, `micro_batch_size = 4`) for `256` Ar atoms and retains intermediate equivariant spherical harmonic tensors across attention blocks (`self.attn` -> `self.so2_1(per_l_scale(xe, rad, self.lmax))` in `/content/WignerFlow/equitraj/layers.py`) for backpropagation.
   - As a result, activation memory—not model weights—drives the `40430 MiB` to `50.20` GB (`60988 MiB`) footprint. An L4 (`22.03 GiB`) cannot fit this computation graph.

---

## 3. Where Training and Rollout Time Goes

From `MyDrive/WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/summary.json`:

1. **Forward (`49.1%`) and Backward (`36.2%`) of `1906.6` ms/step**:
   - `forward` takes `49.1%` and `backward` takes `36.2%` of total training step time (`step_ms_mean = 1906.6` ms).
   - The stack trace in `wignerflow_training.log` shows that the inner loop runs `self.denoiser(cond, graph, y, tau)` over `n_steps` inside `predict_with_grad`, passing edge features through `SO(2)` equivariant attention layers (`equitraj/layers.py`, `per_l_scale` and `self.so2_1`).
2. **Neighbor List Construction (`14.1%` of `1906.6` ms/step)**:
   - `neighbor_list` accounts for `14.1%` of every training step (`profiling/summary.json`).
   - While smaller than the equivariant forward and backward passes, `14.1%` is a substantial secondary cost and can cause host-device synchronization stalls if graph edges are recomputed or filtered frequently.
3. **Startup / Short-Run Overhead vs. Steady-State Utilization**:
   - In the short profile, `util_mean` was `35.3%`, whereas in the longest steady session (`training_session_1793811383`), `util_mean` reached `78.58%` (`profiling/summary.json`). This shows that once compilation/warmup and data loading settle, the GPU is mostly compute/memory-bandwidth bound in the equivariant layers, with remaining time spent on neighbor-list rebuilds, Python dispatch, and checkpoint/validation I/O.
4. **Inference / Rollout Speed (`167.4` ms/step, `5.97` steps/s, `25.81` ns/day)**:
   - Without backward passes or multi-step unroll gradient accumulation, a single rollout step takes `167.4` ms (`profiling/summary.json`). During validation at step `14000` (`wignerflow_training.log`), `val_rollout_wall_s` is `337.40551018714905` s (for `1000.0` steps) and `sel_val_rollout_wall_s` is `673.5191538333893` s (for `2000.0` steps).

---

## 4. Practical Next Steps for Acceleration and Profiling

Do not run any of these changes now. When work resumes later, prioritize the following concrete, conservative steps.

### A. WignerFlow (`yaniguan/WignerFlow`)

1. **Hardware Requirement (Mandatory if training restarts later)**:
   - Use **NVIDIA A100 only** (to cover `40430 MiB` in `training_session_1793811383` and `peak_mem_GB = 50.20` / `60988 MiB` in `profiling/summary.json`).
   - Add a pre-flight check at the top of the Colab launcher/training script that checks `torch.cuda.get_device_properties(0).total_memory` and aborts immediately with a clear message if an NVIDIA L4 (`22.03 GiB`) is attached.
   - Follow the repository restart rule if training is ever restarted later: A100 only, `BRANCH` set to `main` or `cursor/resilient-drive-ckpt-save-0ccc`, and `INIT_SNAPSHOT` set to `step008000.pt`.
2. **Eliminate Unnecessary Intermediate Allocations in `per_l_scale` (`equitraj/layers.py`)**:
   - The L4 OOM occurred at `/content/WignerFlow/equitraj/layers.py`, line 199: `return x * torch.repeat_interleave(scale, reps, dim=1)`.
   - `torch.repeat_interleave` allocates a brand-new edge-by-channel tensor in both forward and backward passes for every attention block and denoiser step.
   - Replace `repeat_interleave` with precomputed index gather (`scale[:, l_index_map]`) or a fused broadcast multiply so `scale` is applied without allocating an extra copy of `x`-sized edge activations.
3. **Gradient / Activation Checkpointing Across Unroll or Denoiser Steps**:
   - Because peak memory reaches `50.20` GB (`profiling/summary.json`) and `40430 MiB` (`training_session_1793811383`) when `unroll_m = 4`, wrap individual transformer blocks (`blk(x, graph)` in `equitraj/model.py`, line 91) or denoiser sub-steps in `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`.
   - This reduces activation memory across unroll steps at the cost of recomputing forward activations during backward, allowing safer execution and larger micro-batch headroom on A100.
   - Also set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (recommended in the PyTorch OOM log in `wignerflow_training.log`) to prevent reserved-memory fragmentation.
4. **Neighbor List Caching / Verlet Skin (`14.1%` step time)**:
   - Since `neighbor_list` consumes `14.1%` of training step time (`profiling/summary.json`), verify whether the neighbor graph is being rebuilt on every denoiser sub-step (`tau`) when atomic positions `ctx_x` are fixed, or across small unroll displacements.
   - During `predict_with_grad`, the conditioning graph `graph` depends only on `ctx_x` and `box`—compute `graph` once per unroll step outside the `n_steps` denoiser loop if not already hoisted, and use a GPU Verlet list with skin buffer during validation rollouts (`val_rollout_wall_s`).
5. **Conservative Mixed Precision (`bfloat16` / `TF32` on A100)**:
   - Enable `torch.set_float32_matmul_precision("high")` (TF32 on A100 Tensor Cores) or test `bfloat16` autocast strictly inside the `SO(2)` linear/attention layers (`forward` `49.1%` and `backward` `36.2%`), while keeping coordinate updates, pair distances (`val_min_pair_dist`), and loss reductions in `float32` so rollout stability (`val_crash_frac = 0.0`) is preserved.

### B. YLM (`yaniguan/YLM`)

1. **Current State and Missing Profiling Numbers**:
   - Fine-grained kernel timing and peak VRAM numbers for YLM are not yet confirmed from the official file in this folder (high-level comparisons appear in the [GPU Profiling Google Slides deck](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)).
   - From the verified `full_C` run (`YLM_runs/options_bc_test/runs/full_C/results.json`), training ran for `60` epochs at `batch = 128` with `minilm` and achieved equivariance errors around `1e-6` (`test force MAE = 0.0048709`).
2. **Concrete Next Steps for YLM Profiling and Acceleration**:
   - **Preserve `1e-6` equivariance**: Before enabling `TF32` or `bfloat16` on spherical harmonic / tensor product operations (including the optional `0e+2e` tensor readout in open PR #20 after Yani reviews it), benchmark equivariance error directly. Keep spherical harmonics and Clebsch-Gordan / tensor readout contractions in `float32` if reduced precision raises equivariance error above `1e-6`.
   - **Precompute and cache frozen language embeddings**: For runs using `minilm` (`full_C`) or `matscibert` (`pol_s2` polarizability track), if text descriptions are static per sample and the text encoder is frozen, precompute and cache embeddings so GPU time goes to the geometric backbone.
   - **Lightweight `summary.json` profiler**: Add the same lightweight step-timer and peak memory summary logger used in WignerFlow (`summary.json` only, without saving large `trace.json` files) to measure forward, backward, and data-loader time shares on `full_C`.

### C. ouroboros-ocsr (`yaniguan/ouroboros-ocsr`)

1. **Current State and Missing Profiling Numbers**:
   - Detailed step-time and memory numbers for `ouroboros-ocsr` are not yet confirmed from the official file in this folder (experiments were run in the "Ouroboros — A100 experiments" Colab notebook and summarized in the [GPU Profiling Google Slides deck](https://docs.google.com/presentation/d/1mgss6yPPQWsCfZrW8gNVMuj1fQKeHVpxaVk3TnKSDPk)).
2. **Concrete Next Steps for OCSR Profiling and Acceleration**:
   - **Pin the small official metric file first**: Before running any new GPU benchmarks, locate the small official metric file on Drive for the `0.540` exact-match result so accuracy comparisons have a pinned baseline without downloading `ouroboros-real-aug-results.zip` (~833 MB).
   - **Separate training vs. decoding profiling**: In OCSR, CPU-heavy image augmentation ("real-aug") often bottlenecks training GPU utilization, while autoregressive SMILES/structure token generation bottlenecks validation. Log data-loading wait time (`data_ms`), encoder/decoder `forward`/`backward` time, and beam/greedy decoding throughput (molecules/sec) in a small `summary.json`.
   - **Batched evaluation and `bfloat16` on A100**: Use `bfloat16` mixed precision and KV-caching during evaluation decoding to reduce validation wall time on A100.
