# GPU Profiling: Full Analysis (WignerFlow / ouroboros-ocsr / YLM, Colab runs)

> Generated: 2026-11-03 (Pacific Time). Every number comes from the raw files on Google Drive (yaniguan2). The raw file IDs are in `data.json` / `docs/profiling/manifest.json`.
> Data we could not find is marked **missing**. No number is made up or estimated.

## 0. One-page summary

| Run | Throughput | Time per step | GPU utilization (nvidia-smi mean) | Peak memory | Where the bottleneck is |
|---|---|---|---|---|---|
| WignerFlow argon `train_stable_ft` (A100-80GB, unroll m=4, bs=8, 256 atoms) | Training 0.524 step/s; rollout 5.97 step/s = **25.8 ns/day** | Training 1906.6 ms (median 1951.9); rollout 167.4 ms | 35.3% (whole profile process; 51% of samples are <10%) | **50.2 GB** (smi max 60,988 MiB) | **Mixed: host-side Python/kernel-launch overhead + FP32 GEMM**. In the profiler window, self CPU is 8.38 s vs self CUDA 4.08 s (2.1×). The GPU has only about 1.36 s of kernel time per step, while the GPU-side step is about 2.27 s → about 60% busy. `ampere_sgemm_*` shows matmuls run as FP32 SIMT (TF32 is off) |
| ocsr A (finetune_v2, bf16, bs=32) | Probe 425.8 img/s; full training median 439.2 img/s | 75.1 ms (forward 21.9 / backward 45.9 / optimizer 4.8 / H2D 1.8 / data 0.7) | Full training 49.9% (max 84) | 2.86 GB | **Kernel-launch / small-kernel bound**: in the trace the GPU is busy only 41.7% of the time; 7,615 kernels over 5 steps, 30.8 µs each on average |
| ocsr C (finetune_v2, bf16) | Probe 99.1 img/s; full training 94.1 img/s | 326.9 ms (forward 96.0 / backward 220.5) | Probe 73.2%, full training 63.8% (max 92) | 4.20 GB | **Mostly GPU compute**, but the GPU is idle about 1/3 of the time (trace is 152 MB and could not be downloaded → missing) |
| YLM full_A (2 segments) | 1,398–1,401 samples/s | 91–99 ms | 34–35% | 1.48 GB | **CPU / kernel-launch bound**: self CPU 5.63 s vs self CUDA 0.78 s (**7.2×**); 38,398 `cudaLaunchKernel` calls in 20 steps |
| YLM full_B (4 segments) | 778–1,270 samples/s | 104–114 ms | 25–33% | 1.45–1.48 GB | **CPU / kernel-launch bound**: 6.35 s vs 0.81 s (**7.8×**); 41,558 launches |
| YLM full_C (6 segments) | 413–452 samples/s | 284–286 ms (backward about 213 ms = 75%) | 75–81% | 11.4 GB | **GPU memory-bandwidth bound (elementwise ops)**: `SliceBackward0` takes 3.27 s of 4.90 s CUDA time (67%); `aten::add_` 2.38 s (49%), `fill_` 0.79 s; `slice_backward` allocated 1,386.7 GB of zero tensors in total |

Charts: `charts/` (this folder) (per repo: `*_step_breakdown.png`, `*_top_cuda.png`; ocsr A trace: `ocsr_A_probe_trace_kernels.png`; cross-repo: `cross_gpu_util.png`, `cross_cpu_vs_cuda.png`, `cross_peak_memory.png`).

---

## 1. How the profiling data was collected (code paths)

All three repos use the same approach, in three layers:

**(a) Phase timers (step-time breakdown)**: call `torch.cuda.synchronize()` before and after each phase, then record time with `time.perf_counter()`. CUDA is asynchronous. Without the sync you only measure when kernels are *submitted*, not how long they run.
- WignerFlow: `equitraj/profiling.py` → `PhaseTimer.phase()` (line 71), `_sync()` (line 36); `profile_training()` records data / neighbor_list / forward / backward / optimizer; `time_neighbor_list()` wraps the neighbor-list build; `rollout_speed()` computes steps/s and ns/day (`rates()`: ns/day = steps/s × stride_fs × 86400 / 1e6). Entry point: `scripts/profile_run.py`.
- ouroboros-ocsr: `scripts/profile_train.py` → `timed()` (lines 113–119, times after a sync), split into data_loading / host_to_device / forward (bf16 autocast) / backward / optimizer; `scripts/profile_infer.py` uses a forward hook to time the encoder and decode separately.
- YLM: `Profiler.phase()` in `src/profiling.py`. It **syncs only for the first `sync_steps=200` steps** (to split phases), and does not sync after that, so training is not slowed down. Throughput = wall-clock time between steps (validation not included). It is hooked in at lines 273–297 of `scripts/04_train.py`. Each `--resume` renames the old folder to `profiling_session_<ns>`, so each resumed run is one segment (seg).

**(b) torch.profiler (op/kernel level)**: `torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True, ...)`, which exports a Chrome trace (`trace.json`) and `key_averages().table()` (`key_averages.txt`).
- WignerFlow `torch_profile()` (`equitraj/profiling.py` line 283): 1 warmup step + 3 recorded steps.
- ocsr: 5 trace steps (`profile_train.py` lines 171–176, `profile_memory=True`).
- YLM: the first `trace_steps=20` steps.

**(c) GPU telemetry**: a background thread calls `nvidia-smi --query-gpu=utilization.gpu,...` on a timer.
- WignerFlow: `gpu_samples.csv` (default interval 1 s, `--gpu-interval`), parsed in `parse_nvidia_smi()`.
- ocsr: `nvidia-smi ... -lms`, about one sample every 200 ms, also with utilization.memory and power.draw → `nvidia_smi_*.csv`.
- YLM: about once every 1 s → `gpu_utilization.jsonl` (the first segment of full_A had a 2 s `nvidia-smi` timeout).

**Memory**: `torch.cuda.max_memory_allocated()` (ocsr also records `max_memory_reserved`).

---

## 2. How to analyze it step by step (reproducible: `scripts/analyze_profiling.py`)

1. **Look at throughput and time per step first** (summary.json) to set a baseline.
2. **Look at the phase breakdown**: if the data phase is >20% of the step, the input is the bottleneck (DataLoader/CPU). In all three repos data is <3 ms, so **none of them is input bound**.
3. **Look at GPU utilization**: nvidia-smi `utilization.gpu` means "the fraction of the sample window in which at least one kernel was running". Low (<50%) means the GPU is waiting for the host. High (>75%) only means the GPU always has work; it **does not mean the compute is fully used**.
4. **Compare host time and device time**: `Self CPU time total` and `Self CUDA time total` at the end of `key_averages.txt`. CPU ≫ CUDA together with low GPU utilization → Python/launch bound.
5. **Look at gaps in the trace**: merge the time intervals of all kernels to get the GPU busy fraction (`trace_gpu_busy()`). Many kernels with an average length of only tens of µs → launch bound.
6. **Look at the top kernels**: `sgemm` means FP32 (no Tensor Cores); `*_elementwise_*`/`add_`/`fill_`/`copy_` are memory-bandwidth ops; `nchwToNhwc` is a memory-layout conversion; `multi_tensor_apply` is the optimizer.
7. **Roofline**: the raw files **do not record FLOP counts**, so we cannot give a real arithmetic intensity or roofline (**missing**). We can only judge by kernel type: GEMM is compute-heavy, elementwise ops are bandwidth-heavy.
8. Give a verdict by rules (`verdict()`), with evidence attached to each verdict.

How to run: `python scripts/analyze_profiling.py --raw <raw folder> --out <output folder>`; tests: `pytest tests/test_analyze_profiling.py`.

---

## 3. How the verdicts are made (rules)

| What you see | Verdict |
|---|---|
| Data phase >20% | Input / DataLoader bound → more `num_workers`, `pin_memory`, `prefetch` |
| GPU utilization <50%, and self CPU / self CUDA >2 | CPU / Python / kernel-launch bound → `torch.compile`, CUDA Graphs, bigger batch, fused ops |
| GPU busy <60% in the trace, many short kernels | Small-kernel / launch bound |
| GPU utilization ≥75%, mostly GEMM/conv | Compute bound → bf16/TF32, better kernels |
| GPU utilization ≥60%, elementwise ops >40% | Memory-bandwidth bound → remove extra allocations/copies, fuse ops |
| Peak memory close to card capacity | Memory-capacity bound → activation checkpointing, smaller batch/unroll |

---

## 4. Results and bottlenecks per run

### 4.1 WignerFlow — argon_full_oct29/train_stable_ft (A100-SXM4-80GB, torch 2.11.0+cu130, 1,945,520 parameters)
- **Training** (30 steps, warmup 3, bs=8, m=4, unroll_sampling_steps=2, truncated backprop): 1906.6 ms per step, 0.524 step/s. Phase shares: forward 936.1 ms (49.1%), backward 689.8 ms (36.2%), **neighbor list 268.3 ms (14.1%, 16 calls per step)**, optimizer 9.9 ms, data 2.3 ms.
- **Peak memory 50.2 GB**; nvidia-smi max 60,988 MiB (so this is an 80 GB card; it would not fit on a 40 GB card).
- **Profiler** (3 steps): self CUDA 4.079 s, self CPU 8.382 s. Top items: `aten::bmm` 1318 ms (32.3%), `ampere_sgemm_128x128_nt` 1062 ms (26.0%), `aten::mm` 499 ms, `addmm` 348 ms, `gemmSN_NN` 236 ms; index / index_put (scatter/gather) about 440 ms in total; `aten::copy_` CPU total 1.65 s.
- **Rollout**: 5.97 step/s, 167.4 ms per step, **25.81 ns/day** (batch 1, 256 atoms, k=3, 10 sampling iterations per step → about 16.7 ms per denoiser evaluation); neighbor list 2.52 ms per call (estimated 6.0%).
- **Verdict**: training is **mixed bound**. (1) The GPU has work only about 60% of the time, and CPU time is 2.1× GPU time → Python and launch overhead; (2) GPU time is mostly FP32 SGEMM → Tensor Cores are not used. The rollout system is small (256 atoms, batch 1) → **launch bound**. In `gpu_calibration.json`, batch 4 takes 0.241 s per step and batch 1 takes 0.165 s per step, so 4× the trajectories cost only 1.46× the time, which also shows the GPU is not fully used.
- **Suggestions**: `torch.backends.cuda.matmul.allow_tf32=True` or `torch.set_float32_matmul_precision('high')` (may help the about 58% taken by bmm/sgemm), or bf16 autocast; cache/reuse the neighbor list on the GPU (Verlet skin) to remove the 16 rebuilds per step; `torch.compile`; use CUDA Graphs for rollout and run several trajectories at once; find where `aten::copy_` comes from (maybe dtype/device conversion).
- **Old L4 run**: **no** L4 profiling summary was found on Drive; there is only `gpu_calibration.json` (m1/m2/m4 = 0.300/0.466/0.813 s/update, rollout b1 0.165 s/step). The file does not record the device model, and it is **not comparable** with the A100 run.

## 5. Missing data
- Chrome traces over the 100 MB download limit: WignerFlow trace.json (290 MB), ocsr trace_C.json (152 MB), all YLM trace.json files (214–278 MB) → used key_averages.txt instead.
- No FLOP counts were recorded → no roofline chart.
- Only two gpu.json files record the GPU model (A100-SXM4-40GB, 2026-10-30; A100-SXM4-80GB, 2026-11-03); which card each ocsr and YLM run used is **not recorded**.
- The old WignerFlow L4 run has no profiling summary.
- Host-device sync (`.item()`) overhead was not separated out; only the ocsr trace shows `cudaDeviceSynchronize` (47.8 ms / 5 steps, mostly from the probe itself).


## Reproduce
```bash
# After downloading the raw files from Drive (file names are in manifest.json) into raw/:
python scripts/analyze_profiling.py --raw raw --out docs/profiling --manifest docs/profiling/manifest.json
pytest tests/test_analyze_profiling.py
```
`data.json` in this folder holds the extracted numbers for all runs in the three repos (the cross-repo charts also use it).
