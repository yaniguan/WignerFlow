# GPU Profiling 全量分析报告（WignerFlow / ouroboros-ocsr / YLM，Colab 运行）

> 生成时间：2026-11-03（太平洋时间）。所有数字都来自 Google Drive（yaniguan2）里的原始产物，原始文件 ID 见 `data.json` / `docs/profiling/manifest.json`。
> 找不到的数据一律标为 **缺失**，没有编造或估算的数字。

## 0. 一页结论

| 运行 | 吞吐 | 每步时间 | GPU 利用率 (nvidia-smi 均值) | 峰值显存 | 瓶颈在哪里 |
|---|---|---|---|---|---|
| WignerFlow argon `train_stable_ft`（A100-80GB，unroll m=4，bs=8，256 原子） | 训练 0.524 step/s；rollout 5.97 step/s = **25.8 ns/day** | 训练 1906.6 ms（中位数 1951.9）；rollout 167.4 ms | 35.3%（整个 profile 进程，51% 的采样 <10%） | **50.2 GB**（smi 最高 60,988 MiB） | **混合：主机侧 Python/kernel launch 开销 + FP32 GEMM**。profiler 窗口里 self CPU 8.38 s 对 self CUDA 4.08 s（2.1×）；GPU 每步只有约 1.36 s 的 kernel 时间，GPU 侧步长约 2.27 s → 约 60% 忙；`ampere_sgemm_*` 说明 matmul 跑的是 FP32 SIMT（没开 TF32） |
| ocsr A（finetune_v2，bf16，bs=32） | probe 425.8 img/s；完整训练中位数 439.2 img/s | 75.1 ms（前向 21.9 / 反向 45.9 / 优化器 4.8 / H2D 1.8 / 数据 0.7） | 完整训练 49.9%（最高 84） | 2.86 GB | **kernel launch / 小 kernel 瓶颈**：trace 中 GPU 只在 41.7% 的时间里忙；5 步共 7,615 个 kernel，平均每个 30.8 µs |
| ocsr C（finetune_v2，bf16） | probe 99.1 img/s；完整训练 94.1 img/s | 326.9 ms（前向 96.0 / 反向 220.5） | probe 73.2%，完整训练 63.8%（最高 92） | 4.20 GB | **大部分是 GPU 计算**，但约 1/3 时间 GPU 空闲（trace 152 MB，下载不了 → 缺失） |
| YLM full_A（2 段） | 1,398–1,401 samples/s | 91–99 ms | 34–35% | 1.48 GB | **CPU / kernel-launch 瓶颈**：self CPU 5.63 s 对 self CUDA 0.78 s（**7.2×**），20 步里调用了 38,398 次 `cudaLaunchKernel` |
| YLM full_B（4 段） | 778–1,270 samples/s | 104–114 ms | 25–33% | 1.45–1.48 GB | **CPU / kernel-launch 瓶颈**：6.35 s 对 0.81 s（**7.8×**），41,558 次 launch |
| YLM full_C（6 段） | 413–452 samples/s | 284–286 ms（反向约 213 ms = 75%） | 75–81% | 11.4 GB | **GPU 显存带宽瓶颈（逐元素操作）**：`SliceBackward0` 占 CUDA 时间的 3.27 s / 4.90 s（67%）；`aten::add_` 2.38 s（49%），`fill_` 0.79 s；`slice_backward` 累计分配了 1,386.7 GB 的零张量 |

图表：`charts/`（本目录）（逐仓库：`*_step_breakdown.png`、`*_top_cuda.png`；ocsr A trace：`ocsr_A_probe_trace_kernels.png`；跨仓库：`cross_gpu_util.png`、`cross_cpu_vs_cuda.png`、`cross_peak_memory.png`）。

---

## 1. Profiling 数据是怎么采集的（代码路径）

三个仓库用的是同一套思路，分三层：

**(a) 阶段计时器（step-time breakdown）**：每个阶段前后都调用 `torch.cuda.synchronize()`，再用 `time.perf_counter()` 记时间。CUDA 是异步的，不同步的话测到的只是 kernel *提交* 时间，而不是执行时间。
- WignerFlow：`equitraj/profiling.py` → `PhaseTimer.phase()`（第 71 行）、`_sync()`（第 36 行）；`profile_training()` 记录 data / neighbor_list / forward / backward / optimizer；`time_neighbor_list()` 包装近邻表构建；`rollout_speed()` 计算 steps/s 和 ns/day（`rates()`：ns/day = steps/s × stride_fs × 86400 / 1e6）。入口 `scripts/profile_run.py`。
- ouroboros-ocsr：`scripts/profile_train.py` → `timed()`（第 113–119 行，同步后计时），分 data_loading / host_to_device / forward（bf16 autocast）/ backward / optimizer；`scripts/profile_infer.py` 用 forward hook 把 encoder 和 decode 分开计时。
- YLM：`src/profiling.py` 的 `Profiler.phase()`，**只在前 `sync_steps=200` 步同步**（拆分阶段），之后不同步，避免拖慢训练；吞吐 = 相邻 step 的墙钟时间（不含验证）。在 `scripts/04_train.py` 第 273–297 行接入；每次 `--resume` 都会把旧目录改名为 `profiling_session_<ns>`，所以每次续训就是一段（seg）。

**(b) torch.profiler（算子/kernel 级）**：`torch.profiler.profile(activities=[CPU, CUDA], record_shapes=True, ...)`，导出 Chrome trace（`trace.json`）和 `key_averages().table()`（`key_averages.txt`）。
- WignerFlow `torch_profile()`（`equitraj/profiling.py` 第 283 行）：warmup 1 步 + 记录 3 步。
- ocsr：5 个 trace step（`profile_train.py` 第 171–176 行，`profile_memory=True`）。
- YLM：前 `trace_steps=20` 步。

**(c) GPU 遥测**：后台线程周期性调用 `nvidia-smi --query-gpu=utilization.gpu,...`
- WignerFlow：`gpu_samples.csv`（默认间隔 1 s，`--gpu-interval`），解析在 `parse_nvidia_smi()`。
- ocsr：`nvidia-smi ... -lms`，约 200 ms 一个样本，还带 utilization.memory 和 power.draw → `nvidia_smi_*.csv`。
- YLM：约 1 s 一次 → `gpu_utilization.jsonl`（第一段 full_A 出现过 `nvidia-smi` 2 s 超时）。

**显存**：`torch.cuda.max_memory_allocated()`（ocsr 另外记录 `max_memory_reserved`）。

---

## 2. 怎么一步步分析（可复现：`scripts/analyze_profiling.py`）

1. **先看吞吐和每步时间**（summary.json），定下基线。
2. **看阶段拆分**：如果 data 阶段占步长 >20%，就是输入瓶颈（DataLoader/CPU）。三个仓库的 data 都 <3 ms，**都不是输入瓶颈**。
3. **看 GPU 利用率**：nvidia-smi 的 `utilization.gpu` 表示“采样窗口里至少有一个 kernel 在跑的时间比例”。低（<50%）说明 GPU 在等主机；高（>75%）只说明 GPU 一直有活干，**并不代表算力被吃满**。
4. **对比主机时间和设备时间**：`key_averages.txt` 末尾的 `Self CPU time total` 和 `Self CUDA time total`。CPU ≫ CUDA，同时 GPU 利用率低 → Python/launch 瓶颈。
5. **看 trace 里的空隙**：把所有 kernel 的时间区间合并，算出 GPU 忙碌比例（`trace_gpu_busy()`）；kernel 数量多、平均时长只有几十 µs → launch 瓶颈。
6. **看排名前几的 kernel**：`sgemm` 说明是 FP32（没用 Tensor Core），`*_elementwise_*`/`add_`/`fill_`/`copy_` 是显存带宽型操作，`nchwToNhwc` 是内存布局转换，`multi_tensor_apply` 是优化器。
7. **Roofline**：原始产物里**没有记录 FLOP 数**，所以无法给出真正的算术强度和 roofline（**缺失**）。只能按 kernel 类型定性判断：GEMM 属于计算密集，逐元素操作属于带宽密集。
8. 用规则给出结论（`verdict()`），每条结论都附上证据。

运行方式：`python scripts/analyze_profiling.py --raw <raw 目录> --out <输出目录>`；测试：`pytest tests/test_analyze_profiling.py`。

---

## 3. 怎么得出结论（判断规则）

| 现象 | 结论 |
|---|---|
| data 阶段 >20% | 输入 / DataLoader 瓶颈 → 加 `num_workers`、`pin_memory`、`prefetch` |
| GPU 利用率 <50%，且 self CPU / self CUDA >2 | CPU / Python / kernel-launch 瓶颈 → `torch.compile`、CUDA Graphs、加大 batch、融合算子 |
| trace 中 GPU 忙碌 <60%，kernel 多且短 | 小 kernel / launch 瓶颈 |
| GPU 利用率 ≥75%，主要是 GEMM/卷积 | 计算瓶颈 → bf16/TF32、更好的 kernel |
| GPU 利用率 ≥60%，逐元素操作 >40% | 显存带宽瓶颈 → 消除多余的分配/拷贝，做融合 |
| 峰值显存接近显卡容量 | 显存容量瓶颈 → activation checkpointing、减小 batch/unroll |

---

## 4. 各运行的结果与瓶颈

### 4.1 WignerFlow — argon_full_oct29/train_stable_ft（A100-SXM4-80GB，torch 2.11.0+cu130，1,945,520 参数）
- **训练**（30 步，warmup 3，bs=8，m=4，unroll_sampling_steps=2，截断反传）：每步 1906.6 ms，0.524 step/s。阶段占比：前向 936.1 ms（49.1%）、反向 689.8 ms（36.2%）、**近邻表 268.3 ms（14.1%，每步 16 次调用）**、优化器 9.9 ms、数据 2.3 ms。
- **峰值显存 50.2 GB**；nvidia-smi 最高 60,988 MiB（所以是 80 GB 卡；40 GB 卡放不下）。
- **Profiler**（3 步）：self CUDA 4.079 s，self CPU 8.382 s。最耗时的项：`aten::bmm` 1318 ms（32.3%），`ampere_sgemm_128x128_nt` 1062 ms（26.0%），`aten::mm` 499 ms，`addmm` 348 ms，`gemmSN_NN` 236 ms；index / index_put（scatter/gather）合计约 440 ms；`aten::copy_` CPU total 1.65 s。
- **Rollout**：5.97 step/s，每步 167.4 ms，**25.81 ns/day**（batch 1，256 原子，k=3，每步 10 次采样迭代 → 每次去噪器评估约 16.7 ms）；近邻表 2.52 ms/次（估计占 6.0%）。
- **结论**：训练是**混合瓶颈**。(1) GPU 只在约 60% 的时间里有活干，CPU 时间是 GPU 的 2.1 倍 → Python 和 launch 开销；(2) GPU 时间以 FP32 SGEMM 为主 → 没有用上 Tensor Core。Rollout 体系很小（256 原子，batch 1）→ **launch 瓶颈**；`gpu_calibration.json` 里 batch 4 每步 0.241 s、batch 1 每步 0.165 s，也就是 4 倍轨迹量只多花 1.46 倍时间，同样说明 GPU 没吃饱。
- **建议**：`torch.backends.cuda.matmul.allow_tf32=True` 或 `torch.set_float32_matmul_precision('high')`（可能对 bmm/sgemm 占的约 58% 有效）或 bf16 autocast；近邻表在 GPU 上缓存/复用（Verlet skin），去掉每步 16 次重建；`torch.compile`；rollout 用 CUDA Graphs，并同时跑多条轨迹；排查 `aten::copy_` 的来源（可能是 dtype/device 转换）。
- **旧的 L4 运行**：Drive 上**没有找到** L4 的 profiling summary；只有 `gpu_calibration.json`（m1/m2/m4 = 0.300/0.466/0.813 s/update，rollout b1 0.165 s/step），文件里没有记录设备型号，与 A100 运行**不可比**。

## 5. 缺失的数据
- Chrome trace 超过 100 MB 下载上限：WignerFlow trace.json（290 MB）、ocsr trace_C.json（152 MB）、YLM 全部 trace.json（214–278 MB）→ 改用 key_averages.txt。
- 没有记录 FLOP 数 → 无法画 roofline。
- 只有两个 gpu.json 记录了 GPU 型号（A100-SXM4-40GB，2026-10-30；A100-SXM4-80GB，2026-11-03）；ocsr 和 YLM 每次运行用的是哪张卡**没有记录**。
- 旧的 WignerFlow L4 运行没有 profiling summary。
- 没有分离出主机-设备同步（`.item()`）的开销；只有 ocsr 的 trace 能看到 `cudaDeviceSynchronize`（47.8 ms / 5 步，主要来自 probe 本身）。


## 复现
```bash
# 把 Drive 上的原始文件（文件名见 manifest.json）下载到 raw/ 后：
python scripts/analyze_profiling.py --raw raw --out docs/profiling --manifest docs/profiling/manifest.json
pytest tests/test_analyze_profiling.py
```
本目录的 `data.json` 收录了三个仓库所有运行的提取数字（跨仓库对比图也用它）。
