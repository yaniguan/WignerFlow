# EMA Shadow Weights, Verlet Unroll Graph Reuse, and TF32 Training

This document describes the opt-in training stability and throughput upgrades in `equitraj/ema.py`, `equitraj/train.py`, and `equitraj/bench.py`. All options are disabled by default so existing training configs reproduce their previous behavior unchanged.

## 1. Exponential Moving Average Shadow Weights (`equitraj/ema.py`)

Autoregressive trajectory unrolling (`m = 4..16`) produces high-variance stochastic gradients at unroll-length switches. Maintaining an Exponential Moving Average (EMA) of trainable parameters damps high-frequency weight oscillations that otherwise compound across $10^4$-step rollouts:

```yaml
train:
  ema_decay: 0.999
  ema_warmup: true
```

- At update $u \ge 1$, `ExponentialMovingAverage` applies warmup decay
  $$\beta_u = \min\!\left(\beta,\; \frac{1 + u}{10 + u}\right)$$
  when `ema_warmup: true`, avoiding early stagnation near random initialization.
- Shadow tensors are stored in at least `float32` (`float64` when parameters are `float64`).
- During `evaluate_and_select()` in `equitraj/train.py`, validation losses, validation rollouts, `best.pt`, `best_stable.pt`, and `snapshots/stepXXXXXX.pt` are evaluated and exported under `ema.average_parameters(model)`, while `ckpt.pt` retains the live optimizer-paired weights and `ema.pt` atomically stores the EMA shadow state for seamless resumption.

## 2. Sliding-Window & Verlet Neighbor Reuse in Unrolled Training (`train.verlet_skin`)

During an $m$-step training unroll (`m = 4, 8, 16`) with context length $k$:
1. Without caching, `EquiTraj.encode` rebuilds $k$ context graphs plus `graph_last` from scratch at every step $j = 0, \dots, m-1$, doing $m(k + 1)$ neighbor searches and Wigner $D$-matrix evaluations per trajectory window even though $k - 1$ context frames at step $j + 1$ are identical to frames $1, \dots, k - 1$ at step $j$.
2. Over a single coarse stride ($\sim 50\text{ fs}$), atoms move $\sim 0.07\text{ Å}$, well inside half a `1.0 Å` Verlet skin.

Setting `train.verlet_skin` enables both optimizations inside `training_loss`:

```yaml
train:
  verlet_skin: 1.0
```

- Each window in the micro-batch keeps a `VerletList(model.cutoff, verlet_skin)` and a rolling $k$-element deque of single-frame `(src, dst, vec, dist, wigner)` graphs.
- `graph_last` is collated directly from the newest frame of the sliding window (eliminating duplicate Wigner builds), and advancing from $j \to j + 1$ computes only **one** new frame graph per window while reusing the cached Verlet pair list whenever atomic displacements stay within `verlet_skin / 2`.
- Full differentiability with respect to predicted coordinates is preserved in both `backprop: truncated` and `backprop: full` modes (`minimum_image(pos[src] - pos[dst], box)` remains in the autograd graph).

## 3. TensorFloat-32 Matmul Precision (`train.matmul_precision`)

On Ampere/Hopper GPUs (A100, H100, RTX 30/40/50-series), enabling TF32 tensor-core matmuls accelerates channel projections and radial MLPs without altering FP64 coordinate accumulation:

```yaml
train:
  matmul_precision: high   # 'highest' | 'high' | 'medium'
```

## 4. Cross-Platform Benchmark Memory Helpers (`equitraj/bench.py`)

`current_rss_mb()` and `peak_rss_mb()` in `equitraj/bench.py` now fall back cleanly on Windows via Win32 `K32GetProcessMemoryInfo` (`WorkingSetSize` and `PeakWorkingSetSize`) when POSIX `resource` / `/proc/self/statm` are unavailable.
