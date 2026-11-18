# CPU inference API: latency budget and measured numbers

`equitraj/api.py` serves the released argon model (`checkpoints/argon_flow_v0.pt`). This model and every number here are **argon only**.
`scripts/bench_api.py` produced the numbers below. Raw output: [`results/cpu_api/bench.json`](../results/cpu_api/bench.json).

## What the 1 s budget covers

One request = `ArgonPredictor.predict(...)` with `n_steps=1`: input checks, one 50 fs step of the flow model
(3 context frames in, 1 frame out), the stability monitor, and building the JSON-ready result.

Not included:

- model loading and warm-up (once per process, in `ArgonPredictor.__init__`);
- HTTP transfer and JSON parsing in FastAPI;
- `n_steps=k` costs about k times one step, so the budget is per 50 fs step.

## Setup

- CPU: `Intel(R) Xeon(R) Processor` (8 logical CPUs), 1 torch thread, torch 2.15 (CPU build).
- The machine was shared with other jobs: load average 15–21 during the runs (more than 2 jobs per core).
  **Wall** time includes waiting for the CPU. **CPU** time is the process time of the request, i.e. about its cost on an idle core.
- Input: the bundled 256-atom argon context stored in the checkpoint. A 108-atom FCC box at the training density shows how cost scales with size.
- 30 requests per setting for 256 atoms, 10 for 108 atoms.

## Latency per request (ms)

S = flow sampling steps (Euler). The released model was evaluated with S = 10; the API default is S = 2.

| Atoms | S | wall p50 / p95 | CPU p50 / p95 | under 1 s? |
|---|---|---|---|---|
| 256 | 1 | 778 / 851 | 755 / 814 | yes |
| **256** | **2 (default)** | **944 / 1109** | **919 / 1021** | p50 yes, p95 no |
| 256 | 4 | 1223 / 1390 | 1183 / 1297 | no |
| 256 | 10 | 3622 / 6181 | 2544 / 2771 | no |
| 256 | 10, `EquiTraj.predict` (old path) | 2722 / 3019 | 2641 / 2788 | no |
| 108 | 1 | 238 / 248 | 237 / 243 | yes |
| **108** | **2 (default)** | **282 / 319** | **279 / 309** | yes |
| 108 | 4 | 413 / 445 | 403 / 426 | yes |
| 108 | 10 | 795 / 1323 | 702 / 758 | p50 yes, p95 no |

Two short repeat runs later in the night (3 requests at 256 atoms, 5 at 108, load 15–20) were slower:
256 atoms S = 2 took 1380 ms wall p50 (1238 ms CPU), and S = 1 took 1022 ms (988 ms CPU). See `bench_csvr95_s*.json`.
On this shared machine, a 256-atom request is right at the 1 s line and can go over it.

**Honest result:** at 256 atoms the default meets 1 s at p50 but not at p95 on this loaded box. Systems of about 100 atoms fit with room to spare.
S = 1 brings 256 atoms under 1 s here, but it is less accurate (below).

## Accuracy of fewer sampling steps

One step, same noise, compared with S = 10 (relative RMS of the predicted displacement dx and velocity v):

| S | dx | v |
|---|---|---|
| 1 | 2.6% | 4.5% |
| 2 | 0.8% | 1.2% |
| 4 | 0.6% | 0.9% |

The frame-split path matches `EquiTraj.predict` exactly (max abs diff 0.0).

100-step rollouts from the bundled context, 3 seeds, no thermostat (reference temperature 94.9 K):

| S | second-half T (K) | RDF first peak position error (Å) | RDF first peak height error |
|---|---|---|---|
| 1 | 44.3 | 0.116 | 33% |
| 2 | 62.8 | 0.077 | 23% |
| 4 | 65.9 | 0.080 | 17% |
| 10 | 62.6 | 0.066 | 22% |

- S = 2 rollouts are about as good as S = 10; both cool from 95 K to about 63 K over 100 steps. The cooling is a property of the v0 model, not of the few-step sampler.
- S = 1 cools much more (44 K), so it is not the default.
- With the optional CSVR thermostat at 95 K (`thermostat_K=95`): S = 2 gives 81.6 K, RDF peak position error 0.060 Å, height error 15%; S = 1 gives 72.5 K, 0.097 Å, 16%.

## Where the time goes

One 256-atom S = 2 request, timed per stage (10 requests, load about 18, 1223 ms total):

| Stage | ms | share |
|---|---|---|
| spatial blocks (2 blocks x 3 frames) | 551 | 45% |
| denoiser (2 sampling steps) | 399 | 33% |
| fusion blocks | 189 | 15% |
| embed, temporal, neighbor graph | 79 | 6% |

## Options to get 256 atoms under 1 s at p95

1. **Cache frames across a rollout.** Two of the three context frames in step t+1 were already encoded in step t.
   A session that keeps the per-frame spatial output would cut the spatial cost by about two thirds (about 350 ms here). Exact, no accuracy cost.
2. **int8 dynamic quantization** of the linear layers in the spatial and denoiser blocks; needs an equivariance and rollout check before use.
3. **A smaller distilled checkpoint** (fewer channels or blocks); needs new training and a new argon evaluation.
4. **ONNX / TorchScript export** of the per-frame blocks to cut Python overhead; gain not measured.
5. **S = 1** fits today (851 ms p95) but cools rollouts more (44 K vs 63 K).
6. A dedicated core: CPU time p95 is 1021 ms, so on an idle core the default is about at the line, not clearly under it.

## Reproduce

```bash
python scripts/bench_api.py --out results/cpu_api/bench.json
python scripts/bench_api.py --sampling-steps 2 --reps 3 --thermostat-K 95 --out results/cpu_api/bench_csvr95_s2.json
```
