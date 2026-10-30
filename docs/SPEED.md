# Opt-in inference speed path

`FastRollout(model, sampling_steps=10, precision="fp32", neighbor="verlet")`
copies the model for inference. `predict(pos, vel, types, box, masses, noise=None)`
accepts `(R,k,N,3)` contexts and returns `(dx, v_new)` with shape `(R,N,3)`.
`rollout(..., n_steps)` accumulates positions in float64, returning saved CPU
positions/velocities and final device contexts. Inputs must already be on the
model device. It is a compute path without crash monitoring or thermostats;
validate trained dynamics using the existing monitored rollout.

The original model APIs and defaults are preserved. Each replica/context slot
has its own Verlet cache, shared by fusion for the newest frame. Pairs are built
at cutoff+skin and filtered at the physical cutoff each call. Movement beyond
skin/2, a box change, or an optional rebuild_every call interval rebuilds them.
Unwrapped reference displacements conservatively rebuild on periodic jumps.
Cell lookup uses sorted sparse cell IDs and vectorised range expansion, with no
atom loops. Sorting costs O(N log N), followed by O(N+E) candidate work at bounded
density; it is not a strict worst-case O(N) algorithm. Dense/clustered systems
can still have quadratic edge counts. Small periodic boxes use brute force.
Triclinic bins use perpendicular cell widths and the same fractional minimum
image as the reference implementation (not a general closest-lattice solver).
The physical cutoff must be below half the smallest perpendicular box width;
the larger skin radius need not satisfy this restriction.

Use precision="bf16" for CPU/CUDA autocast. precision="fp8" lazily tries
`torchao.float8.convert_to_float8_training` (its Linear modules also support
forward inference under no_grad). Only nn.Linear is converted; Wigner rotations,
SO(2) structure and equivariant tensor operations retain fp32/bf16 arithmetic.
Unsupported devices, missing torchao, incompatible dimensions, and execution
errors fall back with a warning. A100 does not have native float8 Tensor Cores;
expect fp8 fallback or no benefit there. Benchmark records requested and effective
precision; never label fallback timing as native fp8 throughput.

`compile=True` accepts compile_mode="reduce-overhead" or "max-autotune" and
falls back to eager on construction/execution failures. `cuda_graphs=True`
pads the two concatenated graphs to power-of-two edge capacities, or a supplied
edge_capacity. Masked dummy edges have zero envelope and attention, including
for isolated nodes. Graph construction and displacement checks remain outside
capture; numeric prediction uses static buffers and externally generated noise.
Shape/capacity changes recapture. Explicit capacity overflow is an error, never
edge truncation. Capture failures disable CUDA graphs with a warning. A repulsion
prior may require data-dependent work and cause capture fallback. Compile/capture
warmup is excluded from timing; inspect activation flags in JSONL results.

## Train one or two flow steps

Running an existing ten-step flow with one Euler step changes its numerical
solution and need not preserve accuracy. Distil before claiming few-step quality.
The progressive endpoint objective couples teacher and student with identical
COM-free Gaussian noise: a frozen n-step teacher supervises an n/2-step student
in normalized displacement/velocity target units. Student integration uses
predict_with_grad; optional teacher-endpoint flow-matching uses model.loss.
Use training contexts only, retain independent validation for model selection,
and assess held-out stability, RDF, diffusion and temperature after every stage.

Export a trained model using scripts/export_ckpt.py. Save training-context tensors
as a dictionary (or list of batch dictionaries) in a .pt file: pos and vel have
shape (B,k,N,3), types/masses shape (N,), box shape (B,3,3) or None. Units Å, Å/fs.
Then progressively distil (start from a teacher validated at four steps):

```sh
python scripts/distill.py --teacher four.pt --data train_contexts.pt --teacher-steps 4 --out two.pt --device cuda
python scripts/distill.py --teacher two.pt --data train_contexts.pt --teacher-steps 2 --out one.pt --device cuda
```

Ten steps can halve to five, but an odd five-step stage cannot halve exactly;
train/validate a four-step teacher first, or start with a power-of-two schedule.
Checkpoint sampling_steps records the intended integration budget. Learning rate,
iteration count, optional supervised-weight and validation remain experiment
choices; distillation is not a guarantee of physical fidelity.

## Compare throughput fairly

```sh
python scripts/bench_speed.py --micro --sizes 128 512 2048 --steps 20 --out micro.jsonl
python scripts/bench_speed.py --device cuda --platform CUDA --neighbors verlet --sampling-steps 1 10 --out speed.jsonl --trace speed_trace
```

Defaults sweep 128–16384 atoms, fp32/bf16/fp8, 1/2/10 sampling steps, 1/4
replicas, compile off/on and brute/cell/Verlet neighbours. Architecture defaults
match B_S3a dimensions: channels=64, lmax=2, heads=8, n_rbf=32, cutoff=6 Å.
Use --channels, --lmax, --heads, --n-rbf and block counts to change compute size.
Synthetic FCC argon or rocksalt NaCl tiles are truncated to exact atom counts;
non-cubic counts leave vacancies in the cubic periodic volume. Both engines
receive identical coordinates. These are compute fixtures, not equilibrated
physical reference datasets. Argon uses periodic Lennard-Jones; NaCl uses simple
charged PME NonbondedForce. Random model weights cannot establish accuracy or
20 ns stability. Compare production checkpoints on matched physical systems
with scripts/bench_openmm.py and the existing validation workflow.

Every timed model call advances all replicas; steps/s and ns/day refer to each
replica, aggregate ns/day multiplies by replicas. Each ns/day speedup is compared
to one OpenMM trajectory. OpenMM and torch GPU work are synchronized. CUDA peak
allocated bytes include setup/warmup and model buffers; CPU/OpenMM peak memory
is unavailable (null). Failure/OOM rows record errors and the sweep continues.
Micro mode includes the first Verlet build amortised over repeated unchanged
frames; this is the best-case reuse limit, not a moving-system throughput claim.

Per model step, work includes k spatial graphs, temporal fusion, and the sampling
budget of denoiser evaluations. OpenMM instead evaluates forces per integrator
step. Physical step size must accompany steps/s: 50 fs model stride vs 2 fs MD
means 25× fewer calls per nanosecond. Thus
`ns/day = steps/s × stride_fs × 86400 / 1e6`.
Report per-step latency and physical throughput together; fewer calls can offset
more expensive steps. Multi-replica aggregate speedup is a throughput comparison,
not single-trajectory latency. Profiler traces and fallback flags explain which
optimisations were actually active.
