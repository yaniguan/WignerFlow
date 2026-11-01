# Argon CPU reduced retrain — October 29, 2026

This is a reduced CPU experiment after moving the GraphAttention cutoff envelope inside softmax with a zero-logit sink. It does not reproduce the original GPU training budget or establish long-rollout release quality. The original v0 weights are evaluated as-is with the current attention implementation, without weight conversion or fine-tuning. No git commands were run.

## Provenance and reconstruction

Read `equitraj/simulate.py`, `equitraj/systems.py::build_argon`, `scripts/gen_data.py`, `equitraj/dataset.py`, README, `docs/release_v0.md`, `results/argon/*/config.yaml`, `results/argon/ground_truth/metrics.json`, and the released checkpoint metadata.

`configs/data/argon.yaml` is a usable reconstruction with deliberately reduced durations and trajectory count, not a claim to have recovered the missing original file byte for byte:

- `build_argon()` fixes 256 atoms, mass 39.948 Da, density 1.374 g/cm³, LJ sigma 0.3405 nm and epsilon 0.996 kJ/mol; the force is switched over 0.80–0.85 nm with dispersion correction off. The builder accepts a system name, so YAML cannot override those builder defaults.
- Released metadata gives `stride_fs=50`, `stride_steps=10`, hence `dt_fs=5`. The archived ground-truth metrics give a 5 fs saved-frame interval, hence `save_every=1`.
- The old reference temperature is 94.90594155226378 K. The 95 K thermostat target is an inference, not an exact recovered setting. Friction 1/ps, constraint tolerance 1e-6, no NPT, 20 ps NVT equilibration, force storage every 10 saved frames, and new seeds are explicit choices for this reduced run; they are not recoverable from the release checkpoint.
- Four seeds 20261029–20261032, each with 20 ps production requested. Train trajectories [0,1], validation [2], test [3]. Each file contains 4,000 frames from 0 to 19.995 ps; the generation loop includes the starting frame, so the saved span is one save interval shorter than the requested duration.
- NVT Langevin equilibration is followed by NVE custom velocity Verlet, with COM velocity removed and initial kinetic energy rescaled to the NVT mean total energy, exactly as implemented in `simulate.py`. OpenMM CPU precision is reported as platform default, regardless of the GPU-only `precision: mixed` config key.

Large trajectory arrays remain in `data/argon/`; docs contain no data arrays. Full metadata and byte sizes are in `results/argon/cpu_reduced/environment_data.json`.

## Implementation changes required for this environment

`equitraj/train.py` now accepts `--no_tensorboard` and imports TensorBoard only when enabled. The existing stdout and validation JSON logs and checkpoint logic remain in use. TensorBoard was missing; the supplied Python also has no pip module. An attempted system-pip install into `/tmp/wigner_deps` failed because network DNS was unavailable. No shim or substitute training loop was used.

`scripts/evaluate.py` now accepts released checkpoints directly with `--data_dir`: it uses their original state dictionaries, architecture, stride, and training step. It checks the dataset stride against release metadata and accepts an explicit `--threads` limit. Existing training checkpoint loading remains supported. The existing evaluation calculations and crash thresholds are used for both checkpoints.

`scripts/compare_ckpt_equivariance.py` compares predicted displacement and new velocity with their rotated original outputs. It transforms the periodic box and COM-free vector-valued flow noise together with positions and velocities. Each trial uses a random proper rotation and that rotation composed with inversion (determinant −1). This measures pathwise equivariance, rather than comparing different random samples. Float64 measurements promote the saved float32 weights/data; model constants are constructed in float64.

## Reproduction commands

Working directory `/workspace/repos/WignerFlow`; Python `/workspace/venv311/bin/python`; `PYTHONPATH=.` throughout. The supplied environment exposes eight CPU cores. The commands below were invoked through a timing wrapper; exact argument lists, exit codes and measured wall seconds are in `results/argon/cpu_reduced/commands.jsonl`. The wrapper and exploratory benchmark driver are archived alongside that file (`timed_command.py`, `benchmark_driver.py`). Logs are in the same results directory.

```bash
OPENMM_CPU_THREADS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/gen_data.py --config configs/data/argon.yaml

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /workspace/venv311/bin/python results/argon/cpu_reduced/benchmark_driver.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/analyze_gt.py --data data/argon \
  --out results/argon/cpu_reduced/ground_truth --vacf_ps 1 --msd_ps 5 --diff_fit_ps 2 5

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENMM_CPU_THREADS=8 PYTHONPATH=. \
  /workspace/venv311/bin/python -m equitraj.train \
  --config configs/argon/B_S3a_cpu_reduced.yaml --device cpu --no_tensorboard

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/evaluate.py --run . \
  --ckpt checkpoints/argon_flow_v0.pt --data_dir data/argon \
  --steps 100 --seeds 1 --starts 1 --sampling_steps 10 --device cpu --threads 2 \
  --out results/argon/cpu_reduced/v0

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /workspace/venv311/bin/python -m pytest tests/test_model.py tests/test_release.py -q
```

The initial benchmark import failed before any measurements because of missing TensorBoard. The rerun produced completed one-, two-, and four-thread measurements; the eight-thread trial was interrupted with Ctrl-C after producing no completed measurement. Those measured samples are retained in `benchmark.json`. The training and v0 evaluation started near the end of that benchmark; timing samples are exploratory and not a controlled hardware speed comparison.

## Interpretation limits

The fresh dataset has only two training trajectories and one validation/test trajectory each, all much shorter than the original ten 0.5 ns production trajectories. New seeds and finite equilibration introduce sampling differences. This run changes the data volume, batch size, step count, warmup, curriculum, validation horizon and attention together; it cannot isolate the attention fix's effect on physical accuracy.

The evaluation uses model-only dynamics without a thermostat, the same held-out trajectory, test starts and random seed for both checkpoints. One start with one stochastic sample does not provide confidence intervals. A 100-step rollout is 5 ps at the model stride; surviving it does not demonstrate stability for the original 10,000-step benchmark. RDF, VACF and especially short-time diffusion estimates have finite-sample and finite-horizon limitations. The v0 criteria printed by the existing evaluator are retained for transparency, not a new release certification.

The maximum combined equivariance error is a numerical diagnostic over two differently dimensioned outputs. The separate displacement (Å) and velocity (Å/fs) errors are the physically interpretable values. Rotating the box together with the coordinates tests covariance of the periodic system; keeping an axis-aligned box fixed would describe a different physical configuration.

The initial v0 attempt used 200 steps, two starts, ten flow steps and OMP/MKL thread counts 4, without the explicit PyTorch thread argument. It produced no completed metrics and was interrupted with Ctrl-C (exit 130) to reduce the evaluation budget. Its log is `evaluation_v0_interrupted.log`; its wall time was not captured by the interrupted timing wrapper. Both reported evaluations use the revised shared 100-step, one-start protocol.

Final-snapshot export and paired comparison commands (queued by `results/argon/cpu_reduced/postprocess.py` after `train/done.json` appears):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/export_ckpt.py \
  --run results/argon/cpu_reduced/train --ckpt snapshots/step001000.pt \
  --out checkpoints/argon_flow_v1_cpu_reduced.pt

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/evaluate.py --run . \
  --ckpt checkpoints/argon_flow_v1_cpu_reduced.pt --data_dir data/argon \
  --steps 100 --seeds 1 --starts 1 --sampling_steps 10 --device cpu --threads 2 \
  --out results/argon/cpu_reduced/v1

OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLCONFIGDIR=/tmp/wigner_mpl PYTHONPATH=. \
  /workspace/venv311/bin/python scripts/compare_ckpt_equivariance.py \
  --checkpoints checkpoints/argon_flow_v0.pt checkpoints/argon_flow_v1_cpu_reduced.pt \
  --trials 3 --sampling_steps 10 --threads 2
```

`interrupted_runs.json` records the exact commands and exit status 130 for both interrupted attempts. Their uncaptured durations are null, not estimates. `compatibility.json` confirms architecture/context/stride/types/masses equality with v0 and a maximum stored-box difference of 0.0 Å. `data_sha256.json` identifies the generated data files.

## Completed run: measured results

Environment: Python 3.11.16, PyTorch 2.10.0+cpu, OpenMM 8.6.1, h5py 3.16.0, NumPy 1.26.4; CPU affinity [0, 1, 2, 3, 4, 5, 6, 7].

Training completed 1000 optimizer steps in 2167.815111 s (36.130252 minutes), measured by the training loop. Fresh initialization, seed 1, 1,945,520 parameters; original model config unchanged. Batch 1, warmup 100, unroll schedule [[0,1],[600,2],[900,4]], 1,000 total updates. Validation every 250 steps: two batches, one start, ten rollout steps, two flow steps. The final `snapshots/step001000.pt` was exported; `best.pt` was not selected using test results.

| Step | Mean training loss (last 50) | Unroll length | Elapsed training seconds |
|---|---:|---:|---:|
| 50 | 1.555197965 | 1 | 156.083737 |
| 100 | 0.6190894806 | 1 | 250.326017 |
| 150 | 0.5515530007 | 1 | 339.990190 |
| 200 | 0.4307954388 | 1 | 429.479011 |
| 250 | 0.415692656 | 1 | 521.290746 |
| 300 | 0.3619877706 | 1 | 627.392252 |
| 350 | 0.3783823987 | 1 | 718.592572 |
| 400 | 0.3875531076 | 1 | 812.018406 |
| 450 | 0.284824722 | 1 | 908.731285 |
| 500 | 0.1961883409 | 1 | 1000.108494 |
| 550 | 0.2274923703 | 1 | 1094.499574 |
| 600 | 0.2446491624 | 1 | 1178.389177 |
| 650 | 0.6448246296 | 2 | 1289.795246 |
| 700 | 0.571851552 | 2 | 1398.157238 |
| 750 | 0.5738118319 | 2 | 1506.478083 |
| 800 | 0.5220340283 | 2 | 1623.961144 |
| 850 | 0.463616149 | 2 | 1732.075758 |
| 900 | 0.434056973 | 2 | 1840.980989 |
| 950 | 1.437588478 | 4 | 2004.034001 |
| 1000 | 1.574730972 | 4 | 2162.266417 |

| Step | Validation loss | Crash fraction | Final T (K) | RDF L1 | Rollout wall (s) |
|---|---:|---:|---:|---:|---:|
| 250 | 0.4587484077 | 0.0 | 154.4604743979181 | 0.18457081440533618 | 5.113706 |
| 500 | 0.4486649483 | 0.0 | 102.79412433491872 | 0.10924696172012442 | 5.508052 |
| 750 | 0.5772725195 | 0.0 | 111.37997097198364 | 0.14401562545912913 | 4.990627 |
| 1000 | 0.08303678781 | 0.0 | 91.93569659859827 | 0.09045348480250594 | 4.472802 |

Ground-truth analysis over all four trajectories: T = 95.51981958 K; mean total-energy drift = -3.345539318e-06 kJ/mol/ns/DOF; maximum absolute per-trajectory drift = 6.892712319e-05 in the same units. RDF peak = 3.697711617 Å, height = 2.855060024; VACF first zero = 310 fs; diffusion = 2.332902965e-05 cm²/s (fit 2–5 ps). This all-trajectory summary is distinct from the test-only reference used in the paired evaluation.

| Command | Wall seconds | Exit |
|---|---:|---:|
| generation | 16.727624 | 0 |
| benchmark | 1.658744 | 1 |
| ground_truth_analysis | 6.682421 | 0 |
| tests | 6.577935 | 0 |
| evaluation_v0 | 114.019891 | 0 |
| training | 2170.927691 | 0 |
| export | 2.178633 | 0 |
| evaluation_v1 | 106.640842 | 0 |
| equivariance | 77.961857 | 0 |
| integrity | 1.104763 | 0 |

Elapsed wall time since initial inspection: 2551.362387 s. Timed jobs overlapped; their sum is not total elapsed time.

The existing evaluator uses the held-out trajectory 3. Its starting window begins at saved-frame index 0 (model-subsampled index 0), with random seed 0 applied identically. It starts from three 50 fs context frames. Test ground-truth temperature is computed independently of the checkpoint reference payload.

| Metric | v0 as-is | v1 CPU reduced |
|---|---:|---:|
| train step | 10000 | 1000 |
| ground-truth T (K) | 94.69629945 | 94.69629945 |
| crashed / 1 | 0 | 0 |
| steps before crash | [100] | [100] |
| second-half T (K) | 64.2638819 | 83.38392931 |
| temperature relative error | 0.3213686039 | 0.1194594741 |
| RDF L1 | 0.7325985914 | 0.1952037799 |
| RDF peak position error (Å) | 0.06328033071 | 0.01908895379 |
| RDF peak height relative error | 0.2192077083 | 0.01877784514 |
| VACF first zero, model (fs) | 318.947932 | 309.4072218 |
| VACF first zero, reference (fs) | 306.6099056 | 306.6099056 |
| VACF first-zero relative error | 0.04024014301 | 0.009123372031 |
| D model (cm²/s) | 1.80411409e-05 | 2.657405026e-05 |
| D reference (cm²/s) | 2.239060007e-05 | 2.239060007e-05 |
| D relative error | 0.1942538011 | 0.1868395743 |
| step-1 RMSD (Å) | 0.00767831644 | 0.008811568841 |
| step-10 RMSD (Å) | 0.137882039 | 0.1171967238 |
| step-20 RMSD (Å) | 0.2993071079 | 0.3396420479 |
| rollout wall (s) | 109.4018221 | 102.9895775 |
| all v0 criteria pass | False | False |

Missing metrics are not replaced with invented values: the evaluator skips structural/dynamical statistics when all rollouts crash, and a VACF zero can be absent in the measured horizon.

### O(3) output errors

Three shared test windows × proper/improper rotation per precision per checkpoint; seed 20261029; flow sampling steps 10. Total script wall 76.428805 s. Maxima below include every tested component, atom, window and transform.

| Checkpoint | Precision | Max Δx error (Å) | Max v_new error (Å/fs) | Max across outputs |
|---|---|---:|---:|---:|
| argon_flow_v0.pt | torch.float32 | 1.86264514923e-06 | 4.31318767369e-08 | 1.86264514923e-06 |
| argon_flow_v0.pt | torch.float64 | 1.90125692967e-15 | 4.70543742859e-17 | 1.90125692967e-15 |
| argon_flow_v1_cpu_reduced.pt | torch.float32 | 1.92783772945e-06 | 2.63535184786e-08 | 1.92783772945e-06 |
| argon_flow_v1_cpu_reduced.pt | torch.float64 | 2.96637714392e-15 | 6.54858112181e-17 | 2.96637714392e-15 |

### Files and integrity

| File / directory | Bytes |
|---|---:|
| `checkpoints/argon_flow_v0.pt` | 7893931 |
| `checkpoints/argon_flow_v1_cpu_reduced.pt` | 7897419 |
| `data/argon` | 206847391 |
| `results/argon/cpu_reduced` | 79884083 |
| `data/argon/traj_000.h5` | 27130768 |
| `data/argon/traj_001.h5` | 27130768 |
| `data/argon/traj_002.h5` | 27130768 |
| `data/argon/traj_003.h5` | 27130768 |

Directory sizes sum regular-file bytes (not disk blocks); result directory size is measured before writing the final summary itself. Checkpoint SHA-256:

- `checkpoints/argon_flow_v0.pt`: `4c8e035ee177bb8ad7d9abe45a95d047b5e5d49996ad8a38bff9ae9619ec3372`
- `checkpoints/argon_flow_v1_cpu_reduced.pt`: `ff3d4cf1552e1ae4a6851c661adbe7007f292e32c5392b5518e1f81837e8717f`

The v0 file hash was checked against the original inspection and its `.sha256` sidecar. Both exported payloads load with `weights_only=True`. Model/release tests: 32 passed; full output is in `tests.log`. Exact per-transform equivariance numbers, per-trajectory metadata, source hashes, validation logs, plots, and evaluation JSON are retained in `results/argon/cpu_reduced/`.


### Exploratory CPU timings

Batch size 1, unchanged architecture. Each update timing averages three measured updates after one warmup. Measurements are short and include overlap caveats described above.

| Threads | Unroll | Training seconds/update | Rollout seconds/step (10 flow steps) |
|---:|---:|---:|---:|
| 1 | 1 | 3.083015160 | — |
| 1 | 2 | 3.325080443 | — |
| 1 | — | — | 1.789153544 |
| 2 | 1 | 1.759849324 | — |
| 2 | 2 | 2.236205505 | — |
| 2 | — | — | 1.005253280 |
| 4 | 1 | 2.469222994 | — |
| 4 | 2 | 1.816113010 | — |
| 4 | — | — | 0.613967772 |

Final checks and report generation:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=. \
  /workspace/venv311/bin/python results/argon/cpu_reduced/verify_artifacts.py
/workspace/venv311/bin/python results/argon/cpu_reduced/finish_notes.py
```

The integrity check passed: v0 is unchanged, both checksum sidecars match, every v1 tensor equals the final step-1,000 snapshot, and all 24 equivariance measurements and both test evaluations contain finite numbers or explicit nulls. The final snapshot also attained the lowest validation selection score among the four saved evaluations; it had already been chosen for export before test evaluation.

[Combined results](../results/argon/cpu_reduced/comparison.json), [v0 metrics](../results/argon/cpu_reduced/v0/metrics.json), [v1 metrics](../results/argon/cpu_reduced/v1/metrics.json), [equivariance measurements](../results/argon/cpu_reduced/equivariance.json).
