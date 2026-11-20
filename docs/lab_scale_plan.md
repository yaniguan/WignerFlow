# WignerFlow lab-scale plan

Status: plan. Nothing below the "What exists today" section has been run yet.

## Role in the lab agent

The lab agent has three parts: YLM is the brain (reasons over protocols), ouroboros-ocsr is the eyes (turns pages and images into structures), and WignerFlow is the physics sense. WignerFlow is a fast generative surrogate for molecular dynamics. The goal: give the agent a cheap feel for how molecules move, and distill expensive simulation into a small model that can run next to the robot.

Analogy: a robot that only drills moves is an athlete who only drills dunks; WignerFlow is the physics class that lets it plan new moves.

## What exists today

Facts from this repo. Argon only for the trained model.

| Item | Fact | Source |
|---|---|---|
| Model | E(3)-equivariant net, jumps 50 fs per step, no integrator | [README.md](../README.md) |
| Release checkpoint | `argon_flow_v0.pt`: LJ argon, 256 atoms, flow head, 7.9 MB | [checkpoints/README.md](../checkpoints/README.md) |
| Stability | 4 of 5 seeds run 10⁴ steps without a crash | [docs/release_v0.md](release_v0.md), [results/argon/B_S3a_seeds.md](../results/argon/B_S3a_seeds.md) |
| All v0 criteria | met by 2 of 5 seeds | same |
| Released checkpoint temperature | −6.0% vs truth (fails the 5% criterion) | [docs/release_v0.md](release_v0.md) |
| Speed, one A100, batch 1 | 25.8 ns/day at 10 flow steps | [results/profiling/a100_oct29/profile.log](../results/profiling/a100_oct29/profile.log) |
| Speed vs classical MD | Slower than OpenMM on CPU for argon | [README.md](../README.md), [docs/SPEED.md](SPEED.md) |
| Flow-step distillation | Code exists (4→2→1), not run on a release model | [equitraj/distill.py](../equitraj/distill.py), [scripts/distill.py](../scripts/distill.py) |
| bf16 inference path | Exists, opt-in | [equitraj/fast.py](../equitraj/fast.py), [docs/SPEED.md](SPEED.md) |
| Small molecules | Ground truth only (ala2, aspirin); no model rollouts | [RESULTS.md](../RESULTS.md) |
| MD17/MD22 test | Coarse-step baseline with a simple pair model, **not** the trained network | [docs/PURE_MD_MD17_BENCHMARK.md](PURE_MD_MD17_BENCHMARK.md) |

New in this plan (scaffold, see [lab/](../lab/)):

- [lab/surrogate_query.py](../lab/surrogate_query.py): JSON query in, JSON report out. Refuses any system without a trained checkpoint.
- [lab/distill_edge.py](../lab/distill_edge.py) + [lab/configs/distill_edge.json](../lab/configs/distill_edge.json): teacher → student plan with `--dry-run`.
- [tests/test_lab_scale.py](../tests/test_lab_scale.py): offline tests, including a real one-step rollout from the release checkpoint.

## Lab tasks it serves

| Task | Input | Output | Benchmark |
|---|---|---|---|
| Physics sanity check (argon today) | system name, steps, seed | crash step, mean T vs reference, closest pair | v0 criteria in [scripts/evaluate.py](../scripts/evaluate.py) |
| Step debugging support | a YLM question about one step | `[physics]` context lines, or "unsupported" | TBD (none in repo) |
| Molecule sense from OCSR | SMILES from ouroboros-ocsr | lookup to a supported system, else refuse | TBD |
| Small-molecule dynamics (planned) | 3-frame context of ala2 / aspirin | short trajectory, h(r), VDOS | metrics of [docs/PURE_MD_MD17_BENCHMARK.md](PURE_MD_MD17_BENCHMARK.md), not run with the network yet |

Protocol benchmarks (LAB-Bench, BioProBench, BioPlanner) score the brain, not this repo. They run in [labbench-posttrain](https://github.com/yaniguan/labbench-posttrain) and YLM.

## Plan

| M | Goal | Data | Method | Metric | Done when |
|---|---|---|---|---|---|
| M1 | Query API on argon | release checkpoint | `lab/surrogate_query.py` | schema tests; refuses unknown systems | YLM reads `[physics]` lines from a JSON call |
| M2 | Few-step argon student | argon training contexts ([configs/data/argon.yaml](../configs/data/argon.yaml)) | 4-step teacher, then 4→2→1 with `scripts/distill.py` | v0 criteria + steps/s | 1- or 2-step student passes v0 gate on val, then test; results in `results/` |
| M3 | First small molecule | ala2 or aspirin ground truth ([results/ala2](../results/ala2), [results/aspirin](../results/aspirin)) | train flow head with bonds | crash rate, h(r), VDOS | a model rollout `metrics.json` exists for one molecule |
| M4 | Edge package | M2 student | bf16, then int8 export | latency + v0 gate on device | one edge box answers a query end to end |

All metric values: TBD / not run yet.

## Distillation and edge path

```
teacher (argon_flow_v0, 10 flow steps)
  -> 4-step teacher (validate)          # 10 cannot halve exactly (docs/SPEED.md)
  -> 2-step -> 1-step student           # scripts/distill.py
  -> narrow student (fewer channels)    # todo
  -> bf16 (exists) -> int8 / edge export (todo)
```

- Every stage must pass the v0 gate before the next. Distillation does not guarantee physics ([docs/SPEED.md](SPEED.md)).
- Teacher size: run `python -m lab.distill_edge --inspect-teacher` (prints the parameter count from the file).
- Student size target: TBD. Latency target: TBD.
- Target hardware: Jetson Orin Nano Super, Jetson AGX Orin 64GB, or one 24GB GPU next to the robot. None tested yet.
- The model is small, so per-step compute is the limit, not memory. Fewer flow steps is the main lever.

## Cost

Shared cost doc: [YLM docs/lab_cost_setup.md](https://github.com/yaniguan/YLM/blob/main/docs/lab_cost_setup.md). Estimates, Nov 2026; prices move daily.

Single-GPU plan (one H200 or B200 kept busy, with this repo's calls batched into the same loop): [YLM docs/single_gpu_saturation.md](https://github.com/yaniguan/YLM/blob/main/docs/single_gpu_saturation.md).

| Need | Lean choice | Price (estimate) | Source |
|---|---|---|---|
| Dev, tests | laptop CPU | $0 | — |
| M2/M3 training | rented L4 / RTX 4090 bursts | L4 from $0.44/hr; 4090 ~$0.34/hr | cirrun.com/gpu/l4.html, runailocal.dev/gpu-prices |
| Long eval rollouts | rented A100 80GB | $1.19–1.39/hr | runpod.io/pricing |
| M4 edge test | one Jetson Orin Nano Super | $399 | developer.nvidia.com/embedded/faq |

- Do not buy a GPU box for this repo. A used 4090 (~$3k) equals ~8,800 rented 4090 hours at $0.34/hr.
- Edge tests run on a rented 24GB GPU first; no hardware purchase in this plan.

## Risks

- Argon only. Nothing says it transfers to molecules. The query API must refuse out-of-domain systems.
- Checkpoint choice matters: temperature swings between checkpoints ([README.md](../README.md)).
- Few-step students may lose fidelity; the gate may fail.
- On CPU it is slower than classical MD for argon. Value comes only where reference MD is expensive.
- Licenses: this repo is Apache-2.0. MD17/MD22 data come from sGDML archives; check terms before commercial use. LAB-Bench (CC-BY-SA-4.0, share-alike) and BioProBench (CC-BY-NC-4.0, **non-commercial**) are used on the YLM side; a startup must flag this.

## Next steps

Each step is one GitHub issue:

1. Wire the surrogate query into the YLM reasoner context (M1).
2. Distill the argon flow head to 1–2 flow steps and gate on v0 (M2).
3. Train and evaluate a first small-molecule model (M3).
4. Run the query API on an edge box (M4).
