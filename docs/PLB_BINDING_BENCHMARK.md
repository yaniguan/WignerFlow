# Ground-Truth Protein–Ligand Pocket MD Benchmark (`OpenForceField / Wang` X-Ray Set)

`equitraj/plb_bench.py` and `scripts/bench_plb_binding.py` evaluate `PocketMicroEngine` + `OpenBPMD CompScore` + mass-calibrated `50-fs tau-RAMD` across real crystal structures (`CDK2`, `MCL1`, `Thrombin`, `TYK2`):

- **Pose Discrimination (`n = 40` complexes: `20` native crystal vs `20` 180°-flipped steric decoys)**:
  - Trajectory `CompScore` alone achieves **AUROC = `1.000`** (`native = -3.087` vs `decoy = -0.764`).
  - Mass-calibrated `50-fs tau-RAMD` yields **`+54.2%` longer unbinding residence time** for native poses (`1.109 ps` vs `0.719 ps`).
- **Throughput**: **`0.406 s / pose` on CPU** (`57,628×` faster than `3×10 ns` explicit-solvent `OpenBPMD` at `~6.5 GPU-hours`).
- Full benchmark output is archived in `results/binding_plb/off_plb_40complex_report.json`.
