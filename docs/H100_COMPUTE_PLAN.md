# H100 GPU Compute Budget and Benchmark Roadmap

This document derives a bottom-up NVIDIA H100 SXM5 (80 GB) compute estimate from the measured A100-80GB profiler trace in `results/profiling/a100_oct29/profile.log`, combined with the memory and mixed-precision upgrades in this branch (`amp: bf16`, cached `per_l_scale` index gather, `gradient_checkpointing`, and `neighbor_method: auto`).

---

## 1. Anchored Calibration from Measured A100 Profiling

From `results/profiling/a100_oct29/profile.log` (`256` Ar atoms, `k = 3`, `cutoff = 6.0 Å`, `4,184` directed edges/frame, `1,945,520` parameters, `batch_size = 8`, `m = 4`, pure `FP32` eager):

| Phase / Metric | Measured A100-80GB (`FP32`) | H100 SXM5 (`BF16` + `TF32` + Gather Fix) | Scaling Basis |
| :--- | :---: | :---: | :--- |
| Train step (`m = 4`, truncated) | `1,906.6 ms` | `~730 ms` (`2.6x`) | `85.3%` `SO2Linear`/MLP GEMMs (`BF16`) + `3.35 vs 2.0 TB/s` HBM3 scatter/gather |
| Train step (`m = 2`) | `~1,030 ms` | `~400 ms` (`2.6x`) | 1 unroll step + 1 flow loss pass |
| Train step (`m = 1`, teacher forcing) | `~600 ms` | `~230 ms` (`2.6x`) | Single `encode` + 1 flow loss pass |
| Rollout step (`B = 1`, `S = 10`) | `167.4 ms` (`25.8 ns/day`) | `~65 ms` (`~66 ns/day`) | Single `encode` + 10 `Denoiser` passes |
| Distilled rollout (`B = 1`, `S = 2`) | `~55 ms` | `~22 ms` (`~196 ns/day`) | Single `encode` + 2 `Denoiser` passes |
| Peak VRAM (`B = 8, m = 4`, no ckpt) | `50.20 GB` | `~27 GB` (`~14 GB` with ckpt) | `BF16` activations + removed `repeat_interleave` copy + optional block checkpointing |

*Note on PCIe H100 vs SXM5 H100*: PCIe H100 (`2.0 TB/s` HBM2e, `350 W`) is roughly `1.25x` slower than SXM5 H100 (`3.35 TB/s` HBM3, `700 W`) on scatter-heavy GNN layers; multiply SXM5 estimates below by `1.25x` if running on PCIe nodes.

---

## 2. Per-System Single-Seed H100 Cost (`40,000` Steps + Validation + `15 x 10^4` Evaluation)

| System & Model | Atoms | Edges / Frame | Train (`40k` steps) | In-Loop Val (`20x`) | Final `15 x 10^4` Eval | **1-Seed Total** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Argon `EquiTraj` (Flow, `v1`)** | `256` | `~4,180` | `5.8 h` | `2.2 h` | `0.6 h` | **`8.6 H100-h`** |
| **Argon `TrajCastBaseline` (CG, `k=1`)** | `256` | `~4,180` | `2.4 h` | `0.6 h` | `0.2 h` | **`3.2 H100-h`** |
| **Alanine Dipeptide `EquiTraj` (Flow)** | `22` | `~460` | `1.4 h` | `0.5 h` | `0.3 h` | **`2.2 H100-h`** |
| **Alanine Dipeptide `TrajCastBaseline`** | `22` | `~460` | `0.7 h` | `0.2 h` | `0.1 h` | **`1.0 H100-h`** |
| **Aspirin / MD17-CCSD `EquiTraj` (Flow)** | `21` | `~420` | `1.3 h` | `0.5 h` | `0.3 h` | **`2.1 H100-h`** |
| **`Li+/EC/PF6-` `EquiTraj` (`C=96`, ckpt)** | `1,008` | `~24,000` | `38.0 h` | `5.5 h` | `2.5 h` | **`46.0 H100-h`** |
| **`Li+/EC/PF6-` `TrajCastBaseline` (`C=96`)** | `1,008` | `~24,000` | `16.0 h` | `2.0 h` | `1.0 h` | **`19.0 H100-h`** |

Why small covalent molecules (`ala2`, `aspirin`, `md17_aspirin`) are `~4x` cheaper than `256`-atom Argon: graph attention and Wigner/CG operations scale linearly with edge count `E`, and `21-22` atom molecules have `~9x` fewer edges per frame than `256`-atom liquid argon.

---

## 3. Three Execution Tiers

### Tier 1 — Minimum Viable Counter-Critique (`~82 H100 GPU-hours`)
Closes every major scientific gap (replaces legacy `v0.1.0` Argon checkpoint, proves covalent-bond & dihedral dynamics on `ala2`, proves real quantum AIMD speedup on `MD17` CCSD Aspirin, and adds a head-to-head `TrajCast` baseline):

1. OpenMM NVE data generation (`argon_full`, `ala2`, `aspirin`) + `MD17` CCSD import: **`0.5 h`**
2. `Argon` `EquiTraj` `v1` retrain (`5` seeds x `8.6 h`): **`43.0 h`**
3. `Alanine dipeptide` `EquiTraj` (`5` seeds x `2.2 h`, evaluated model-only + S4 RATTLE + S6 CSVR): **`11.0 h`**
4. `MD17` CCSD `Aspirin` `EquiTraj` (`3` seeds x `2.1 h`, `10 fs` jump = `20x` quantum MD step): **`6.3 h`**
5. Head-to-head `TrajCastBaseline` (`3` seeds on `Argon` + `3` seeds on `Ala2` + `3` seeds on `MD17 Aspirin`): **`15.6 h`**
6. Progressive distillation (`10 -> 4 -> 2 -> 1` flow steps on `Argon` & `Ala2`) + `bench_speed.py` H100 sweep: **`5.6 h`**
- **Total Tier 1**: **`~82 H100 GPU-hours`** (`~20.5 hours` wall-clock on `4x H100`).

### Tier 2 — Full Systematic Ablation & Multi-Seed Suite (`~165 H100 GPU-hours`)
Adds full `5`-seed coverage across `PlainTraj`, `EquiFF` + Verlet, regression vs flow heads, stride sweep `n in {1, 5, 10, 25, 50}`, context history `k in {1, 3, 6}`, `l_max in {1, 2, 3}`, and unconstrained `0.5 fs` `Ala2`:
- Tier 1 (`82 h`) + `Argon` & `Ala2` ablation grid (`24` runs, mix of `2.2 h` `Ala2` and `6.5 h` `Argon` runs = `~83 h`)
- **Total Tier 2**: **`~165 H100 GPU-hours`**.

### Tier 3 — Condensed-Phase `1,008`-Atom `Li+/EC/PF6-` Electrolyte (`~490 H100 GPU-hours`)
Adds full explicit-solvent periodic electrolyte transport (`CN(t)`, residence time, `Li-O` RDF, diffusion):
- Tier 2 (`165 h`)
- OpenMM PME `Li+/EC/PF6-` ground-truth trajectories (`12 x 0.5 ns` + `2 x 20 ns`): **`6 h`**
- Small-model (`C=64`, `20k` steps) stride & stabilization sweep (`6` runs x `16 h`): **`96 h`**
- Full `Li+/EC/PF6-` `EquiTraj` (`3` seeds x `46 h`) + `TrajCastBaseline` (`3` seeds x `19 h`) + `PlainTraj` (`1` seed x `28 h`): **`223 h`**
- **Total Tier 3**: **`~490 H100 GPU-hours`**.
