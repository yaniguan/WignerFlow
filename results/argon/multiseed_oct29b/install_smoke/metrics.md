# Multistart / multiseed evaluation

Per-rollout metrics; paired explicit RNG seeds across checkpoints. Percentile 95% cluster bootstrap of means over starts, retaining all seeds in each sampled start. Physical metrics use evaluate.py analysis conventions.

Physical metrics condition on survival; missing VACF zeros are excluded and counted. Errors are fractions; RDF L1 is in Å.

| Checkpoint | Metric | Mean | Bootstrap 95% CI | Valid / total |
|---|---|---:|---|---:|
| argon_flow_v1_cpu_reduced.pt | temperature_rel_err | 0.026995474 | [0.026995474, 0.026995474] | 1 / 1 |
| argon_flow_v1_cpu_reduced.pt | rdf_l1 | 0.43598123 | [0.43598123, 0.43598123] | 1 / 1 |
| argon_flow_v1_cpu_reduced.pt | vacf_zero_rel_err | null | [null, null] | 0 / 1 |
| argon_flow_v1_cpu_reduced.pt | D_rel_err | 0.0073662927 | [0.0073662927, 0.0073662927] | 1 / 1 |
| argon_flow_v1_cpu_reduced.pt | crash_fraction | 0 | [0, 0] | 1 / 1 |
| argon_flow_v1_cpu_reduced.pt | steps_survived | 8 | [8, 8] | 1 / 1 |

Measured total wall time: 5.706 s.

These intervals quantify start-window sampling, conditional on these checkpoints and test trajectories. Overlapping windows and few starts limit coverage. Short-time diffusion need not be in the diffusive regime; this is not a release certification.
