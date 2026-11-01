# Multistart / multiseed evaluation

Per-rollout metrics; paired explicit RNG seeds across checkpoints. Percentile 95% cluster bootstrap of means over starts, retaining all seeds in each sampled start. Physical metrics use evaluate.py analysis conventions.

Physical metrics condition on survival; missing VACF zeros are excluded and counted. Errors are fractions; RDF L1 is in Å.

| Checkpoint | Metric | Mean | Bootstrap 95% CI | Valid / total |
|---|---|---:|---|---:|
| argon_flow_v0.pt | temperature_rel_err | 0.1558561 | [0.1545346, 0.15682615] | 6 / 6 |
| argon_flow_v0.pt | rdf_l1 | 0.53286719 | [0.46451996, 0.613808] | 6 / 6 |
| argon_flow_v0.pt | vacf_zero_rel_err | 0.073578583 | [0.040960953, 0.13469149] | 6 / 6 |
| argon_flow_v0.pt | D_rel_err | 0.021814484 | [0.0060705275, 0.050879195] | 6 / 6 |
| argon_flow_v0.pt | crash_fraction | 0 | [0, 0] | 6 / 6 |
| argon_flow_v0.pt | steps_survived | 50 | [50, 50] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | temperature_rel_err | 0.074289966 | [0.05977829, 0.083340193] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | rdf_l1 | 0.21452979 | [0.21047435, 0.21866377] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | vacf_zero_rel_err | 0.09285379 | [0.031304672, 0.18559472] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | D_rel_err | 0.057614679 | [0.042335306, 0.070293299] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | crash_fraction | 0 | [0, 0] | 6 / 6 |
| argon_flow_v1_cpu_reduced.pt | steps_survived | 50 | [50, 50] | 6 / 6 |

Measured total wall time: 719.681 s.

These intervals quantify start-window sampling, conditional on these checkpoints and test trajectories. Overlapping windows and few starts limit coverage. Short-time diffusion need not be in the diffusive regime; this is not a release certification.
