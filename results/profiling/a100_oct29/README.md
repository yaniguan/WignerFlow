# A100 profiling, argon (Oct 29 full-data run)

`profile.log` is copied unchanged from the Colab run output on Google Drive
(`WignerFlow_runs/argon_full_oct29/train_stable_ft/profiling/profile.log`), written by
`scripts/profile_run.py` on an A100-80GB Colab GPU.

- Rollout: 5.97 steps/s × 50 fs stride = 25.8 ns/day, batch 1, 256 atoms, 10 flow steps per model step.
- Training: 1.91 s/step (batch 8, unroll m = 4), peak memory 50.2 GB.

The GPU model is not recorded in the log itself; it comes from the run notes in
[`docs/ARGON_STABLE_FT.md`](../../../docs/ARGON_STABLE_FT.md).
