# Review round 1 (2026-10-20)

An external review of the repository raised specific points. Each one that can be checked in the code was checked. This file records what
was confirmed, what was changed, and what is still open. Literature claims in the review (newer versions of TrajCast, FlashMD, Hamiltonian Flow
Maps, TITO) are not verified here; `docs/related_work.md` must be updated against the primary sources before any submission.

## Confirmed in the code, and fixed on branch `fix/review-round1`

| Claim | Check | Change |
|---|---|---|
| The released checkpoint (seed 1, step 10000) and the best seed 4 checkpoint (step 10000) saw no unrolled training | `unroll_length` returns m = 1 for training steps 0–9999 and the schedule switches at 10000 | Documented as an erratum (below). Ablation runs `B_S1`, `B_S2` (no unroll), seeds 1–3 submitted, to compare with `B_S3a` |
| Softmax is taken over neighbours, then multiplied by the envelope, so the output jumps when an edge crosses the cutoff | `GraphAttention.forward`; `tests/test_review_fixes.py` shows a jump at the cutoff more than 10x larger than the smooth variant, for both the equivariant and the plain model | New option `model.smooth_attention` (envelope inside the softmax, plus a sink term). Default off so v0 checkpoints are unchanged. Runs `B_S3a_smooth` seeds 1–3 submitted |
| Temperatures of molecules ignore the constraints (ala2: 12, aspirin: 8) | `temperature()` was always called with 0 constraints; ala2 uses 63 degrees of freedom instead of 51, aspirin 60 instead of 52 | Constraint count comes from the data metadata and is passed to every temperature call, the crash test, and CSVR. Model-to-truth ratios were unaffected because both used the same wrong count |
| The last default validation start had one frame of ground truth, yet was ranked on a long rollout | `validation_rollout` used evenly spaced indices over all windows | Starts are now taken only from windows with enough ground truth, and an error is raised if there is none |
| `use_velocity=False` still feeds the true velocity through the residual output (`v_last`) | `loss` and `predict_with_grad` used `vel[:, -1]` | `base_velocity()`: for a position-only model the baseline velocity comes from the positions. A test checks that outputs do not change with the input velocities. Runs `B_S3a_novel` seeds 1–3 submitted |
| No energy evaluation of generated trajectories | `evaluate.py` had none | `equitraj/energy.py` (argon LJ with the OpenMM switch, tested against OpenMM) and an `energy` block in `evaluate.py`: U, K, total energy offset, drift per ns per degree of freedom |

## New diagnostics

- `scripts/kinetic_diagnostics.py`: splits the one-step kinetic energy change into cross, squared and variance terms from many flow samples of one context,
  and compares with the true step. First check on 4 contexts: the flow's variance term is about 10^-4 of the cross and squared terms and the mean velocity change
  has cosine similarity 0.9998 with the true one, so on a full (x, v) state the flow is nearly deterministic. This needs the full-size run before it is stated anywhere.
- `scripts/evaluate.py --sampling_steps N` on the released checkpoint for N = 2, 5, 20 (submitted) tests whether the stability depends on the flow solver.

## Erratum for v0.1.0

The released checkpoint was trained for 10000 steps with teacher forcing and input noise only; the multi-step unroll schedule starts at step 10000.
The v0 text that presents unrolled training as part of what makes the rollout stable is therefore not supported by that checkpoint. The ablation above decides it.

## Open items from the review, not done yet

- Observation-completeness matrix (full state vs positions only, deterministic vs flow, k = 1 vs 3) with bifurcated reference trajectories.
- A calibrated-noise control for the regression head against the flow head.
- Horizon normalised by the reference system's perturbation growth (Lyapunov time), instead of the fixed 200 fs.
- Fresh sealed test trajectories; 10 seeds for the main method and the strongest baseline.
- Green–Kubo diffusion next to the MSD diffusion; time-composition and reversal consistency checks.
- Strong baselines: a real MACE + Verlet (blocked: `mace-torch` import untested on the cluster), TrajCast, FlashMD, an action-learning control.
- Flow cost against force-field cost on equal hardware, with a dynamics-error versus wall-time curve.
- Related work rewrite against the newest versions of the cited papers.
