"""Write one training config per experiment of the one-factor-at-a-time grid (PLAN.md, "Options to test").

    python scripts/make_configs.py argon

Each experiment starts from the default config and changes one thing. Configs go to configs/<system>/.
"""

import copy
import os
import sys

import yaml

SYSTEMS = {
    "argon": {"data_dir": "data/argon", "cutoff": 6.0, "stride_steps": 10, "strides": [1, 5, 10, 50], "dt_fs": 5.0},
}

# Unroll schedules: [[training step, m], ...].
# The unroll horizon m * stride must stay below the time over which the model's trajectory and the
# ground-truth trajectory stay close. Beyond it, the target "true frame minus model frame" is no longer
# a learnable correction. With m = 16 at 50 fs (0.8 ps), the first argon run diverged (loss ~1e5, all
# rollouts crashed). So m is capped at UNROLL_HORIZON_FS / stride.
UNROLL_HORIZON_FS = 200.0


def unroll_schedule(stride_fs, horizon_fs=UNROLL_HORIZON_FS):
    m_max = max(1, min(16, int(horizon_fs // stride_fs)))
    sched = [[0, 1]]
    if m_max >= 2:
        sched.append([10000, min(2, m_max)])
    if m_max > 2:
        sched.append([20000, m_max])
    return sched


# Model size: 64 channels, measured on a V100 (argon, batch 8): 0.34 s per teacher-forcing step,
# 9.7 GB. 128 channels with a 7 Å cutoff took 1.23 s and filled the 32 GB card.
def default_config(system):
    s = SYSTEMS[system]
    return {
        "data_dir": s["data_dir"],
        "stride_steps": s["stride_steps"],
        "k": 3,
        "seed": 0,
        "model": {"head": "flow", "lmax": 2, "channels": 64, "heads": 8, "n_rbf": 32, "cutoff": s["cutoff"],
                  "n_spatial": 2, "n_fusion": 2, "n_denoiser": 2, "parity": True, "use_velocity": True,
                  "target": "residual"},
        "train": {"batch_size": 8, "lr": 3.0e-4, "steps": 40000, "warmup": 1000, "grad_clip": 1.0,
                  "noise_sigma": 0.01, "unroll": unroll_schedule(s["stride_steps"] * s["dt_fs"]),
                  "backprop": "truncated", "unroll_sampling_steps": 2,
                  "workers": 4, "start_every": 1, "ckpt_minutes": 20},
        "eval": {"every": 2000, "val_batches": 20, "rollout_steps": 1000, "rollout_starts": 4, "sampling_steps": 10},
    }


def experiments(system):
    """(name, function that edits a config) for every experiment."""
    exps = []
    for head, tag in [("flow", "B"), ("regression", "A")]:
        def set_head(c, head=head):
            c["model"]["head"] = head
        # strategy ladder S1 -> S3b
        exps.append((f"{tag}_S1", [set_head, lambda c: c["train"].update(noise_sigma=0.0, unroll=[[0, 1]])]))
        exps.append((f"{tag}_S2", [set_head, lambda c: c["train"].update(unroll=[[0, 1]])]))
        exps.append((f"{tag}_S3a", [set_head]))
        exps.append((f"{tag}_S3b", [set_head, lambda c: c["train"].update(backprop="full")]))
        # residual targets (default) vs direct Δx, v_new
        exps.append((f"{tag}_S3a_direct", [set_head, lambda c: c["model"].update(target="direct")]))
        # velocity input vs positions only
        exps.append((f"{tag}_S3a_novel", [set_head, lambda c: c["model"].update(use_velocity=False)]))
        # stride sweep
        for n in SYSTEMS[system]["strides"]:
            if n != SYSTEMS[system]["stride_steps"]:
                dt = SYSTEMS[system]["dt_fs"]
                exps.append((f"{tag}_S3a_n{n}", [set_head, lambda c, n=n, dt=dt: (c.update(stride_steps=n),
                                                                                 c["train"].update(unroll=unroll_schedule(n * dt)))]))
    # one-factor tests around the default (head B, S3a)
    for sigma in [0.005, 0.02, 0.04]:
        exps.append((f"B_S3a_noise{sigma}", [lambda c, s=sigma: c["train"].update(noise_sigma=s)]))
    # longer unroll horizon (400 fs, m = 8 at 50 fs) to test the horizon cap itself
    exps.append(("B_S3a_h400", [lambda c: c["train"].update(unroll=unroll_schedule(c["stride_steps"] * SYSTEMS[system]["dt_fs"], 400.0))]))
    for lmax in [1, 3]:
        exps.append((f"B_S3a_lmax{lmax}", [lambda c, l=lmax: c["model"].update(lmax=l)]))
    for k in [1, 6]:
        exps.append((f"B_S3a_k{k}", [lambda c, k=k: c.update(k=k)]))
    return exps


def main():
    system = sys.argv[1]
    out = os.path.join("configs", system)
    os.makedirs(out, exist_ok=True)
    names = []
    for name, edits in experiments(system):
        c = copy.deepcopy(default_config(system))
        for edit in edits:
            edit(c)
        c["name"] = name
        c["out_dir"] = f"results/{system}/{name}"
        with open(os.path.join(out, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(c, f, sort_keys=False)
        names.append(name)
    print(f"{len(names)} configs in {out}: {' '.join(names)}")


if __name__ == "__main__":
    main()
