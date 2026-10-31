"""Training loop: teacher forcing (S1), input noise (S2), unrolled training (S3a truncated, S3b full).

    python -m equitraj.train --config configs/argon/flow_s1_n10.yaml

Resumable: a checkpoint is written every `ckpt_minutes` and at every evaluation; rerunning the
same command continues from it. Model selection uses short validation rollouts, not the loss:
best.pt by legacy_score, best_stable.pt by selection_score (stability first). Optional stability
options (repulsion prior, noise schedule, unroll weights, overlap penalty) are documented in docs/STABILITY.md.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import yaml
from torch.utils.checkpoint import checkpoint

from equitraj import analysis
from equitraj.dataset import TrajWindows, compute_stats
from equitraj.model import EquiTraj
from equitraj.monitor import REASONS
from equitraj.prior import estimate_rmin, overlap_penalty
from equitraj.rollout import rollout, temperature


def unroll_length(schedule, step):
    """schedule: [[step, m], ...] sorted by step; returns the m in effect at this training step."""
    m = schedule[0][1]
    for s, mm in schedule:
        if step >= s:
            m = mm
    return m


def noise_sigma_at(tc, step):
    """Input-noise level at this step: train.noise_schedule [[step, sigma], ...] interpolated linearly
    (constant before the first and after the last point), else the constant train.noise_sigma."""
    sched = tc.get("noise_schedule")
    if not sched:
        return tc.get("noise_sigma", 0.0)
    steps = [float(s) for s, _ in sched]
    sigmas = [float(v) for _, v in sched]
    return float(np.interp(step, steps, sigmas))


def lr_scale_after_switch(step, switch_step, rewarmup):
    """LR multiplier after an unroll-length switch: linear 0.1 -> 1 over `rewarmup` steps (1 when disabled)."""
    if not rewarmup or switch_step is None or step - switch_step >= rewarmup:
        return 1.0
    return 0.1 + 0.9 * max(0, step - switch_step) / float(rewarmup)


def last_switch_step(schedule, step):
    """Training step at which the unroll length in effect at `step` started (None for the first entry)."""
    sw = None
    m_prev = schedule[0][1]
    for s, mm in schedule:
        if s <= step and mm != m_prev:
            sw = s
        if s <= step:
            m_prev = mm
    return sw


class SpikeGuard:
    """Skips optimizer updates whose loss is non-finite or > factor x the running median of the last `window`
    accepted losses at the current unroll length. History is reset when m changes; guarding starts after
    `min_history` accepted losses. factor <= 0 disables the guard (only non-finite losses are skipped)."""

    def __init__(self, factor=0.0, window=200, min_history=50):
        self.factor, self.window, self.min_history = float(factor or 0.0), int(window), int(min_history)
        self.hist, self.m, self.skipped = [], None, 0

    def check(self, loss_value, m):
        if m != self.m:
            self.hist, self.m = [], m
        if not math.isfinite(loss_value):
            self.skipped += 1
            return False
        if self.factor > 0 and len(self.hist) >= self.min_history:
            if loss_value > self.factor * float(np.median(self.hist)):
                self.skipped += 1
                return False
        self.hist.append(loss_value)
        self.hist = self.hist[-self.window:]
        return True


def load_init_weights(model, path, device):
    """Initialise `model` from a training checkpoint/snapshot (fine-tuning). Non-strict only for the optional
    repulsion-prior buffer, so snapshots trained without the prior load into a model that has it.
    Returns (missing_keys, the source step)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    sd = ck["model"]
    res = model.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("prior.")] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"init_from {path}: incompatible keys {bad}")
    absent = [k for k in model.state_dict() if k not in sd]  # e.g. prior.rmin (the model may tolerate it itself)
    return absent, ck.get("step")


def unroll_weights_for(weights, m, like):
    """Per-unroll-step loss weights (m,): the list is cut to m entries or padded with its last value."""
    w = [float(x) for x in weights][:m]
    w = w + [w[-1]] * (m - len(w))
    return torch.tensor(w, dtype=like.dtype, device=like.device)


def training_loss(model, pos, vel, box, types, masses, m, noise_sigma, backprop, unroll_sampling_steps,
                  unroll_weights=None, noise_v_ratio=1.0, noise_type_scale=None, overlap_weight=0.0, overlap_r0=None):
    """Loss for one batch. pos, vel: (B, k + m_max, N, 3); only the first k + m frames are used.

    m = 1 is teacher forcing. For m > 1 the model first predicts m-1 frames from its own outputs:
      truncated: those predictions carry no gradient; the loss is on the last step only
      full: gradients flow through all predictions; the loss is averaged over all m steps
    Input noise (S2) is added to the context frames; the Δx target is taken relative to the
    (noisy or predicted) last context frame, so the model learns to correct its input.

    Optional (defaults reproduce the behavior above exactly):
      unroll_weights: list of per-step weights; the loss is evaluated at every unroll step (in truncated
          mode on detached contexts) and averaged with these weights, so later steps can count more
      noise_v_ratio: velocity noise = noise_v_ratio * position noise level (both relative to dx_std / v_std)
      noise_type_scale: (n_types,) multiplier on the noise of each element
      overlap_weight, overlap_r0: add overlap_weight * overlap_penalty on the predicted positions
          (every unrolled step in full mode, plus one differentiable prediction from the final context);
          overlap_r0 is the (T, T) wall distance in Å
    """
    k = model.k
    b = pos.shape[0]
    ctx_x, ctx_v = pos[:, :k].clone(), vel[:, :k].clone()
    if noise_sigma > 0:
        tb = types.repeat(b).view(b, 1, -1, 1)
        nx = torch.randn_like(ctx_x) * noise_sigma * model.dx_std[tb]
        nv = torch.randn_like(ctx_v) * noise_sigma * model.v_std[tb]
        if noise_v_ratio != 1.0:
            nv = nv * noise_v_ratio
        if noise_type_scale is not None:
            ts = noise_type_scale.to(nx)[tb]
            nx, nv = nx * ts, nv * ts
        ctx_x = ctx_x + nx
        ctx_v = ctx_v + nv
    per_step = unroll_weights is not None
    losses, pred_x = [], []
    for j in range(m):
        last = (j == m - 1)
        if backprop == "full" or last or per_step:
            dx_t = pos[:, k + j] - ctx_x[:, -1]
            losses.append(model.loss(ctx_x, ctx_v, types, box, masses, dx_t, vel[:, k + j]))
        if last:
            break
        if backprop == "full":
            dx, v_new = checkpoint(model.predict_with_grad, ctx_x, ctx_v, types, box, masses,
                                   unroll_sampling_steps, use_reentrant=False)
        else:
            with torch.no_grad():
                dx, v_new = model.predict(ctx_x, ctx_v, types, box, masses, n_steps=unroll_sampling_steps)
        ctx_x = torch.cat([ctx_x[:, 1:], (ctx_x[:, -1] + dx)[:, None]], dim=1)
        ctx_v = torch.cat([ctx_v[:, 1:], v_new[:, None]], dim=1)
        if backprop == "full":
            pred_x.append(ctx_x[:, -1])
    if per_step:
        w = unroll_weights_for(unroll_weights, len(losses), losses[0])
        loss = (torch.stack(losses) * w).sum() / w.sum()
    else:
        loss = torch.stack(losses).mean()
    if overlap_weight > 0:
        dx, _ = model.predict_with_grad(ctx_x, ctx_v, types, box, masses, unroll_sampling_steps)
        pred_x.append(ctx_x[:, -1] + dx)
        pen = torch.stack([overlap_penalty(x, box, types, overlap_r0) for x in pred_x]).mean()
        loss = loss + overlap_weight * pen
    return loss


def validation_rollout(model, ds, n_starts, n_steps, sampling_steps, device, t_ref, diag_sink=None):
    """Short rollouts from evenly spaced validation windows. Returns stability and RDF metrics.

    diag_sink: optional list; the rollout's crash diagnostics are appended to it.
    """
    starts = np.linspace(0, len(ds) - 1, n_starts).astype(int)
    items = [ds[i] for i in starts]
    pos = torch.stack([it[0][:model.k] for it in items])
    vel = torch.stack([it[1][:model.k] for it in items])
    box = torch.stack([it[2] for it in items]) if ds.periodic else None
    every = max(1, n_steps // 50)
    out = rollout(model, pos, vel, ds.types, box, ds.masses, n_steps,
                  sampling_steps=sampling_steps, save_every=every, t_max=3 * t_ref)
    crash = out["crash_step"]
    crashed = crash >= 0
    metrics = {"val_crash_frac": crashed.float().mean().item(),
               "val_steps_before_crash": torch.where(crashed, crash, torch.full_like(crash, n_steps)).float().mean().item(),
               "val_rollout_wall_s": out["wall_s"]}
    diag = out["diagnostics"]
    for reason in REASONS:
        metrics[f"val_crash_{reason}"] = float(sum(r == reason for r in diag["crash_reason"]))
    dmin = [d for d in diag.get("min_pair_distance", []) if np.isfinite(d)]
    if dmin:
        metrics["val_min_pair_dist"] = float(min(dmin))
    if diag_sink is not None:
        diag_sink.append({"starts": starts.tolist(), "n_steps": n_steps,
                          **{key: val for key, val in diag.items() if key != "series"}})
    ok = torch.nonzero(~crashed).flatten().tolist()
    t_drift = diag.get("T_drift_K_per_ns", [])
    drift = [abs(t_drift[i]) for i in ok if i < len(t_drift) and np.isfinite(t_drift[i])]
    if drift:
        metrics["val_abs_T_drift_K_per_ns"] = float(np.mean(drift))
    if not ok or out["positions"] is None:
        return metrics
    metrics["val_T_end"] = temperature(out["velocities"][-1, ok], ds.masses).mean().item()
    metrics["val_T_rel_err"] = abs(metrics["val_T_end"] - t_ref) / t_ref
    # all-atom RDF of each rollout vs the ground-truth frames of the same trajectory over the same time span
    idx = np.arange(len(ds.types))
    l1 = []
    for bi in ok:
        t, s0 = ds.index[starts[bi]]
        gt = np.asarray(ds.pos[t][s0 + (model.k - 1) * ds.f: s0 + (model.k - 1 + n_steps) * ds.f + 1: every * ds.f])
        x = out["positions"][:, bi].numpy()
        bx = None if box is None else box[bi].numpy()
        r_max = 8.0 if bx is None else min(8.0, 0.49 * float(np.diag(bx).min()))
        r, g_model, _ = analysis.rdf(x, None if bx is None else np.tile(bx, (len(x), 1, 1)), idx, idx, r_max, 80)
        _, g_gt, _ = analysis.rdf(gt, None if bx is None else np.tile(bx, (len(gt), 1, 1)), idx, idx, r_max, 80)
        l1.append(analysis.rdf_l1(r, g_model, g_gt))
    metrics["val_rdf_l1"] = float(np.mean(l1))
    return metrics


def legacy_score(m):
    """Lower is better: crashes dominate, then RDF error. Selects best.pt (unchanged)."""
    return m["val_crash_frac"] * 100.0 + m.get("val_rdf_l1", 10.0)


def selection_score(m):
    """Stability-first ranking key for best_stable.pt; lower (tuple comparison) is better:
        1. crash fraction, 2. longer mean steps before crash, 3. RDF L1 + relative temperature error.
    Uses the longer selection rollouts (keys prefixed sel_, eval.rollout_steps_select) when present.
    Missing structural metrics (all rollouts crashed) count as a large error.
    """
    p = "sel_" if "sel_val_crash_frac" in m else ""
    err = m.get(p + "val_rdf_l1", 10.0) + m.get(p + "val_T_rel_err", 1.0)
    if not np.isfinite(err):
        err = 1e9
    return (float(m[p + "val_crash_frac"]), -float(m[p + "val_steps_before_crash"]), float(err))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda", help="cuda (default) or cpu/mps for small local tests")
    ap.add_argument("--no_tensorboard", action="store_true", help="keep JSON logs without the optional tensorboard dependency")
    args = ap.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        # fail loudly: a cluster job that silently trains on CPU wastes a day
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False (CPU-only torch build?)")
    cfg = yaml.safe_load(open(args.config))
    tc, ec = cfg["train"], cfg["eval"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    device = torch.device(args.device)
    torch.manual_seed(cfg.get("seed", 0))

    m_max = max(m for _, m in tc["unroll"])
    train_ds = TrajWindows(cfg["data_dir"], "train", cfg["stride_steps"], cfg["k"], m=m_max, start_every=tc.get("start_every", 1))
    val_ds = TrajWindows(cfg["data_dir"], "val", cfg["stride_steps"], cfg["k"], m=1)
    workers = tc.get("workers", 4)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=tc["batch_size"], shuffle=True, drop_last=True,
                                         num_workers=workers, persistent_workers=workers > 0)
    types, masses = train_ds.types.to(device), train_ds.masses.to(device)

    model = EquiTraj(n_types=len(train_ds.elements), k=cfg["k"], stride_fs=train_ds.stride_fs, **cfg["model"]).to(device)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    step, best, best_stable = 0, math.inf, None
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        best_stable = tuple(ck["best_stable"]) if ck.get("best_stable") is not None else None
        print(f"resumed from step {step}", flush=True)
    else:
        if model.prior is not None and not model.prior.rmin_fixed:
            model.prior.rmin.copy_(estimate_rmin(train_ds, model.prior.percentile))
        st = compute_stats(train_ds)
        model.dx_std.copy_(st["dx_std"])
        model.v_std.copy_(st["v_std"])
        residual = model.target == "residual"
        model.out_dx_std.copy_(st["res_dx_std"] if residual else st["dx_std"])
        model.out_v_std.copy_(st["res_v_std"] if residual else st["v_std"])
        if tc.get("init_from"):
            # fine-tune: weights and normalisation buffers come from the snapshot; a missing prior buffer
            # keeps the r_min just estimated from the training data
            missing, src_step = load_init_weights(model, tc["init_from"], device)
            print(json.dumps({"init_from": tc["init_from"], "init_source_step": src_step, "missing_keys": missing}), flush=True)
    print(json.dumps({"elements": train_ds.elements, "dx_std": model.dx_std.tolist(), "v_std": model.v_std.tolist(),
                      "out_dx_std": model.out_dx_std.tolist(), "out_v_std": model.out_v_std.tolist(), "target": model.target,
                      "n_params": sum(p.numel() for p in model.parameters()), "n_train_windows": len(train_ds)}), flush=True)
    if model.prior is not None:
        print(json.dumps({"repulsion_prior_rmin": model.prior.rmin.tolist(), "r0": model.prior.r0().tolist()}), flush=True)

    # optional stability options (absent keys keep the original behavior)
    loss_opts = {"unroll_weights": tc.get("unroll_loss_weights"), "noise_v_ratio": float(tc.get("noise_v_ratio", 1.0))}
    nts = tc.get("noise_type_scale")
    if nts is not None:  # dict element -> scale (missing elements: 1) or list in train_ds.elements order
        if isinstance(nts, dict):
            nts = [float(nts.get(e, 1.0)) for e in train_ds.elements]
        loss_opts["noise_type_scale"] = torch.tensor(nts, dtype=torch.float32, device=device)
    op = tc.get("overlap_penalty") or {}
    if op.get("weight", 0.0) > 0:
        if model.prior is not None:
            rmin = model.prior.rmin
        else:
            rmin = estimate_rmin(train_ds, op.get("percentile", 0.1)).to(device)
        loss_opts["overlap_weight"] = float(op["weight"])
        loss_opts["overlap_r0"] = rmin * float(op.get("rmin_scale", 1.0))
        print(json.dumps({"overlap_penalty_r0": loss_opts["overlap_r0"].tolist()}), flush=True)

    def lr_at(s):
        warm = tc.get("warmup", 1000)
        if s < warm:
            return tc["lr"] * (s + 1) / warm
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, tc["steps"] - warm))))

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best,
                    "best_stable": None if best_stable is None else list(best_stable), "cfg": cfg,
                    "elements": train_ds.elements}, path + ".tmp")
        os.replace(path + ".tmp", path)

    if args.no_tensorboard:
        class ScalarWriter:
            def add_scalar(self, *args, **kwargs):
                pass
        writer = ScalarWriter()
    else:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(os.path.join(out_dir, "tb"))
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    t_ref = temperature(torch.stack([val_ds[i][1][0] for i in range(0, len(val_ds), max(1, len(val_ds) // 20))]), val_ds.masses).mean().item()
    last_ckpt, t_start, running = time.time(), time.time(), []
    guard = SpikeGuard(tc.get("loss_spike_factor", 0.0), tc.get("loss_spike_window", 200))
    rewarmup = int(tc.get("rewarmup_steps", 0))
    gnorms = []
    def evaluate_and_select():
        nonlocal best, best_stable, last_ckpt
        model.eval()
        with torch.no_grad():
            val = []
            for i in np.linspace(0, len(val_ds) - 1, ec.get("val_batches", 20) * tc["batch_size"]).astype(int).reshape(-1, tc["batch_size"]):
                items = [val_ds[j] for j in i]
                vp = torch.stack([it[0] for it in items]).to(device)
                vv = torch.stack([it[1] for it in items]).to(device)
                vb = torch.stack([it[2] for it in items]).to(device) if val_ds.periodic else None
                val.append(training_loss(model, vp, vv, vb, types, masses, 1, 0.0, "truncated", 2).item())
            metrics = {"step": step, "val_loss": float(np.mean(val))}
            diags = []
            metrics.update(validation_rollout(model, val_ds, ec.get("rollout_starts", 4), ec.get("rollout_steps", 500),
                                              ec.get("sampling_steps", 10), device, t_ref, diag_sink=diags))
            if ec.get("rollout_steps_select"):  # longer rollouts used only for best_stable.pt
                sel = validation_rollout(model, val_ds, ec.get("rollout_starts_select", ec.get("rollout_starts", 4)),
                                         ec["rollout_steps_select"], ec.get("sampling_steps", 10), device, t_ref,
                                         diag_sink=diags)
                metrics.update({"sel_" + key: value for key, value in sel.items()})
        for key, value in metrics.items():
            if key != "step":
                writer.add_scalar(f"val/{key}", value, step)
        score = legacy_score(metrics)
        metrics["score"] = score
        if score < best:
            best = score
            save(os.path.join(out_dir, "best.pt"))
        stable = selection_score(metrics)
        metrics["stability_score"] = list(stable)
        if best_stable is None or stable < best_stable:
            best_stable = stable
            save(os.path.join(out_dir, "best_stable.pt"))
        crashed_diags = [d for d in diags if any(c is not None for c in d["crashes"])]
        if crashed_diags:
            with open(os.path.join(out_dir, "crash_diagnostics.jsonl"), "a") as f:
                for d in crashed_diags:
                    f.write(json.dumps({"step": step, **d}) + "\n")
        # keep every evaluated checkpoint: short validation rollouts do not predict 10^4-step
        # stability well (argon v0), so checkpoints are re-ranked later with long validation rollouts
        os.makedirs(os.path.join(out_dir, "snapshots"), exist_ok=True)
        torch.save({"model": model.state_dict(), "step": step, "cfg": cfg, "elements": train_ds.elements},
                   os.path.join(out_dir, "snapshots", f"step{step:06d}.pt"))
        log.write(json.dumps(metrics) + "\n")
        log.flush()
        print(json.dumps(metrics), flush=True)
        save(ckpt_path)
        last_ckpt = time.time()
        model.train()

    if step == 0 and tc.get("init_from") and ec.get("eval_at_start", True):
        # fine-tune: score the initial weights so best_stable.pt can never be worse than the starting point
        evaluate_and_select()
    model.train()
    while step < tc["steps"]:
        for pos, vel, box in loader:
            if step >= tc["steps"]:
                break
            pos, vel = pos.to(device), vel.to(device)
            box = box.to(device) if train_ds.periodic else None
            m = unroll_length(tc["unroll"], step)
            lr_now = lr_at(step) * lr_scale_after_switch(step, last_switch_step(tc["unroll"], step), rewarmup)
            for g in opt.param_groups:
                g["lr"] = lr_now
            loss = training_loss(model, pos, vel, box, types, masses, m, noise_sigma_at(tc, step),
                                 tc.get("backprop", "truncated"), tc.get("unroll_sampling_steps", 2), **loss_opts)
            opt.zero_grad(set_to_none=True)
            loss_value = loss.item()
            if guard.check(loss_value, m):
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
                if torch.isfinite(gnorm):
                    opt.step()
                    gnorms.append(gnorm.item())
                else:
                    guard.skipped += 1
                    opt.zero_grad(set_to_none=True)
            step += 1
            if math.isfinite(loss_value):
                running.append(loss_value)
            if step % 50 == 0:
                mean_loss = float(np.mean(running)) if running else float("nan")
                mean_gn = float(np.mean(gnorms)) if gnorms else float("nan")
                writer.add_scalar("train/loss", mean_loss, step)
                writer.add_scalar("train/grad_norm", mean_gn, step)
                writer.add_scalar("train/unroll_m", m, step)
                print(json.dumps({"step": step, "loss": mean_loss, "m": m, "lr": lr_now,
                                  "grad_norm": mean_gn, "skipped_updates": guard.skipped,
                                  "elapsed_h": (time.time() - t_start) / 3600}), flush=True)
                running, gnorms = [], []
            if step % ec["every"] == 0 or step == tc["steps"]:
                evaluate_and_select()
            elif time.time() - last_ckpt > 60 * tc.get("ckpt_minutes", 20):
                save(ckpt_path)
                last_ckpt = time.time()
    save(ckpt_path)
    with open(os.path.join(out_dir, "done.json"), "w") as f:
        json.dump({"step": step, "best_score": best, "hours": (time.time() - t_start) / 3600}, f)


if __name__ == "__main__":
    main()
