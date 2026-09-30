"""Training loop: teacher forcing (S1), input noise (S2), unrolled training (S3a truncated, S3b full).

    python -m equitraj.train --config configs/argon/flow_s1_n10.yaml

Resumable: a checkpoint is written every `ckpt_minutes` and at every evaluation; rerunning the
same command continues from it. Model selection uses short validation rollouts, not the loss.
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
from torch.utils.tensorboard import SummaryWriter

from equitraj import analysis
from equitraj.dataset import TrajWindows, compute_stats
from equitraj.model import EquiTraj
from equitraj.rollout import rollout, temperature


def unroll_length(schedule, step):
    """schedule: [[step, m], ...] sorted by step; returns the m in effect at this training step."""
    m = schedule[0][1]
    for s, mm in schedule:
        if step >= s:
            m = mm
    return m


def training_loss(model, pos, vel, box, types, masses, m, noise_sigma, backprop, unroll_sampling_steps):
    """Loss for one batch. pos, vel: (B, k + m_max, N, 3); only the first k + m frames are used.

    m = 1 is teacher forcing. For m > 1 the model first predicts m-1 frames from its own outputs:
      truncated: those predictions carry no gradient; the loss is on the last step only
      full: gradients flow through all predictions; the loss is averaged over all m steps
    Input noise (S2) is added to the context frames; the Δx target is taken relative to the
    (noisy or predicted) last context frame, so the model learns to correct its input.
    """
    k = model.k
    b = pos.shape[0]
    ctx_x, ctx_v = pos[:, :k].clone(), vel[:, :k].clone()
    if noise_sigma > 0:
        tb = types.repeat(b).view(b, 1, -1, 1)
        ctx_x = ctx_x + torch.randn_like(ctx_x) * noise_sigma * model.dx_std[tb]
        ctx_v = ctx_v + torch.randn_like(ctx_v) * noise_sigma * model.v_std[tb]
    losses = []
    for j in range(m):
        last = (j == m - 1)
        if backprop == "full" or last:
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
    return torch.stack(losses).mean()


def validation_rollout(model, ds, n_starts, n_steps, sampling_steps, device, t_ref):
    """Short rollouts from evenly spaced validation windows. Returns stability and RDF metrics."""
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
    ok = torch.nonzero(~crashed).flatten().tolist()
    if not ok or out["positions"] is None:
        return metrics
    metrics["val_T_end"] = temperature(out["velocities"][-1, ok], ds.masses).mean().item()
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


def selection_score(m):
    """Lower is better: crashes dominate, then RDF error."""
    return m["val_crash_frac"] * 100.0 + m.get("val_rdf_l1", 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
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

    model = EquiTraj(n_types=len(train_ds.elements), k=cfg["k"], **cfg["model"]).to(device)
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
    step, best = 0, math.inf
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, best = ck["step"], ck["best"]
        print(f"resumed from step {step}", flush=True)
    else:
        dx_std, v_std = compute_stats(train_ds)
        model.dx_std.copy_(dx_std)
        model.v_std.copy_(v_std)
    print(json.dumps({"elements": train_ds.elements, "dx_std": model.dx_std.tolist(), "v_std": model.v_std.tolist(),
                      "n_params": sum(p.numel() for p in model.parameters()), "n_train_windows": len(train_ds)}), flush=True)

    def lr_at(s):
        warm = tc.get("warmup", 1000)
        if s < warm:
            return tc["lr"] * (s + 1) / warm
        return tc["lr"] * 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, tc["steps"] - warm))))

    def save(path):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "best": best, "cfg": cfg,
                    "elements": train_ds.elements}, path + ".tmp")
        os.replace(path + ".tmp", path)

    writer = SummaryWriter(os.path.join(out_dir, "tb"))
    log = open(os.path.join(out_dir, "log.jsonl"), "a")
    t_ref = temperature(torch.stack([val_ds[i][1][0] for i in range(0, len(val_ds), max(1, len(val_ds) // 20))]), val_ds.masses).mean().item()
    last_ckpt, t_start, running = time.time(), time.time(), []
    model.train()
    while step < tc["steps"]:
        for pos, vel, box in loader:
            if step >= tc["steps"]:
                break
            pos, vel = pos.to(device), vel.to(device)
            box = box.to(device) if train_ds.periodic else None
            m = unroll_length(tc["unroll"], step)
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            loss = training_loss(model, pos, vel, box, types, masses, m, tc.get("noise_sigma", 0.0),
                                 tc.get("backprop", "truncated"), tc.get("unroll_sampling_steps", 2))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.get("grad_clip", 1.0))
            opt.step()
            step += 1
            running.append(loss.item())
            if step % 50 == 0:
                writer.add_scalar("train/loss", np.mean(running), step)
                writer.add_scalar("train/grad_norm", gnorm.item(), step)
                writer.add_scalar("train/unroll_m", m, step)
                print(json.dumps({"step": step, "loss": float(np.mean(running)), "m": m, "lr": lr_at(step),
                                  "elapsed_h": (time.time() - t_start) / 3600}), flush=True)
                running = []
            if step % ec["every"] == 0 or step == tc["steps"]:
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
                    metrics.update(validation_rollout(model, val_ds, ec.get("rollout_starts", 4), ec.get("rollout_steps", 500),
                                                      ec.get("sampling_steps", 10), device, t_ref))
                for key, value in metrics.items():
                    if key != "step":
                        writer.add_scalar(f"val/{key}", value, step)
                score = selection_score(metrics)
                metrics["score"] = score
                if score < best:
                    best = score
                    save(os.path.join(out_dir, "best.pt"))
                log.write(json.dumps(metrics) + "\n")
                log.flush()
                print(json.dumps(metrics), flush=True)
                save(ckpt_path)
                last_ckpt = time.time()
                model.train()
            elif time.time() - last_ckpt > 60 * tc.get("ckpt_minutes", 20):
                save(ckpt_path)
                last_ckpt = time.time()
    save(ckpt_path)
    with open(os.path.join(out_dir, "done.json"), "w") as f:
        json.dump({"step": step, "best_score": best, "hours": (time.time() - t_start) / 3600}, f)


if __name__ == "__main__":
    main()
