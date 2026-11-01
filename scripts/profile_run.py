"""Profile one training configuration: training-step breakdown, rollout speed, profiler trace, GPU load.

    python scripts/profile_run.py --config configs/argon/B_S3a.yaml --ckpt results/argon/B_S3a/best.pt
    python scripts/profile_run.py --config configs/argon/smoke.yaml --random --device cpu
    python scripts/bench_openmm.py --system argon --model-random > omm.json
    python scripts/profile_run.py --config configs/argon/B_S3a.yaml --random --openmm-json omm.json

Writes to --out (default <config out_dir>/profiling):
    summary.json       training phases (data / neighbor_list / forward / backward / optimizer), peak memory,
                       rollout steps/s and ns/day, neighbor-list time, GPU utilization, OpenMM comparison
    trace.json         torch.profiler chrome trace (open in https://ui.perfetto.dev)
    key_averages.txt   torch.profiler operator table
    gpu_samples.csv    nvidia-smi samples (only if nvidia-smi exists)
Training steps run on a copy of the model, so the rollout uses the loaded weights. Random models
(--random) get data-derived scales and only measure compute cost.
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import yaml

from equitraj.dataset import TrajWindows, compute_stats
from equitraj.model import EquiTraj
from equitraj.prior import estimate_rmin
from equitraj.profiling import GpuSampler, profile_training, rollout_speed, torch_profile, write_summary
from equitraj.train import noise_sigma_at, training_loss, unroll_length


def load_model(args, cfg):
    """Returns (model, k, model_cfg, source). The dataset config (--config) supplies data_dir and stride."""
    if args.random:
        return None, cfg["k"], dict(cfg["model"]), "random"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if "cfg" in ck:
        mc, k = dict(ck["cfg"]["model"]), ck["cfg"]["k"]
    else:  # released checkpoint (scripts/export_ckpt.py)
        mc, k = dict(ck["model_cfg"]), ck["k"]
    return ck, k, mc, args.ckpt


def loss_options(tc, ds, model, device):
    """The optional training_loss arguments, built as in equitraj.train.main."""
    opts = {"unroll_weights": tc.get("unroll_loss_weights"), "noise_v_ratio": float(tc.get("noise_v_ratio", 1.0))}
    nts = tc.get("noise_type_scale")
    if nts is not None:
        if isinstance(nts, dict):
            nts = [float(nts.get(e, 1.0)) for e in ds.elements]
        opts["noise_type_scale"] = torch.tensor(nts, dtype=torch.float32, device=device)
    op = tc.get("overlap_penalty") or {}
    if op.get("weight", 0.0) > 0:
        rmin = model.prior.rmin if model.prior is not None else estimate_rmin(ds, op.get("percentile", 0.1)).to(device)
        opts["overlap_weight"] = float(op["weight"])
        opts["overlap_r0"] = rmin * float(op.get("rmin_scale", 1.0))
    return opts


def openmm_comparison(path, model_ns_per_day):
    """Reads scripts/bench_openmm.py output ({"openmm": {...}, "model": {...}}) or a bare rates dict."""
    with open(path) as f:
        bench = json.load(f)
    omm = bench.get("openmm", bench)
    if "ns_per_day" not in omm:
        raise ValueError(f"{path}: no OpenMM ns_per_day")
    out = {"source": path, "platform": omm.get("platform"), "ns_per_day": omm["ns_per_day"],
           "steps_per_s": omm.get("steps_per_s"), "step_fs": omm.get("step_fs"), "atoms": bench.get("atoms"),
           "speed_ratio": model_ns_per_day / omm["ns_per_day"] if omm["ns_per_day"] > 0 else None}
    if "model" in bench and "ns_per_day" in bench["model"]:
        out["bench_model_ns_per_day"] = bench["model"]["ns_per_day"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ckpt", help="training (ckpt.pt / best.pt / snapshot) or released checkpoint")
    src.add_argument("--random", action="store_true", help="random weights: compute cost only")
    ap.add_argument("--out", help="output directory (default: <config out_dir>/profiling)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", default="train")
    ap.add_argument("--train-steps", type=int, default=20)
    ap.add_argument("--train-warmup", type=int, default=3)
    ap.add_argument("--batch-size", type=int, help="default: train.batch_size")
    ap.add_argument("--m", type=int, help="unroll length (default: the last train.unroll entry)")
    ap.add_argument("--workers", type=int, help="DataLoader workers (default: train.workers)")
    ap.add_argument("--rollout-steps", type=int, default=50)
    ap.add_argument("--rollout-warmup", type=int, default=3)
    ap.add_argument("--rollout-batch", type=int, default=1, help="1 is comparable with bench_openmm.py")
    ap.add_argument("--sampling-steps", type=int, help="flow steps per rollout step (default: eval.sampling_steps)")
    ap.add_argument("--profile", choices=["train", "rollout", "none"], default="train",
                    help="what torch.profiler traces")
    ap.add_argument("--profile-steps", type=int, default=3)
    ap.add_argument("--gpu-interval", type=float, default=1.0, help="nvidia-smi sampling period (s)")
    ap.add_argument("--openmm-json", help="output of scripts/bench_openmm.py")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    tc, ec = cfg["train"], cfg.get("eval", {})
    out_dir = args.out or os.path.join(cfg.get("out_dir", "results"), "profiling")
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch.manual_seed(args.seed)

    sampler = GpuSampler(os.path.join(out_dir, "gpu_samples.csv"), args.gpu_interval).start()
    t_start = time.time()
    try:
        ck, k, mc, source = load_model(args, cfg)
        m = args.m if args.m is not None else tc["unroll"][-1][1]
        ds = TrajWindows(cfg["data_dir"], args.split, cfg["stride_steps"], k, m=m, start_every=tc.get("start_every", 1))
        model = EquiTraj(n_types=len(ds.elements), k=k, stride_fs=ds.stride_fs, **mc)
        if ck is not None:
            if list(ck["elements"]) != list(ds.elements):
                raise ValueError(f"checkpoint elements {list(ck['elements'])} != dataset elements {ds.elements}")
            if "stride_fs" in ck and not np.isclose(ck["stride_fs"], ds.stride_fs):
                raise ValueError("dataset stride does not match the checkpoint")
            model.load_state_dict(ck["model"])
        else:
            st = compute_stats(ds, n_samples=200)
            model.dx_std.copy_(st["dx_std"])
            model.v_std.copy_(st["v_std"])
            residual = model.target == "residual"
            model.out_dx_std.copy_(st["res_dx_std"] if residual else st["dx_std"])
            model.out_v_std.copy_(st["res_v_std"] if residual else st["v_std"])
            if model.prior is not None and not model.prior.rmin_fixed:
                model.prior.rmin.copy_(estimate_rmin(ds, model.prior.percentile))
        model = model.to(device)
        types, masses = ds.types.to(device), ds.masses.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        write_summary(out_dir, {"run": {"config": args.config, "model_source": source, "device": str(device),
                                        "torch": torch.__version__, "n_params": n_params, "atoms": len(ds.types),
                                        "elements": ds.elements, "stride_fs": ds.stride_fs, "k": k,
                                        "model_cfg": mc}})

        # ---- rollout speed (loaded weights)
        sampling_steps = args.sampling_steps or ec.get("sampling_steps", 10)
        starts = np.linspace(0, len(ds) - 1, args.rollout_batch).astype(int)
        items = [ds[i] for i in starts]
        pos = torch.stack([it[0][:k] for it in items])
        vel = torch.stack([it[1][:k] for it in items])
        box = torch.stack([it[2] for it in items]) if ds.periodic else None
        roll = rollout_speed(model, pos, vel, ds.types, box, ds.masses, args.rollout_steps, ds.stride_fs,
                             sampling_steps, warmup=args.rollout_warmup)
        print(json.dumps({"rollout": roll}), flush=True)
        result = {"rollout": roll}
        if args.openmm_json:
            result["openmm"] = openmm_comparison(args.openmm_json, roll["ns_per_day"])
            print(json.dumps({"openmm": result["openmm"]}), flush=True)
        write_summary(out_dir, result)

        # ---- training-step breakdown (on a copy)
        bs = args.batch_size or tc["batch_size"]
        workers = tc.get("workers", 4) if args.workers is None else args.workers
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers,
                                             persistent_workers=workers > 0)
        train_model = copy.deepcopy(model)
        opt = torch.optim.AdamW(train_model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.0))
        loss_opts = loss_options(tc, ds, train_model, device)
        noise = noise_sigma_at(tc, tc["steps"])
        backprop, unroll_ss = tc.get("backprop", "truncated"), tc.get("unroll_sampling_steps", 2)
        train = profile_training(train_model, loader, types, masses, opt=opt, n_steps=args.train_steps,
                                 warmup=args.train_warmup, m=m, noise_sigma=noise, backprop=backprop,
                                 unroll_sampling_steps=unroll_ss, grad_clip=tc.get("grad_clip", 1.0),
                                 loss_opts=loss_opts, periodic=ds.periodic, device=device)
        train["schedule_m_final"] = unroll_length(tc["unroll"], tc["steps"])
        train["noise_sigma"] = noise
        train["workers"] = workers
        print(json.dumps({"train": train}), flush=True)
        write_summary(out_dir, {"train": train})

        # ---- torch.profiler trace
        if args.profile != "none":
            if args.profile == "train":
                batch = next(iter(loader))
                bp, bv = batch[0].to(device), batch[1].to(device)
                bb = batch[2].to(device) if ds.periodic else None

                def fn():
                    loss = training_loss(train_model, bp, bv, bb, types, masses, m, noise, backprop, unroll_ss,
                                         **loss_opts)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), tc.get("grad_clip", 1.0))
                    opt.step()
            else:
                model.eval()
                cx = pos.to(device)
                cv = vel.to(device)
                cb = None if box is None else box.to(device)

                def fn():
                    with torch.no_grad():
                        model.predict(cx, cv, types, cb, masses, n_steps=sampling_steps)
            prof = torch_profile(fn, args.profile_steps, out_dir)
            prof["target"] = args.profile
            write_summary(out_dir, {"profiler": prof})
    finally:
        sampler.stop()
    path = write_summary(out_dir, {"gpu": sampler.summary(), "run": {"wall_s": time.time() - t_start}})
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
