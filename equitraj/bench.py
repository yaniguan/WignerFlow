"""Helpers for the CPU benchmarks in scripts/bench_*.py (and their tests in tests/test_bench.py).

    random_orthogonal      Haar-random rotation, or an improper rotation (rotation times inversion)
    make_liquid            synthetic argon-like input: jittered cubic lattice at the argon density, k frames
    make_model             an (untrained) EquiTraj with the B_S3a architecture by default
    equivariance_error     max/mean error of the outputs under a rotation/reflection plus translation
    bench_config           forward/backward/predict wall time and peak RSS for one (atoms, frames) setting
    bench_config_isolated  the same in a fresh subprocess, so peak memory is not polluted by earlier runs
    make_cutoff_probe, place_probe, scan_outputs, jump_stats
                           continuity check of the outputs when one edge crosses the cutoff

The model has no energy or forces (README, "Limits"), so there is no force-consistency or NVE check;
the continuity check is the replacement. EquiTraj is imported lazily so that the pure helpers work
without e3nn.
"""

import math
import multiprocessing
import os
import queue
import resource
import sys
import time

import torch

from equitraj.neighbors import brute_force

ARGON_DENSITY = 256 / 23.12**3  # atoms per Å^3: the argon ground-truth box (256 atoms, L = 2.312 nm)
ARGON_MASS = 39.948  # amu
ARGON_V_SIGMA = 0.0014  # Å/fs, sqrt(kT/m) at 94.4 K

# architecture of configs/argon/B_S3a.yaml (the v0 run)
B_S3A = dict(lmax=2, channels=64, heads=8, n_rbf=32, cutoff=6.0, n_spatial=2, n_fusion=2, n_denoiser=2,
             parity=True, use_velocity=True, target="residual")


# ---------------------------------------------------------------- inputs and model

def random_orthogonal(reflect=False, generator=None, dtype=torch.float64):
    """Haar-random 3x3 rotation (det +1); with reflect=True, minus that rotation (det -1)."""
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=generator, dtype=torch.float64))
    q = q * torch.sign(torch.diagonal(r))[None, :]  # makes the distribution uniform
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    if reflect:
        q = -q
    return q.to(dtype)


def make_liquid(n_atoms, k=3, batch=1, periodic=True, density=ARGON_DENSITY, stride_fs=50.0,
                v_sigma=ARGON_V_SIGMA, jitter=0.2, seed=0, dtype=torch.float32):
    """Synthetic argon-like context: k frames of B systems on a jittered cubic lattice.

    Frame f+1 = frame f + v_{f+1} * stride, so the displacement inputs match the velocities.
    Returns a dict with pos, vel (B, k, N, 3), types (N,), masses (N,), box (B, 3, 3) or None.
    """
    g = torch.Generator().manual_seed(seed)
    length = (n_atoms / density) ** (1 / 3)
    side = max(1, round(n_atoms ** (1 / 3)))
    while side**3 < n_atoms:
        side += 1
    grid = torch.stack(torch.meshgrid(*[torch.arange(side, dtype=torch.float64)] * 3, indexing="ij"), -1)
    grid = grid.reshape(-1, 3)[:n_atoms]
    pos0 = (grid + 0.5) * (length / side) + jitter * torch.randn(batch, n_atoms, 3, generator=g, dtype=torch.float64)
    vel = v_sigma * torch.randn(batch, k, n_atoms, 3, generator=g, dtype=torch.float64)
    pos = pos0[:, None] + torch.cumsum(vel, dim=1) * stride_fs
    box = (torch.eye(3, dtype=torch.float64) * length).expand(batch, 3, 3).contiguous() if periodic else None
    return {"pos": pos.to(dtype), "vel": vel.to(dtype), "types": torch.zeros(n_atoms, dtype=torch.long),
            "masses": torch.full((n_atoms,), ARGON_MASS, dtype=dtype),
            "box": None if box is None else box.to(dtype)}


def make_model(head="flow", dtype=torch.float32, seed=0, k=3, stride_fs=50.0, **overrides):
    """Untrained EquiTraj for one atom type, B_S3a architecture unless overridden, in eval mode."""
    from equitraj.model import EquiTraj  # noqa: PLC0415  (needs e3nn)

    cfg = dict(B_S3A, head=head)
    cfg.update(overrides)
    torch.manual_seed(seed)
    return EquiTraj(n_types=1, k=k, stride_fs=stride_fs, **cfg).to(dtype).eval()


def run_model(model, inp, noise=None, n_steps=2):
    """One prediction step (no grad). Returns (dx, v_new), each (B, N, 3)."""
    return model.predict(inp["pos"], inp["vel"], inp["types"], inp["box"], inp["masses"], n_steps=n_steps, noise=noise)


def count_edges(inp, cutoff):
    """Total number of directed neighbor edges over all systems and frames."""
    pos, box = inp["pos"], inp["box"]
    return sum(int(brute_force(pos[b, f], None if box is None else box[b], cutoff)[0].numel())
               for b in range(pos.shape[0]) for f in range(pos.shape[1]))


# ---------------------------------------------------------------- equivariance

def transform_inputs(inp, rot, shift=None):
    """Apply x -> R x + shift to positions, R to velocities and to the box vectors (rows of box)."""
    out = dict(inp)
    out["pos"] = inp["pos"] @ rot.T + (0.0 if shift is None else shift)
    out["vel"] = inp["vel"] @ rot.T
    out["box"] = None if inp["box"] is None else inp["box"] @ rot.T
    return out


def equivariance_error(model, inp, rot, shift=None, n_steps=2, generator=None):
    """Compare f(R x + t) with R f(x) for both outputs (dx, v_new).

    The flow head's starting noise is an input vector field, so it is rotated too.
    Errors are per-atom vector norms |f(Rx) - R f(x)|; "rel" divides by the largest per-atom norm of R f(x).
    Returns {"dx": {...}, "v": {...}} with max_abs, mean_abs, max_rel, mean_rel.
    """
    dtype = inp["pos"].dtype
    rot = rot.to(dtype)
    shift = None if shift is None else shift.to(dtype)
    noise = None
    if model.head_type == "flow":
        noise = model.com_free_noise(inp["pos"].shape[0], inp["types"], inp["masses"], generator=generator, like=inp["pos"])
    dx, v = run_model(model, inp, noise, n_steps)
    dx_r, v_r = run_model(model, transform_inputs(inp, rot, shift), None if noise is None else noise @ rot.T, n_steps)
    stats = {}
    for name, ref, got in [("dx", dx @ rot.T, dx_r), ("v", v @ rot.T, v_r)]:
        err = (got - ref).norm(dim=-1)
        scale = ref.norm(dim=-1).max()
        stats[name] = {"max_abs": err.max().item(), "mean_abs": err.mean().item(),
                       "max_rel": (err.max() / scale).item(), "mean_rel": (err.mean() / scale).item()}
    return stats


def summarize_trials(trials):
    """List of equivariance_error results -> worst max_* and average mean_* over trials, per output."""
    out = {}
    for name in trials[0]:
        out[name] = {
            "max_abs": max(t[name]["max_abs"] for t in trials),
            "mean_abs": sum(t[name]["mean_abs"] for t in trials) / len(trials),
            "max_rel": max(t[name]["max_rel"] for t in trials),
            "mean_rel": sum(t[name]["mean_rel"] for t in trials) / len(trials),
        }
    return out


# ---------------------------------------------------------------- speed and memory

def current_rss_mb():
    """Resident memory of this process now, in MiB (Linux /proc only; None elsewhere)."""
    try:
        with open("/proc/self/statm") as f:
            pages = int(f.read().split()[1])
    except OSError:
        return None
    return pages * os.sysconf("SC_PAGE_SIZE") / 2**20


def peak_rss_mb():
    """High-water resident memory of this process since it started, in MiB."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return peak / 2**20 if sys.platform == "darwin" else peak / 1024  # bytes on macOS, KiB on Linux


def _median(xs):
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def bench_config(n_atoms, k=3, batch=1, head="flow", repeats=3, warmup=1, threads=None, sampling_steps=10,
                 seed=0, dtype="float32", **model_kwargs):
    """Time the training forward (model.loss), its backward, and one no-grad prediction step.

    Peak memory is the process high-water RSS minus the RSS after building the model and inputs, so it
    is only meaningful in a fresh process (see bench_config_isolated): the high-water mark never resets.
    Times are medians over `repeats` after `warmup` untimed iterations.
    """
    if threads is not None:
        torch.set_num_threads(threads)
    dt = getattr(torch, dtype)
    model = make_model(head, dt, seed=seed, k=k, **model_kwargs)
    inp = make_liquid(n_atoms, k=k, batch=batch, seed=seed, dtype=dt)
    g = torch.Generator().manual_seed(seed)
    dx_t = 0.07 * torch.randn(inp["vel"][:, 0].shape, generator=g, dtype=dt)
    v_t = ARGON_V_SIGMA * torch.randn(inp["vel"][:, 0].shape, generator=g, dtype=dt)
    edges_per_frame = count_edges(inp, model.cutoff) / (batch * k)
    torch.manual_seed(seed)

    rss0 = current_rss_mb()
    fwd, bwd, pred = [], [], []
    model.train()
    for i in range(warmup + repeats):
        model.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        loss = model.loss(inp["pos"], inp["vel"], inp["types"], inp["box"], inp["masses"], dx_t, v_t)
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        if i >= warmup:
            fwd.append(t1 - t0)
            bwd.append(t2 - t1)
    model.eval()
    for i in range(warmup + repeats):
        t0 = time.perf_counter()
        dx, _ = run_model(model, inp, None, sampling_steps)
        if i >= warmup:
            pred.append(time.perf_counter() - t0)
    peak = peak_rss_mb()
    return {
        "n_atoms": n_atoms, "frames": k, "batch": batch, "head": head, "dtype": dtype,
        "threads": torch.get_num_threads(), "params": sum(p.numel() for p in model.parameters()),
        "edges_per_frame": edges_per_frame, "loss": loss.item(), "finite": bool(torch.isfinite(dx).all()),
        "forward_s": _median(fwd), "backward_s": _median(bwd), "fwd_bwd_s": _median([a + b for a, b in zip(fwd, bwd)]),
        "predict_s": _median(pred), "sampling_steps": sampling_steps,
        "rss_before_mb": rss0, "peak_rss_mb": peak, "peak_delta_mb": None if rss0 is None else max(peak - rss0, 0.0),
    }


def _bench_child(q, kwargs):
    try:
        q.put(bench_config(**kwargs))
    except Exception as e:  # noqa: BLE001  (report any failure to the parent instead of hanging)
        q.put({"error": repr(e)})


def bench_config_isolated(timeout_s=900, **kwargs):
    """bench_config in a fresh 'spawn' subprocess. Returns {"error": ...} if it fails, is killed (e.g. out of memory) or times out."""
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=_bench_child, args=(q, kwargs))
    p.start()
    deadline = time.monotonic() + timeout_s
    res = None
    while res is None:
        try:
            res = q.get(timeout=1.0)
        except queue.Empty:
            if not p.is_alive():
                try:
                    res = q.get(timeout=1.0)
                except queue.Empty:
                    res = {"error": f"worker exited with code {p.exitcode} (killed, e.g. out of memory?)"}
            elif time.monotonic() > deadline:
                p.terminate()
                res = {"error": f"timeout after {timeout_s} s"}
    p.join(10)
    if p.is_alive():
        p.kill()
    return res


# ---------------------------------------------------------------- continuity across the cutoff

def make_cutoff_probe(n_cluster=4, k=3, stride_fs=50.0, seed=0, dtype=torch.float64):
    """Vacuum system for the cutoff scan: atom 0 at the origin, n_cluster - 1 atoms near x = -3 Å, and a probe atom (last).

    Atom 0 and the probe do not move (zero velocity in every frame), so their distance is the same in
    all k frames; the other cluster atoms move a little. The probe is put on the +x axis by place_probe.
    Every cluster atom except atom 0 is more than r + 2.5 Å from the probe, so for r near a cutoff of
    a few Å only the edge pair (0, probe) can appear or disappear.
    """
    g = torch.Generator().manual_seed(seed)
    n = n_cluster + 1
    pos0 = torch.zeros(n, 3, dtype=torch.float64)
    pos0[1:n_cluster, 0] = -3.0 + 0.3 * torch.rand(n_cluster - 1, generator=g, dtype=torch.float64)
    pos0[1:n_cluster, 1:] = 1.5 * (2 * torch.rand(n_cluster - 1, 2, generator=g, dtype=torch.float64) - 1)
    vel = ARGON_V_SIGMA * torch.randn(1, k, n, 3, generator=g, dtype=torch.float64)
    vel[:, :, 0] = 0.0
    vel[:, :, -1] = 0.0
    pos = pos0[None, None] + torch.cumsum(vel, dim=1) * stride_fs
    return {"pos": pos.to(dtype), "vel": vel.to(dtype), "types": torch.zeros(n, dtype=torch.long),
            "masses": torch.full((n,), ARGON_MASS, dtype=dtype), "box": None}


def place_probe(base, r):
    """Copy of `base` with the probe (last atom) at (r, 0, 0) in every frame."""
    out = dict(base)
    out["pos"] = base["pos"].clone()
    out["pos"][:, :, -1, :] = torch.tensor([r, 0.0, 0.0], dtype=base["pos"].dtype)
    return out


def scan_outputs(model, base, radii, n_steps=2, noise=None):
    """Outputs (concatenated dx, v_new of all atoms) and the edge count for each probe distance r.

    For the flow head pass a fixed `noise`, otherwise every point gets fresh noise.
    Returns (outputs (S, D), n_edges (S,)).
    """
    outs, n_edges = [], []
    for r in radii:
        inp = place_probe(base, float(r))
        dx, v = run_model(model, inp, noise, n_steps)
        outs.append(torch.cat([dx.flatten(), v.flatten()]))
        n_edges.append(count_edges(inp, model.cutoff))
    return torch.stack(outs), torch.tensor(n_edges)


def jump_stats(outputs, n_edges):
    """Continuity check on outputs sampled along a uniform grid.

    outputs: (S, D), n_edges: (S,). A step i -> i+1 is a "crossing" step if the neighbor graph
    changes there. For a continuous output, the crossing step is about as large as the other steps
    and shrinks with the grid spacing; a jump shows up as crossing_to_max_smooth >> 1 that does not
    go away when the spacing is reduced.
    """
    steps = (outputs[1:] - outputs[:-1]).norm(dim=-1)
    crossing = n_edges[1:] != n_edges[:-1]
    smooth = steps[~crossing]
    res = {"n_steps": int(steps.numel()), "n_crossing_steps": int(crossing.sum()),
           "output_norm_max": outputs.norm(dim=-1).max().item(),
           "median_smooth_step": smooth.median().item() if smooth.numel() else None,
           "max_smooth_step": smooth.max().item() if smooth.numel() else None}
    if crossing.any():
        jump = steps[crossing].max()
        res["max_crossing_step"] = jump.item()
        res["crossing_rel"] = (jump / outputs.norm(dim=-1).max()).item()
        if smooth.numel():
            res["crossing_to_max_smooth"] = (jump / smooth.max()).item() if smooth.max() > 0 else math.inf
    return res
