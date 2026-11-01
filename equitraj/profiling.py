"""Profiling helpers: where a training step spends its time, rollout speed, GPU load, profiler traces.

    timer = StepTimer(device)
    with timer.step():
        with timer.phase("data"): ...
        with timer.phase("forward"): ...
    timer.summary()

Phases nest: time spent in an inner phase is subtracted from the outer one, so the per-phase
times add up to the step time. `time_neighbor_list` uses this to separate graph construction
(equitraj.model.build_graph, called inside encode) from the rest of the forward pass.
On CUDA every phase boundary calls torch.cuda.synchronize, so the numbers are wall-clock GPU
times; the synchronizations themselves slow training down a little.

scripts/profile_run.py combines everything into <out>/summary.json.
"""

import contextlib
import json
import math
import os
import shutil
import subprocess
import threading
import time

import numpy as np
import torch

from equitraj import model as model_module
from equitraj.train import training_loss

PHASES = ("data", "neighbor_list", "forward", "backward", "optimizer")


def _sync(device):
    if device is not None and torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def rates(steps, wall_s, stride_fs):
    """Throughput of `steps` model steps of stride_fs each in wall_s seconds (same keys as scripts/bench_openmm.py)."""
    sps = steps / wall_s if wall_s > 0 else math.inf
    return {"steps_per_s": sps, "ns_per_day": sps * stride_fs * 86400 / 1e6, "steps_per_ns": 1e6 / stride_fs,
            "wall_s": wall_s, "step_fs": stride_fs}


# ---------------------------------------------------------------- step timer

class StepTimer:
    """Accumulates exclusive wall time per named phase over training steps.

    device: the device the work runs on; on CUDA each phase boundary synchronizes it.
    None: synchronize the current CUDA device if CUDA is initialized, else nothing.
    """

    def __init__(self, device=None, phases=PHASES):
        if device is None and torch.cuda.is_available() and torch.cuda.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = None if device is None else torch.device(device)
        self.phases = tuple(phases)
        self.reset()

    def reset(self):
        self.totals = {p: 0.0 for p in self.phases}
        self.counts = {p: 0 for p in self.phases}
        self.step_times = []
        self._stack = []  # [name, time spent in nested phases]

    @contextlib.contextmanager
    def phase(self, name):
        _sync(self.device)
        t0 = time.perf_counter()
        self._stack.append([name, 0.0])
        try:
            yield
        finally:
            _sync(self.device)
            dt = time.perf_counter() - t0
            _, inner = self._stack.pop()
            self.totals[name] = self.totals.get(name, 0.0) + dt - inner
            self.counts[name] = self.counts.get(name, 0) + 1
            if self._stack:
                self._stack[-1][1] += dt

    __call__ = phase

    @contextlib.contextmanager
    def step(self):
        """Marks one training step; phases inside it are counted towards that step."""
        _sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            _sync(self.device)
            self.step_times.append(time.perf_counter() - t0)

    def summary(self):
        n = len(self.step_times)
        total = sum(self.totals.values())
        per = max(n, 1)
        phases = {p: {"total_s": t, "mean_ms": 1e3 * t / per, "frac": t / total if total > 0 else 0.0,
                      "calls": self.counts.get(p, 0)}
                  for p, t in self.totals.items()}
        step_s = float(np.mean(self.step_times)) if n else total / per
        return {"n_steps": n, "step_ms_mean": 1e3 * step_s,
                "step_ms_median": 1e3 * float(np.median(self.step_times)) if n else 1e3 * step_s,
                "steps_per_s": 1.0 / step_s if step_s > 0 else None, "phases": phases}


@contextlib.contextmanager
def time_neighbor_list(timer, phase="neighbor_list"):
    """Time every equitraj.model.build_graph call (neighbor list + edge Wigner matrices) as `phase`.

    Temporarily replaces the module attribute that EquiTraj.encode looks up; restored on exit.
    """
    original = model_module.build_graph

    def timed(*args, **kwargs):
        with timer.phase(phase):
            return original(*args, **kwargs)

    model_module.build_graph = timed
    try:
        yield
    finally:
        model_module.build_graph = original


# ---------------------------------------------------------------- training-step breakdown

def _batch_stream(batches):
    """Endless iterator over an iterable of batches (a DataLoader is re-iterated, i.e. reshuffled)."""
    while True:
        n = 0
        for b in batches:
            n += 1
            yield b
        if n == 0:
            raise ValueError("no batches to profile")


def _peak_rss_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux
    except (ImportError, AttributeError):
        return None


def profile_training(model, batches, types, masses, opt=None, n_steps=10, warmup=2, m=1, noise_sigma=0.0,
                     backprop="truncated", unroll_sampling_steps=2, grad_clip=1.0, loss_opts=None,
                     periodic=True, lr=1e-4, device=None):
    """Run warmup + n_steps training steps (as in equitraj.train) and time their phases.

    batches: iterable of (pos, vel, box) as yielded by a DataLoader over TrajWindows.
    opt: optimizer (default AdamW(lr)). loss_opts: extra keyword arguments of training_loss.
    Phases: data (next batch + transfer), neighbor_list (build_graph, also when recomputed in
    backward by gradient checkpointing), forward (loss without the graphs), backward (zero_grad +
    backward without the graphs), optimizer (grad clipping + step).
    Peak memory: torch.cuda.max_memory_allocated on CUDA, process max RSS on CPU.
    """
    if n_steps < 1 or warmup < 0:
        raise ValueError("n_steps must be positive and warmup nonnegative")
    device = torch.device(device) if device is not None else next(model.parameters()).device
    types, masses = types.to(device), masses.to(device)
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_opts = loss_opts or {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    timer = StepTimer(device)
    stream = _batch_stream(batches)
    losses = []
    was_training = model.training
    model.train()
    with time_neighbor_list(timer):
        for i in range(warmup + n_steps):
            if i == warmup:
                timer.reset()
                losses = []
            with timer.step():
                with timer.phase("data"):
                    pos, vel, box = next(stream)
                    pos, vel = pos.to(device, non_blocking=True), vel.to(device, non_blocking=True)
                    box = box.to(device) if periodic and box is not None else None
                with timer.phase("forward"):
                    loss = training_loss(model, pos, vel, box, types, masses, m, noise_sigma, backprop,
                                         unroll_sampling_steps, **loss_opts)
                with timer.phase("backward"):
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                with timer.phase("optimizer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    opt.step()
            losses.append(loss.item())
    model.train(was_training)
    out = timer.summary()
    out.update({"warmup": warmup, "batch_size": int(pos.shape[0]), "atoms": int(pos.shape[2]), "m": m,
                "backprop": backprop, "unroll_sampling_steps": unroll_sampling_steps, "device": str(device),
                "loss_mean": float(np.mean(losses)) if losses else None,
                "loss_finite": bool(np.all(np.isfinite(losses)))})
    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device)
        out.update({"peak_mem_bytes": int(peak), "peak_mem_GB": peak / 1e9})
    else:
        out.update({"peak_mem_bytes": None, "peak_mem_GB": None, "peak_rss_MB": _peak_rss_mb()})
    return out


# ---------------------------------------------------------------- rollout speed

@torch.no_grad()
def rollout_speed(model, pos, vel, types, box, masses, n_steps, stride_fs=None, sampling_steps=10,
                  warmup=2, nl_repeats=5):
    """Autoregressive rollout throughput, without crash checks or frame saving.

    pos, vel: (B, k, N, 3) context in Å and Å/fs. box: (B, 3, 3) or None. stride_fs defaults to
    model.stride_fs. Positions are accumulated in float64 as in equitraj.rollout. Warmup steps are
    not timed. ns_per_day is per trajectory (comparable with batch-one OpenMM numbers);
    ns_per_day_aggregate multiplies by the batch size.
    Neighbor-list time: build_graph on the last frame (mean of nl_repeats calls); each model step
    builds graphs for all k context frames plus the last frame again, so the per-step estimate is
    (k + 1) times that.
    """
    if n_steps < 1 or warmup < 0:
        raise ValueError("n_steps must be positive and warmup nonnegative")
    stride_fs = float(stride_fs if stride_fs is not None else getattr(model, "stride_fs", 1.0))
    p = next(model.parameters())
    device, dtype = p.device, p.dtype
    x = pos.to(device, torch.float64)
    v = vel.to(device, dtype)
    types, masses = types.to(device), masses.to(device, dtype)
    box = None if box is None else box.to(device, dtype)
    was_training = model.training
    model.eval()

    def step(x, v):
        dx, v_new = model.predict(x.to(dtype), v, types, box, masses, n_steps=sampling_steps)
        x_new = x[:, -1] + dx.double()
        return torch.cat([x[:, 1:], x_new[:, None]], dim=1), torch.cat([v[:, 1:], v_new[:, None]], dim=1)

    for _ in range(warmup):
        x, v = step(x, v)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x, v = step(x, v)
    _sync(device)
    wall = time.perf_counter() - t0
    finite = bool(torch.isfinite(x).all() and torch.isfinite(v).all())

    last = x[:, -1].to(dtype)

    def graph():
        return model_module.build_graph(last, box, model.cutoff, model.lmax, model.neighbor_method)

    g = graph()  # warmup
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(max(1, nl_repeats)):
        g = graph()
    _sync(device)
    nl_s = (time.perf_counter() - t0) / max(1, nl_repeats)
    model.train(was_training)

    b, k, n = pos.shape[0], pos.shape[1], pos.shape[2]
    out = rates(n_steps, wall, stride_fs)
    step_s = wall / n_steps if n_steps > 0 else math.nan
    out.update({"n_steps": n_steps, "warmup": warmup, "batch": b, "k": k, "atoms": n,
                "sampling_steps": sampling_steps, "device": str(device),
                "ns_per_day_aggregate": out["ns_per_day"] * b, "step_ms": 1e3 * step_s,
                "neighbor_list_ms": 1e3 * nl_s, "neighbor_list_edges": int(g["src"].shape[0]),
                "neighbor_list_ms_per_step_est": 1e3 * nl_s * (k + 1),
                "neighbor_list_frac_est": nl_s * (k + 1) / step_s if step_s > 0 else None,
                "finite": finite})
    return out


# ---------------------------------------------------------------- torch.profiler

def torch_profile(fn, n_steps, out_dir, warmup=1, row_limit=60):
    """Profile n_steps calls of fn() with torch.profiler (CPU, plus CUDA when available).

    Writes out_dir/trace.json (chrome://tracing or https://ui.perfetto.dev) and
    out_dir/key_averages.txt (operator table sorted by device time on CUDA, else CPU time).
    """
    from torch.profiler import ProfilerActivity, profile, record_function

    os.makedirs(out_dir, exist_ok=True)
    cuda = torch.cuda.is_available()
    activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize()
    with profile(activities=activities) as prof:
        for i in range(n_steps):
            with record_function(f"profile_step_{i}"):
                fn()
            if cuda:
                torch.cuda.synchronize()
    trace = os.path.join(out_dir, "trace.json")
    prof.export_chrome_trace(trace)
    averages = prof.key_averages()
    table = None
    for key in (["device_time_total", "cuda_time_total"] if cuda else []) + ["cpu_time_total"]:
        try:
            table = averages.table(sort_by=key, row_limit=row_limit)
            break
        except (KeyError, AttributeError, ValueError, RuntimeError):
            continue
    if table is None:
        table = averages.table(row_limit=row_limit)
    table_path = os.path.join(out_dir, "key_averages.txt")
    with open(table_path, "w") as f:
        f.write(table)
    return {"trace": trace, "key_averages": table_path, "n_steps": n_steps, "warmup": warmup,
            "activities": ["cpu"] + (["cuda"] if cuda else [])}


# ---------------------------------------------------------------- GPU sampling

NVIDIA_SMI_ARGS = ["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]


def _number(s):
    try:
        x = float(s.strip())
    except ValueError:  # "[N/A]", "[Not Supported]"
        return None
    return x if math.isfinite(x) else None


def parse_nvidia_smi(text):
    """'util, mem' lines (one per GPU) -> [(util_pct or None, mem_used_MiB or None), ...]."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(",")
        if len(parts) >= 2:
            rows.append((_number(parts[0]), _number(parts[1])))
    return rows


def _stats(values):
    vals = [x for x in values if x is not None]
    if not vals:
        return None, None
    return float(np.mean(vals)), float(np.max(vals))


class GpuSampler:
    """Background thread polling nvidia-smi every interval_s seconds; rows go to csv_path.

    Columns: t_s (since start), gpu (index in nvidia-smi order), util_pct, mem_used_MiB.
    If nvidia-smi is not found, start/stop do nothing and summary() reports available=False.
        with GpuSampler("gpu.csv", 1.0) as s:
            ...
        s.summary()
    """

    def __init__(self, csv_path=None, interval_s=1.0, executable="nvidia-smi", timeout_s=5.0):
        self.csv_path = csv_path
        self.interval_s = float(interval_s)
        self.timeout_s = timeout_s
        self.exe = shutil.which(executable)
        self.available = self.exe is not None
        self.samples = []  # (t_s, gpu, util, mem)
        self.errors = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._file = None
        self._t0 = None

    def sample_once(self):
        """Query nvidia-smi once; returns the new rows ([] on failure or when unavailable)."""
        if not self.available:
            return []
        try:
            res = subprocess.run([self.exe, *NVIDIA_SMI_ARGS], capture_output=True, text=True,
                                 timeout=self.timeout_s, check=True)
        except (OSError, subprocess.SubprocessError):
            self.errors += 1
            return []
        t = time.time() - (self._t0 if self._t0 is not None else time.time())
        rows = [(t, i, u, mem) for i, (u, mem) in enumerate(parse_nvidia_smi(res.stdout))]
        with self._lock:
            self.samples.extend(rows)
            if self._file is not None:
                for r in rows:
                    self._file.write(",".join("" if x is None else (f"{x:.3f}" if isinstance(x, float) else str(x))
                                              for x in r) + "\n")
                self._file.flush()
        return rows

    def _loop(self):
        while True:
            self.sample_once()
            if self._stop.wait(self.interval_s):
                break

    def start(self):
        if not self.available or self._thread is not None:
            return self
        self._t0 = time.time()
        self._stop.clear()
        if self.csv_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.csv_path)), exist_ok=True)
            self._file = open(self.csv_path, "w")
            self._file.write("t_s,gpu,util_pct,mem_used_MiB\n")
            self._file.flush()
        self._thread = threading.Thread(target=self._loop, name="GpuSampler", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._thread is None:
            return self
        self._stop.set()
        self._thread.join(timeout=self.interval_s + self.timeout_s + 1.0)
        self._thread = None
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False

    def summary(self):
        with self._lock:
            samples = list(self.samples)
        out = {"available": self.available, "n_samples": len(samples), "interval_s": self.interval_s,
               "errors": self.errors, "csv": self.csv_path if self.available else None}
        u_mean, u_max = _stats([s[2] for s in samples])
        m_mean, m_max = _stats([s[3] for s in samples])
        out.update({"util_mean": u_mean, "util_max": u_max, "mem_used_mean_MiB": m_mean, "mem_used_max_MiB": m_max})
        per_gpu = {}
        for gpu in sorted({s[1] for s in samples}):
            rows = [s for s in samples if s[1] == gpu]
            um, ux = _stats([s[2] for s in rows])
            mm, mx = _stats([s[3] for s in rows])
            per_gpu[str(gpu)] = {"util_mean": um, "util_max": ux, "mem_used_mean_MiB": mm, "mem_used_max_MiB": mx}
        out["per_gpu"] = per_gpu
        return out


# ---------------------------------------------------------------- summary file

def to_jsonable(o):
    """Tensors / numpy values -> plain Python; non-finite floats -> None (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): to_jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_jsonable(v) for v in o]
    if isinstance(o, torch.Tensor):
        return to_jsonable(o.item() if o.dim() == 0 else o.tolist())
    if isinstance(o, np.ndarray):
        return to_jsonable(o.tolist())
    if isinstance(o, np.generic):
        return to_jsonable(o.item())
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if o is None or isinstance(o, (str, int, bool)):
        return o
    return str(o)


def _merge(old, new):
    out = dict(old)
    for k, v in new.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def write_summary(out_dir, data):
    """Merge `data` into out_dir/summary.json (nested dicts merged key by key, new values win). Returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    old = {}
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
        except json.JSONDecodeError:
            old = {}
        if not isinstance(old, dict):
            old = {}
    merged = _merge(old, to_jsonable(data))
    with open(path + ".tmp", "w") as f:
        json.dump(merged, f, indent=2, allow_nan=False)
    os.replace(path + ".tmp", path)
    return path
