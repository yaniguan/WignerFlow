"""Opt-in inference acceleration. The supplied model is never mutated."""
import copy
import warnings
from contextlib import nullcontext

import torch

from equitraj import so3
from equitraj.neighbors import VerletList, brute_force, cell_list


class FastRollout:
    """Batch replicas as (R,k,N,3), sharing weights but not neighbour caches.

    predict returns physical (dx, velocity); rollout returns saved CPU frames and
    final device contexts. This compute-only rollout does not perform stability
    monitoring: use equitraj.rollout.rollout for production validation.
    CUDA graphs capture only fixed-shape model work, not neighbour construction.
    """
    def __init__(self, model, *, sampling_steps=10, precision="fp32", neighbor="verlet",
                 skin=1.0, rebuild_every=None, compile=False, compile_mode="reduce-overhead",
                 cuda_graphs=False, edge_capacity=None):
        if precision not in {"fp32", "bf16", "fp8"} or sampling_steps < 1:
            raise ValueError("invalid precision or sampling_steps")
        if neighbor not in {"brute", "cell", "verlet"}:
            raise ValueError("neighbor must be brute, cell or verlet")
        if compile_mode not in {"reduce-overhead", "max-autotune"}:
            raise ValueError("invalid compile mode")
        self.model = copy.deepcopy(model).float().eval()
        self.sampling_steps, self.precision = sampling_steps, precision
        self.neighbor, self.skin, self.rebuild_every = neighbor, skin, rebuild_every
        self.cuda_graphs, self.edge_capacity = cuda_graphs, edge_capacity
        self._lists = {}
        self._capture = None
        self._compiled = None
        self._fp32_model = None
        self.effective_precision = precision
        if precision == "fp8":
            try:
                if next(self.model.parameters()).device.type != "cuda":
                    raise RuntimeError("float8 Linear requires supported CUDA hardware")
                if torch.cuda.get_device_capability(next(self.model.parameters()).device) < (8, 9):
                    raise RuntimeError("native float8 requires CUDA compute capability >= 8.9")
                from torchao.float8 import convert_to_float8_training
                original = copy.deepcopy(self.model)
                # torchao replaces nn.Linear only; Wigner and equivariant matmuls
                # retain their original floating-point representation.
                convert_to_float8_training(self.model)
                self._fp32_model = original
            except Exception as exc:
                warnings.warn(f"fp8 unavailable; using fp32: {exc}", RuntimeWarning)
                if 'original' in locals():
                    self.model = original
                self.effective_precision = "fp32"
        if compile:
            try:
                self._compiled = torch.compile(self.model.predict_with_grad, mode=compile_mode)
            except Exception as exc:
                warnings.warn(f"compile unavailable; using eager: {exc}", RuntimeWarning)

    def _graph(self, frames, boxes, keys):
        n = frames.shape[1]
        src, dst, vec = [], [], []
        for f, key in enumerate(keys):
            box = None if boxes is None else boxes[f]
            if key in self._step_pairs:
                s, d, v = self._step_pairs[key]
            elif self.neighbor == "verlet":
                nl = self._lists.setdefault(key, VerletList(self.model.cutoff, self.skin, self.rebuild_every))
                s, d, v = nl(frames[f], box)
            else:
                fn = brute_force if self.neighbor == "brute" else cell_list
                s, d, v = fn(frames[f], box, self.model.cutoff)
            self._step_pairs[key] = (s, d, v)
            src.append(s + f * n)
            dst.append(d + f * n)
            vec.append(v)
        s, d, v = torch.cat(src), torch.cat(dst), torch.cat(vec)
        count = s.numel()
        if self.cuda_graphs:
            capacity = self.edge_capacity or max(1, 2 ** (max(1, count) - 1).bit_length())
            if count > capacity:
                raise ValueError(f"edge capacity {capacity} exceeded by {count}; increase edge_capacity")
            pad = capacity - count
            s = torch.cat((s, s.new_zeros(pad)))
            d = torch.cat((d, d.new_zeros(pad)))
            # Nonzero dummy direction avoids undefined angles. cutoff distance
            # makes both embedding and attention contributions exactly zero.
            dummy = v.new_zeros(pad, 3)
            dummy[:, 1] = self.model.cutoff
            v = torch.cat((v, dummy))
        mask = torch.arange(s.numel(), device=s.device) < count
        dist = torch.linalg.norm(v, dim=-1)
        dist = torch.where(mask, dist, dist.new_full((), self.model.cutoff))
        return dict(src=s, dst=d, vec=v, dist=dist, mask=mask,
                    wigner=so3.edge_wigner(v, self.model.lmax))

    def _autocast(self, device):
        return (torch.autocast(device.type, dtype=torch.bfloat16)
                if self.effective_precision == "bf16" else nullcontext())

    def _eager(self, args):
        with self._autocast(args[0].device):
            fn = self._compiled or self.model.predict_with_grad
            try:
                return fn(*args)
            except Exception as exc:
                if self._compiled is not None:
                    warnings.warn(f"compiled execution failed; using eager: {exc}", RuntimeWarning)
                    self._compiled = None
                    return self._eager(args)
                if self._fp32_model is not None:
                    warnings.warn(f"fp8 execution failed; using fp32: {exc}", RuntimeWarning)
                    self.model, self._fp32_model = self._fp32_model, None
                    self.effective_precision = "fp32"
                    return self._eager(args)
                raise

    @torch.no_grad()
    def predict(self, pos, vel, types, box, masses, noise=None):
        pos, vel = pos.float(), vel.float()
        if box is not None:
            box = box.float()
        r, k, n, _ = pos.shape
        self._step_pairs = {}
        boxes = None if box is None else box[:, None].expand(r, k, 3, 3).reshape(-1, 3, 3)
        graphs = (self._graph(pos.reshape(-1, n, 3), boxes,
                              [(a, t) for a in range(r) for t in range(k)]),
                  self._graph(pos[:, -1], box, [(a, k - 1) for a in range(r)]))
        if noise is None and self.model.head_type != "regression":
            noise = self.model.com_free_noise(r, types, masses, like=pos)
        args = (pos, vel, types, box, masses, self.sampling_steps, noise, graphs)
        if self.cuda_graphs and pos.is_cuda:
            try:
                # Tensor tree copies keep captured addresses stable across calls.
                def signature(x):
                    if isinstance(x, torch.Tensor):
                        return (tuple(x.shape), x.dtype, x.device)
                    if isinstance(x, dict):
                        return tuple((k, signature(v)) for k, v in x.items())
                    if isinstance(x, tuple):
                        return tuple(map(signature, x))
                    return x
                def clone(x):
                    if isinstance(x, torch.Tensor): return x.clone()
                    if isinstance(x, dict): return {k: clone(v) for k, v in x.items()}
                    if isinstance(x, tuple): return tuple(map(clone, x))
                    return x
                def update(dst, src):
                    if isinstance(src, torch.Tensor): dst.copy_(src)
                    elif isinstance(src, dict):
                        for key in src: update(dst[key], src[key])
                    elif isinstance(src, tuple):
                        for a, b in zip(dst, src): update(a, b)
                sig = signature(args)
                if self._capture is None or self._capture[0] != sig:
                    static = clone(args)
                    stream = torch.cuda.Stream(device=pos.device)
                    stream.wait_stream(torch.cuda.current_stream(pos.device))
                    with torch.cuda.stream(stream):
                        for _ in range(3): self._eager(static)
                    torch.cuda.current_stream(pos.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        output = self._eager(static)
                    self._capture = (sig, static, graph, output)
                _, static, graph, output = self._capture
                update(static, args)
                graph.replay()
                return tuple(x.clone().float() for x in output)
            except Exception as exc:
                warnings.warn(f"CUDA graphs unavailable; using eager: {exc}", RuntimeWarning)
                self.cuda_graphs, self._capture = False, None
        elif self.cuda_graphs:
            warnings.warn("CUDA graphs require CUDA; using eager", RuntimeWarning)
            self.cuda_graphs = False
        return tuple(x.float() for x in self._eager(args))

    @torch.no_grad()
    def rollout(self, pos, vel, types, box, masses, n_steps, *, save_every=1):
        if n_steps < 1 or save_every < 1:
            raise ValueError("n_steps and save_every must be positive")
        x, v = pos.double(), vel.float()
        xs, vs = [], []
        for step in range(1, n_steps + 1):
            dx, vn = self.predict(x, v, types, box, masses)
            xn = x[:, -1] + dx.double()
            x = torch.cat((x[:, 1:], xn[:, None]), 1)
            v = torch.cat((v[:, 1:], vn[:, None]), 1)
            if step % save_every == 0:
                xs.append(xn.float().cpu())
                vs.append(vn.cpu())
        return dict(positions=torch.stack(xs) if xs else None,
                    velocities=torch.stack(vs) if vs else None,
                    context_positions=x, context_velocities=v, steps_done=n_steps)
