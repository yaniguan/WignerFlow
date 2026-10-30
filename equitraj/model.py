"""EquiTraj: predicts (Δx, v_new) over a stride of n MD steps from the last k frames.

Data flow (see PLAN.md, Stage 1):
    each of the k frames -> spatial encoder (graph attention on that frame's neighbor graph)
    -> temporal attention per atom over the k frames
    -> fusion blocks on the newest frame's graph -> condition features c
    -> head A (regression) or head B (flow matching)

Units: positions Å, velocities Å/fs.

Targets (option `target`):
    "residual" (default): the model predicts the change relative to coasting at constant velocity,
        a = Δx - v_t * Δt   and   b = v_new - v_t
    "direct": the model predicts Δx and v_new themselves.
Why residual: an imperfect network shrinks its outputs toward zero. With direct targets that means
v_new -> 0, i.e. the system loses kinetic energy every step (seen in the first argon run: -4% per
step, frozen after ~100 steps). With residual targets, shrinking toward zero means "keep the old
velocity", so errors no longer cool the system systematically.
Each target component is divided by its per-element std from the training data (out_dx_std,
out_v_std); inputs use dx_std, v_std (see `compute_stats` in dataset.py).

Shapes: B systems in a batch, all with the same N atoms; k frames of context.
"""

import math

import torch
import torch.nn as nn

from equitraj import so3
from equitraj.layers import MLP, Block, EquivariantLinear, TemporalAttention, envelope, gaussian_rbf
from equitraj.neighbors import brute_force, cell_list
from equitraj.prior import RepulsionPrior


def build_graph(pos, box, cutoff, lmax, method="brute_force"):
    """Neighbor graph for F frames at once, concatenated.

    pos: (F, N, 3). box: (F, 3, 3) or None. Node index of atom i in frame f is f*N + i.
    Returns a dict with src, dst, vec (points from dst to src), dist, and the edge Wigner-D matrices.
    """
    fn = cell_list if method == "cell_list" else brute_force
    n = pos.shape[1]
    srcs, dsts, vecs = [], [], []
    for f in range(pos.shape[0]):
        s, d, v = fn(pos[f], None if box is None else box[f], cutoff)
        srcs.append(s + f * n)
        dsts.append(d + f * n)
        vecs.append(v)
    src, dst, vec = torch.cat(srcs), torch.cat(dsts), torch.cat(vecs)
    return {"src": src, "dst": dst, "vec": vec, "dist": torch.linalg.norm(vec, dim=-1),
            "wigner": so3.edge_wigner(vec, lmax)}


def remove_com(vec, masses):
    """Remove the mass-weighted mean from per-atom vectors. vec: (B, N, 3), masses: (N,)."""
    w = masses / masses.sum()
    return vec - torch.einsum("n,bnd->bd", w, vec)[:, None, :]


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the flow time t in [0, 1]: (B,) -> (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half)
    ang = 1000.0 * t[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class Denoiser(nn.Module):
    """Head B network: predicts the flow-matching velocity for the noisy target y_tau.

    y_tau (B*N, 2, 3) is added into the l=1 channels of the condition features, tau and the
    norms of y_tau into the l=0 channels; then a few graph attention blocks and a linear readout.
    """

    def __init__(self, lmax, c, heads, n_rbf, cutoff, n_blocks, parity):
        super().__init__()
        self.c = c
        self.y_in = nn.Linear(2, c, bias=False)  # 2 input vectors -> c vector channels
        self.scalar_in = MLP([c + 3, c, c])
        self.blocks = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity) for _ in range(n_blocks)])
        self.readout = EquivariantLinear(lmax, c, c)
        self.y_out = nn.Linear(c, 2, bias=False)

    def forward(self, cond, graph, y_tau, tau_per_node):
        x = cond.clone()
        x[:, 1:4, :] = x[:, 1:4, :] + self.y_in(y_tau.transpose(1, 2))  # (B*N, 3, c)
        inv = torch.stack([y_tau[:, 0].norm(dim=-1), y_tau[:, 1].norm(dim=-1), (y_tau[:, 0] * y_tau[:, 1]).sum(-1)], -1)
        x[:, 0, :] = x[:, 0, :] + self.scalar_in(torch.cat([timestep_embedding(tau_per_node, self.c), inv], -1))
        for blk in self.blocks:
            x = blk(x, graph)
        return self.y_out(self.readout(x)[:, 1:4, :]).transpose(1, 2)  # (B*N, 2, 3)


class EquiTraj(nn.Module):
    def __init__(self, n_types, lmax=2, channels=128, heads=8, n_rbf=32, cutoff=5.0, k=3,
                 n_spatial=2, n_fusion=2, n_denoiser=2, head="flow", parity=True,
                 use_velocity=True, neighbor_method="brute_force", target="residual", stride_fs=1.0,
                 repulsion_prior=None):
        super().__init__()
        self.lmax, self.c, self.cutoff, self.k = lmax, channels, cutoff, k
        self.n_rbf = n_rbf
        self.head_type = head
        self.use_velocity = use_velocity
        self.neighbor_method = neighbor_method
        self.target = target
        self.stride_fs = stride_fs
        c = channels

        # per-element scales, filled from training data
        self.register_buffer("dx_std", torch.ones(n_types))
        self.register_buffer("v_std", torch.ones(n_types))
        self.register_buffer("out_dx_std", torch.ones(n_types))
        self.register_buffer("out_v_std", torch.ones(n_types))

        self.type_embed = nn.Embedding(n_types, c)
        self.vec_in = nn.Linear(2, c, bias=False)  # (velocity, displacement) -> c vector channels
        self.scalar_in = MLP([c + 3, c, c])
        self.edge_radial = MLP([n_rbf, 64, (lmax + 1) * c])
        self.edge_type = nn.Linear(c, c, bias=False)
        self.spatial = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity) for _ in range(n_spatial)])
        self.temporal = TemporalAttention(lmax, c, heads)
        self.fusion = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity) for _ in range(n_fusion)])
        if head == "regression":
            self.readout = EquivariantLinear(lmax, c, c)
            self.y_out = nn.Linear(c, 2, bias=False)
        else:
            self.denoiser = Denoiser(lmax, c, heads, n_rbf, cutoff, n_denoiser, parity)
        # optional short-range repulsion applied to predictions (equitraj/prior.py); when disabled no
        # module or buffer is registered, so state dicts are the same as before
        self.prior = None
        if repulsion_prior and repulsion_prior.get("enabled", True):
            self.prior = RepulsionPrior(n_types, **{k: v for k, v in repulsion_prior.items() if k != "enabled"})

    # ------------------------------------------------------------ encoder

    def embed(self, types, v_norm, d_norm, graph, n_nodes):
        """Initial node features (n_nodes, S, c) for all frames."""
        s = so3.n_components(self.lmax)
        x = torch.zeros(n_nodes, s, self.c, dtype=v_norm.dtype, device=v_norm.device)
        t = self.type_embed(types)
        inv = torch.stack([v_norm.norm(dim=-1), d_norm.norm(dim=-1), (v_norm * d_norm).sum(-1)], -1)
        x[:, 0, :] = self.scalar_in(torch.cat([t, inv], -1))
        x[:, 1:4, :] = self.vec_in(torch.stack([v_norm, d_norm], -1))  # (n, 3, 2) -> (n, 3, c)

        # "edge-degree" embedding: each edge adds a learned function of its length on the m=0
        # components in the edge frame, rotated back (= a function of the edge's spherical harmonics)
        e = graph["src"].shape[0]
        rad = self.edge_radial(gaussian_rbf(graph["dist"], self.cutoff, self.n_rbf)).view(e, self.lmax + 1, self.c)
        rad = rad * self.edge_type(t[graph["src"]])[:, None, :] * envelope(graph["dist"], self.cutoff)[:, None, None]
        ye = torch.zeros(e, s, self.c, dtype=x.dtype, device=x.device)
        ye[:, [so3.index(l, 0) for l in range(self.lmax + 1)], :] = rad
        agg = torch.zeros_like(x).index_add_(0, graph["dst"], so3.from_edge_frame(graph["wigner"], ye))
        return x + agg / 10.0

    def encode(self, pos, vel, types, box):
        """pos, vel: (B, k, N, 3) in Å and Å/fs, oldest frame first. types: (N,). box: (B, 3, 3) or None.

        Returns condition features (B*N, S, c) for the newest frame and that frame's graph.
        """
        b, k, n, _ = pos.shape
        tf = types.repeat(b * k)
        v_norm = vel.reshape(-1, 3) / self.v_std[tf][:, None]
        if not self.use_velocity:
            v_norm = torch.zeros_like(v_norm)
        disp = torch.zeros_like(pos)
        disp[:, 1:] = pos[:, 1:] - pos[:, :-1]  # displacement since the previous frame (0 for the oldest)
        d_norm = disp.reshape(-1, 3) / self.dx_std[tf][:, None]

        box_f = None if box is None else box[:, None].expand(b, k, 3, 3).reshape(b * k, 3, 3)
        graph = build_graph(pos.reshape(b * k, n, 3), box_f, self.cutoff, self.lmax, self.neighbor_method)
        x = self.embed(tf, v_norm, d_norm, graph, b * k * n)
        for blk in self.spatial:
            x = blk(x, graph)
        x = self.temporal(x.view(b, k, n, *x.shape[1:]))  # (B, N, S, c)

        box_last = None if box is None else box
        graph_last = build_graph(pos[:, -1], box_last, self.cutoff, self.lmax, self.neighbor_method)
        x = x.reshape(b * n, *x.shape[2:])
        for blk in self.fusion:
            x = blk(x, graph_last)
        return x, graph_last

    # ------------------------------------------------------------ scaling helpers

    def scale_up(self, y, types, b):
        """(B*N, 2, 3) normalized outputs -> the two physical target vectors, each (B, N, 3)."""
        tb = types.repeat(b)
        n = types.shape[0]
        a = (y[:, 0] * self.out_dx_std[tb][:, None]).view(b, n, 3)
        c = (y[:, 1] * self.out_v_std[tb][:, None]).view(b, n, 3)
        return a, c

    def scale_down(self, a, c, types):
        b = a.shape[0]
        tb = types.repeat(b)
        return torch.stack([a.reshape(-1, 3) / self.out_dx_std[tb][:, None],
                            c.reshape(-1, 3) / self.out_v_std[tb][:, None]], 1)

    def targets_to_y(self, dx, v_new, v_last, types):
        """Physical (Δx, v_new) -> normalized model outputs, given the last context velocity."""
        if self.target == "residual":
            return self.scale_down(dx - v_last * self.stride_fs, v_new - v_last, types)
        return self.scale_down(dx, v_new, types)

    def y_to_targets(self, y, types, v_last):
        """Normalized model outputs -> physical (Δx, v_new)."""
        a, c = self.scale_up(y, types, v_last.shape[0])
        if self.target == "residual":
            return a + v_last * self.stride_fs, c + v_last
        return a, c

    def com_free_noise(self, b, types, masses, generator=None, like=None):
        """Standard normal noise in normalized units whose physical counterpart has no center-of-mass part."""
        n = types.shape[0]
        eps = torch.randn(b * n, 2, 3, generator=generator, dtype=like.dtype, device=like.device)
        a, c = self.scale_up(eps, types, b)
        return self.scale_down(remove_com(a, masses), remove_com(c, masses), types)

    # ------------------------------------------------------------ training losses

    def loss(self, pos, vel, types, box, masses, dx_target, v_target):
        """MSE loss in normalized units. dx_target, v_target: (B, N, 3)."""
        b = pos.shape[0]
        cond, graph = self.encode(pos, vel, types, box)
        y = self.targets_to_y(dx_target, v_target, vel[:, -1], types)
        if self.head_type == "regression":
            pred = self.y_out(self.readout(cond)[:, 1:4, :]).transpose(1, 2)
            return ((pred - y) ** 2).mean()
        # flow matching: y_tau = (1 - tau) eps + tau y, target velocity y - eps
        eps = self.com_free_noise(b, types, masses, like=y)
        tau = torch.rand(b, dtype=y.dtype, device=y.device)
        tau_n = tau.repeat_interleave(types.shape[0])
        y_tau = (1 - tau_n)[:, None, None] * eps + tau_n[:, None, None] * y
        pred = self.denoiser(cond, graph, y_tau, tau_n)
        return ((pred - (y - eps)) ** 2).mean()

    # ------------------------------------------------------------ prediction

    @torch.no_grad()
    def predict(self, pos, vel, types, box, masses, n_steps=10, noise=None):
        """One autoregressive step. Returns (dx, v_new) each (B, N, 3), center-of-mass motion removed."""
        return self.predict_with_grad(pos, vel, types, box, masses, n_steps, noise)

    def predict_with_grad(self, pos, vel, types, box, masses, n_steps=10, noise=None):
        """Same as predict, but differentiable (used for unrolled training)."""
        b = pos.shape[0]
        cond, graph = self.encode(pos, vel, types, box)
        if self.head_type == "regression":
            y = self.y_out(self.readout(cond)[:, 1:4, :]).transpose(1, 2)
        else:
            y = self.com_free_noise(b, types, masses, like=cond) if noise is None else noise
            n = types.shape[0]
            for i in range(n_steps):  # Euler integration of the learned flow from tau=0 to 1
                tau = torch.full((b * n,), i / n_steps, dtype=cond.dtype, device=cond.device)
                y = y + self.denoiser(cond, graph, y, tau) / n_steps
        dx, v = self.y_to_targets(y, types, vel[:, -1])
        dx, v = remove_com(dx, masses), remove_com(v, masses)
        if self.prior is not None:  # mass-weighted pair pushes: the center of mass stays fixed
            dx, v = self.prior(pos[:, -1], dx, v, types, box, masses)
        return dx, v
