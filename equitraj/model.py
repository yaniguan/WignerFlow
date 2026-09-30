"""EquiTraj: predicts (Δx, v_new) over a stride of n MD steps from the last k frames.

Data flow (see PLAN.md, Stage 1):
    each of the k frames -> spatial encoder (graph attention on that frame's neighbor graph)
    -> temporal attention per atom over the k frames
    -> fusion blocks on the newest frame's graph -> condition features c
    -> head A (regression) or head B (flow matching)

Units: positions Å, velocities Å/fs. Targets are scaled per element:
    dx_norm = Δx / dx_std[element],  v_norm = v / v_std[element]
The stds come from the training data (see `compute_stats` in dataset.py) and are stored in the model.

Shapes: B systems in a batch, all with the same N atoms; k frames of context.
"""

import math

import torch
import torch.nn as nn

from equitraj import so3
from equitraj.layers import MLP, Block, EquivariantLinear, TemporalAttention, envelope, gaussian_rbf
from equitraj.neighbors import brute_force, cell_list


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
                 use_velocity=True, neighbor_method="brute_force"):
        super().__init__()
        self.lmax, self.c, self.cutoff, self.k = lmax, channels, cutoff, k
        self.n_rbf = n_rbf
        self.head_type = head
        self.use_velocity = use_velocity
        self.neighbor_method = neighbor_method
        c = channels

        # per-element scales, filled from training data
        self.register_buffer("dx_std", torch.ones(n_types))
        self.register_buffer("v_std", torch.ones(n_types))

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

    def to_physical(self, y_norm, types, b):
        """(B*N, 2, 3) normalized -> (dx, v) each (B, N, 3) in Å and Å/fs."""
        tb = types.repeat(b)
        dx = y_norm[:, 0] * self.dx_std[tb][:, None]
        v = y_norm[:, 1] * self.v_std[tb][:, None]
        n = types.shape[0]
        return dx.view(b, n, 3), v.view(b, n, 3)

    def to_normalized(self, dx, v, types):
        b, n, _ = dx.shape
        tb = types.repeat(b)
        return torch.stack([dx.reshape(-1, 3) / self.dx_std[tb][:, None], v.reshape(-1, 3) / self.v_std[tb][:, None]], 1)

    def com_free_noise(self, b, types, masses, generator=None, like=None):
        """Standard normal noise in normalized units whose physical (dx, v) have no center-of-mass part."""
        n = types.shape[0]
        eps = torch.randn(b * n, 2, 3, generator=generator, dtype=like.dtype, device=like.device)
        dx, v = self.to_physical(eps, types, b)
        return self.to_normalized(remove_com(dx, masses), remove_com(v, masses), types)

    # ------------------------------------------------------------ training losses

    def loss(self, pos, vel, types, box, masses, dx_target, v_target):
        """MSE loss in normalized units. dx_target, v_target: (B, N, 3)."""
        b = pos.shape[0]
        cond, graph = self.encode(pos, vel, types, box)
        y = self.to_normalized(dx_target, v_target, types)
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
        dx, v = self.to_physical(y, types, b)
        return remove_com(dx, masses), remove_com(v, masses)
