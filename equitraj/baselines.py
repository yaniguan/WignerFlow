"""Non-equivariant baseline: the same task, inputs, targets, heads and training as EquiTraj, but a plain graph transformer.

PlainTraj sees raw Cartesian vectors (velocity, displacement, edge vectors) as ordinary numbers. It is invariant to
translations (only relative vectors enter) and equivariant to permutations of atoms, but not to rotations or reflections.
It answers one question: does building in E(3) symmetry help the stability of long rollouts?

Everything else is shared with EquiTraj (it is a subclass): per-element output scaling, residual or direct targets, the
regression head A and the flow-matching head B, center-of-mass removal, losses and prediction. Only the encoder and the
two head networks are replaced. Select it with `model: {arch: plain, ...}` in the config. Width (`channels`) is chosen so
that the parameter count is close to the equivariant model; train.py prints the count.
"""

import math

import torch
import torch.nn as nn

from equitraj.layers import MLP, envelope, gaussian_rbf
from equitraj.model import EquiTraj, build_graph_plain, timestep_embedding


def segment_softmax(logits, index, n):
    """Softmax of logits (E, H) over the edges that share the same index (destination node)."""
    mx = torch.full((n, logits.shape[1]), -float("inf"), dtype=logits.dtype, device=logits.device)
    mx = mx.scatter_reduce(0, index[:, None].expand_as(logits), logits, "amax", include_self=True)
    ex = torch.exp(logits - mx[index])
    den = torch.zeros((n, logits.shape[1]), dtype=logits.dtype, device=logits.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-12)


class PlainBlock(nn.Module):
    """Graph attention with raw edge vectors as edge features, then a feed-forward layer. Both with residuals and LayerNorm."""

    def __init__(self, c, heads, n_rbf, cutoff):
        super().__init__()
        self.c, self.heads, self.cutoff, self.n_rbf = c, heads, cutoff, n_rbf
        self.edge = MLP([n_rbf + 3, 64, c])
        self.q, self.k, self.v, self.o = (nn.Linear(c, c) for _ in range(4))
        self.ek, self.ev = nn.Linear(c, c, bias=False), nn.Linear(c, c, bias=False)
        self.norm1, self.norm2 = nn.LayerNorm(c), nn.LayerNorm(c)
        self.ff = MLP([c, 2 * c, c])

    def forward(self, h, graph):
        src, dst, d = graph["src"], graph["dst"], graph["dist"]
        e = self.edge(torch.cat([gaussian_rbf(d, self.cutoff, self.n_rbf), graph["vec"] / self.cutoff], -1))
        hd = self.c // self.heads
        q = self.q(h)[dst].view(-1, self.heads, hd)
        k = (self.k(h)[src] + self.ek(e)).view(-1, self.heads, hd)
        v = (self.v(h)[src] + self.ev(e)).view(-1, self.heads, hd)
        alpha = segment_softmax((q * k).sum(-1) / math.sqrt(hd), dst, h.shape[0])  # (E, H)
        msg = (alpha[..., None] * v * envelope(d, self.cutoff)[:, None, None]).reshape(-1, self.c)
        h = self.norm1(h + self.o(torch.zeros_like(h).index_add_(0, dst, msg)))
        return self.norm2(h + self.ff(h))


class PlainTraj(EquiTraj):
    def __init__(self, n_types, channels=128, heads=8, n_rbf=32, cutoff=5.0, k=3, n_spatial=2, n_fusion=2, n_denoiser=2,
                 head="flow", use_velocity=True, neighbor_method="brute_force", target="residual", stride_fs=1.0,
                 lmax=None, parity=None):
        nn.Module.__init__(self)  # not EquiTraj.__init__: none of the equivariant modules are wanted
        self.lmax, self.c, self.cutoff, self.k, self.n_rbf = 0, channels, cutoff, k, n_rbf
        self.head_type, self.use_velocity, self.neighbor_method = head, use_velocity, neighbor_method
        self.target, self.stride_fs = target, stride_fs
        c = channels
        for name in ["dx_std", "v_std", "out_dx_std", "out_v_std"]:
            self.register_buffer(name, torch.ones(n_types))
        self.type_embed = nn.Embedding(n_types, c)
        self.vec_in = MLP([6, c, c])  # raw (velocity, displacement)
        self.spatial = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff) for _ in range(n_spatial)])
        self.frame_embed = nn.Parameter(torch.zeros(k, c))
        self.temporal = nn.MultiheadAttention(c, heads, batch_first=True)
        self.fusion = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff) for _ in range(n_fusion)])
        if head == "regression":
            self.readout = MLP([c, c, 6])
        else:
            self.y_in = nn.Linear(6, c)
            self.tau_in = MLP([c, c, c])
            self.denoiser = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff) for _ in range(n_denoiser)])
            self.readout = MLP([c, c, 6])

    def encode(self, pos, vel, types, box):
        """Same signature as EquiTraj.encode. Returns per-atom features (B*N, c) of the newest frame and its graph."""
        b, k, n, _ = pos.shape
        tf = types.repeat(b * k)
        v_norm = vel.reshape(-1, 3) / self.v_std[tf][:, None]
        if not self.use_velocity:
            v_norm = torch.zeros_like(v_norm)
        disp = torch.zeros_like(pos)
        disp[:, 1:] = pos[:, 1:] - pos[:, :-1]
        d_norm = disp.reshape(-1, 3) / self.dx_std[tf][:, None]

        box_f = None if box is None else box[:, None].expand(b, k, 3, 3).reshape(b * k, 3, 3)
        graph = build_graph_plain(pos.reshape(b * k, n, 3), box_f, self.cutoff, self.neighbor_method)
        h = self.type_embed(tf) + self.vec_in(torch.cat([v_norm, d_norm], -1))
        for blk in self.spatial:
            h = blk(h, graph)
        h = h.view(b, k, n, self.c).permute(0, 2, 1, 3).reshape(b * n, k, self.c) + self.frame_embed
        h = h[:, -1] + self.temporal(h[:, -1:], h, h, need_weights=False)[0][:, 0]  # newest frame attends to all k frames

        graph_last = build_graph_plain(pos[:, -1], box, self.cutoff, self.neighbor_method)
        for blk in self.fusion:
            h = blk(h, graph_last)
        return h, graph_last

    def regress(self, cond):
        return self.readout(cond).view(-1, 2, 3)

    def denoise(self, cond, graph, y_tau, tau):
        h = cond + self.y_in(y_tau.reshape(-1, 6)) + self.tau_in(timestep_embedding(tau, self.c))
        for blk in self.denoiser:
            h = blk(h, graph)
        return self.readout(h).view(-1, 2, 3)
