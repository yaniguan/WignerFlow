"""Equivariant building blocks.

Node features: x of shape (N, S, C), S = (lmax+1)^2, layout described in so3.py.
All features have natural parity (-1)^l, so the layers are O(3)-equivariant
(rotations and reflections), not just SO(3).

Blocks:
    EquivariantLinear   mixes channels separately for each l
    EquivariantRMSNorm  layer norm for l=0, RMS norm per l for l>0
    Gate                SiLU on scalars, sigmoid gates on l>0
    SO2Linear           eSCN linear map in the edge frame (the cheap replacement for a CG tensor product)
    GraphAttention      Equiformer-v2-style attention over neighbors
    FeedForward         per-node MLP
    TemporalAttention   per-atom attention over the past k frames
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from equitraj import so3


# ---------------------------------------------------------------- radial functions

def gaussian_rbf(d, cutoff, n):
    """Gaussian radial basis on [0, cutoff]: (E,) -> (E, n)."""
    centers = torch.linspace(0.0, cutoff, n, device=d.device, dtype=d.dtype)
    width = cutoff / n
    return torch.exp(-0.5 * ((d[:, None] - centers) / width) ** 2)


def envelope(d, cutoff, p=5):
    """Smooth cutoff (DimeNet): 1 at d=0, goes to 0 at d=cutoff with zero first and second derivative.

    Multiplying edge contributions by this makes the network output a smooth function of the
    positions even when atoms cross the cutoff. That matters for a dynamics model.
    """
    x = d / cutoff
    e = 1 - (p + 1) * (p + 2) / 2 * x**p + p * (p + 2) * x ** (p + 1) - p * (p + 1) / 2 * x ** (p + 2)
    return torch.where(x < 1, e, torch.zeros_like(e))


class MLP(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------- per-node layers

class EquivariantLinear(nn.Module):
    """y_l = x_l @ W_l for each l; a bias only on l=0."""

    def __init__(self, lmax, c_in, c_out):
        super().__init__()
        self.lmax = lmax
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(c_in, c_out) / math.sqrt(c_in)) for _ in range(lmax + 1)])
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x):
        out = []
        for l in range(self.lmax + 1):
            out.append(x[:, l * l:(l + 1) ** 2, :] @ self.weights[l])
        out = torch.cat(out, dim=1)
        out[:, 0, :] = out[:, 0, :] + self.bias
        return out


class EquivariantRMSNorm(nn.Module):
    """LayerNorm on the l=0 channels; for l>0, divide by the RMS over (m, channels) of that l.

    eps is added to the mean square. It must not be tiny: when a node's l>0 features are close
    to zero (e.g. an atom with no neighbors), dividing by their RMS amplifies float32 rounding
    noise, and the output stops being equivariant (measured: 4e-3 relative error with eps=1e-6,
    6e-6 with eps=1e-3).
    """

    def __init__(self, lmax, c, eps=1e-3):
        super().__init__()
        self.lmax = lmax
        self.eps = eps
        self.norm0 = nn.LayerNorm(c)
        self.scale = nn.Parameter(torch.ones(lmax, c))

    def forward(self, x):
        out = [self.norm0(x[:, 0:1, :])]
        for l in range(1, self.lmax + 1):
            xl = x[:, l * l:(l + 1) ** 2, :]
            rms = torch.sqrt(xl.pow(2).mean(dim=(1, 2), keepdim=True) + self.eps)
            out.append(xl / rms * self.scale[l - 1])
        return torch.cat(out, dim=1)


class Gate(nn.Module):
    """SiLU on the l=0 channels; each l>0 channel is multiplied by a sigmoid computed from the scalars."""

    def __init__(self, lmax, c):
        super().__init__()
        self.lmax = lmax
        self.gates = nn.Linear(c, lmax * c) if lmax > 0 else None

    def forward(self, x):
        s = x[:, 0, :]
        out = [F.silu(s)[:, None, :]]
        if self.lmax > 0:
            g = torch.sigmoid(self.gates(s)).view(x.shape[0], self.lmax, -1)
            for l in range(1, self.lmax + 1):
                out.append(x[:, l * l:(l + 1) ** 2, :] * g[:, l - 1:l, :])
        return torch.cat(out, dim=1)


class FeedForward(nn.Module):
    def __init__(self, lmax, c, hidden):
        super().__init__()
        self.lin1 = EquivariantLinear(lmax, c, hidden)
        self.gate = Gate(lmax, hidden)
        self.lin2 = EquivariantLinear(lmax, hidden, c)

    def forward(self, x):
        return self.lin2(self.gate(self.lin1(x)))


# ---------------------------------------------------------------- edge-frame layers

class SO2Linear(nn.Module):
    """Linear map on edge-frame features that commutes with rotations about the edge (y) axis.

    In the edge frame, a rotation by angle g about y acts on each pair (x_{+m}, x_{-m}) like
    multiplication of the complex number x_{+m} + i x_{-m} by exp(i m g). A linear map commutes
    with that iff it acts on each pair like a complex matrix W1 + i W2:
        y_{+m} = W1 x_{+m} - W2 x_{-m}
        y_{-m} = W2 x_{+m} + W1 x_{-m}
    and for m = 0 any real matrix works. W1, W2 may mix all l >= m (that is what makes this
    equivalent to a CG tensor product with the edge's spherical harmonics, at O(L^3) cost).

    Reflections: a mirror through a plane that contains the y axis flips the sign of all m < 0
    components and keeps all m >= 0 components, for every l (checked numerically in tests).
    The W2 term changes sign under that mirror, so parity=True (default) sets W2 = 0, which
    makes the layer O(3)-equivariant.
    """

    def __init__(self, lmax, c_in, c_out, parity=True):
        super().__init__()
        self.lmax = lmax
        self.c_out = c_out
        self.parity = parity
        self.lin_m0 = nn.Linear((lmax + 1) * c_in, (lmax + 1) * c_out)
        self.w1 = nn.ModuleList()
        self.w2 = nn.ModuleList()
        for m in range(1, lmax + 1):
            n_l = lmax + 1 - m  # number of l values with l >= m
            self.w1.append(nn.Linear(n_l * c_in, n_l * c_out, bias=False))
            if not parity:
                self.w2.append(nn.Linear(n_l * c_in, n_l * c_out, bias=False))
        # index lists into the S axis
        self.idx_m0 = [so3.index(l, 0) for l in range(lmax + 1)]
        self.idx_plus = [[so3.index(l, m) for l in range(m, lmax + 1)] for m in range(1, lmax + 1)]
        self.idx_minus = [[so3.index(l, -m) for l in range(m, lmax + 1)] for m in range(1, lmax + 1)]

    def forward(self, x):
        e = x.shape[0]
        out = x.new_zeros(e, so3.n_components(self.lmax), self.c_out)
        out[:, self.idx_m0, :] = self.lin_m0(x[:, self.idx_m0, :].reshape(e, -1)).view(e, -1, self.c_out)
        for k in range(self.lmax):
            xp = x[:, self.idx_plus[k], :].reshape(e, -1)
            xm = x[:, self.idx_minus[k], :].reshape(e, -1)
            yp = self.w1[k](xp)
            ym = self.w1[k](xm)
            if not self.parity:
                yp = yp - self.w2[k](xm)
                ym = ym + self.w2[k](xp)
            out[:, self.idx_plus[k], :] = yp.view(e, -1, self.c_out)
            out[:, self.idx_minus[k], :] = ym.view(e, -1, self.c_out)
        return out


def per_l_scale(x, scale, lmax):
    """Multiply every component of degree l by scale[:, l, :] (E, lmax+1, C). Invariant, so allowed in the edge frame."""
    reps = torch.tensor([2 * l + 1 for l in range(lmax + 1)], device=x.device)
    return x * torch.repeat_interleave(scale, reps, dim=1)


def segment_softmax(logits, index, n):
    """Softmax of logits (E, H) over all edges that share the same index (receiving node)."""
    h = logits.shape[1]
    idx = index[:, None].expand(-1, h)
    max_per_node = torch.full((n, h), -torch.inf, dtype=logits.dtype, device=logits.device)
    max_per_node = max_per_node.scatter_reduce(0, idx, logits, reduce="amax", include_self=True)
    ex = torch.exp(logits - max_per_node[index])
    denom = torch.zeros((n, h), dtype=logits.dtype, device=logits.device).index_add_(0, index, ex)
    return ex / (denom[index] + 1e-16)


class GraphAttention(nn.Module):
    """Equivariant attention over neighbors (a simplified Equiformer-v2 block).

    For each edge src -> dst:
      1. rotate x_src and x_dst into the edge frame and concatenate them
      2. scale each l by a learned function of the distance (invariant)
      3. h = SO2Linear(...)                              (E, S, C)
      4. attention logits from the l=0 part of h         (E, heads), softmax over edges into dst
      5. value = SO2Linear(Gate(h)), rotated back to the global frame
      6. sum of attention * envelope(d) * value into dst, then a linear layer
    """

    def __init__(self, lmax, c, heads, n_rbf, cutoff, parity=True):
        super().__init__()
        assert c % heads == 0
        self.lmax, self.c, self.heads, self.n_rbf, self.cutoff = lmax, c, heads, n_rbf, cutoff
        self.radial = MLP([n_rbf, 64, (lmax + 1) * 2 * c])
        self.so2_1 = SO2Linear(lmax, 2 * c, c, parity)
        self.logits = MLP([c, c, heads])
        self.gate = Gate(lmax, c)
        self.so2_2 = SO2Linear(lmax, c, c, parity)
        self.out = EquivariantLinear(lmax, c, c)

    def forward(self, x, graph):
        src, dst, dist, wigner = graph["src"], graph["dst"], graph["dist"], graph["wigner"]
        e = src.shape[0]
        xe = torch.cat([so3.to_edge_frame(wigner, x[src]), so3.to_edge_frame(wigner, x[dst])], dim=-1)
        rad = self.radial(gaussian_rbf(dist, self.cutoff, self.n_rbf)).view(e, self.lmax + 1, 2 * self.c)
        h = self.so2_1(per_l_scale(xe, rad, self.lmax))
        alpha = segment_softmax(self.logits(h[:, 0, :]), dst, x.shape[0])  # (E, heads)
        alpha = alpha * envelope(dist, self.cutoff)[:, None]
        v = so3.from_edge_frame(wigner, self.so2_2(self.gate(h)))  # (E, S, C)
        v = v.view(e, v.shape[1], self.heads, -1) * alpha[:, None, :, None]
        agg = torch.zeros_like(x).index_add_(0, dst, v.view(e, v.shape[1], -1))
        return self.out(agg)


class Block(nn.Module):
    """x + attention(norm(x)), then x + feedforward(norm(x))."""

    def __init__(self, lmax, c, heads, n_rbf, cutoff, parity=True):
        super().__init__()
        self.norm1 = EquivariantRMSNorm(lmax, c)
        self.attn = GraphAttention(lmax, c, heads, n_rbf, cutoff, parity)
        self.norm2 = EquivariantRMSNorm(lmax, c)
        self.ff = FeedForward(lmax, c, 2 * c)

    def forward(self, x, graph):
        x = x + self.attn(self.norm1(x), graph)
        return x + self.ff(self.norm2(x))


class TemporalAttention(nn.Module):
    """Each atom attends over its own features in the past k frames.

    Attention weights use only invariants (the l=0 channels and the norm of each l>0 block),
    plus a learned embedding of how many frames back each frame is. The values are mixed per l
    with a linear layer, so the output stays equivariant. The query is the newest frame.
    Input (B, k, N, S, C) -> output (B, N, S, C).
    """

    def __init__(self, lmax, c, heads, k_max=16):
        super().__init__()
        assert c % heads == 0
        self.lmax, self.heads = lmax, heads
        n_inv = c * (lmax + 1)
        self.query = nn.Linear(n_inv, c)
        self.key = nn.Linear(n_inv, c)
        self.age = nn.Embedding(k_max, c)
        self.value = EquivariantLinear(lmax, c, c)
        self.out = EquivariantLinear(lmax, c, c)
        self.norm = EquivariantRMSNorm(lmax, c)

    def invariants(self, h):
        parts = [h[..., 0, :]]
        for l in range(1, self.lmax + 1):
            parts.append(torch.linalg.norm(h[..., l * l:(l + 1) ** 2, :], dim=-2))
        return torch.cat(parts, dim=-1)

    def forward(self, h):
        b, k, n, s, c = h.shape
        hn = self.norm(h.reshape(-1, s, c)).view(b, k, n, s, c)
        inv = self.invariants(hn)  # (B, k, N, n_inv)
        age = self.age(torch.arange(k - 1, -1, -1, device=h.device))  # frame k-1 is the newest (age 0)
        q = self.query(inv[:, -1]).view(b, n, self.heads, -1)
        key = (self.key(inv) + age[None, :, None, :]).view(b, k, n, self.heads, -1)
        logits = torch.einsum("bnhd,bknhd->bknh", q, key) / math.sqrt(q.shape[-1])
        w = torch.softmax(logits, dim=1)  # over frames
        v = self.value(hn.reshape(-1, s, c)).view(b, k, n, s, self.heads, -1)
        mixed = torch.einsum("bknh,bknshd->bnshd", w, v).reshape(b * n, s, c)
        return h[:, -1] + self.out(mixed).view(b, n, s, c)
