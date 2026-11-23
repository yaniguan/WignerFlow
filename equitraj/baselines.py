"""Controlled baselines sharing the exact task, targets, scaling, training loop, and rollout harness of EquiTraj:

  1. `PlainTraj` (`arch: plain`): Non-equivariant Cartesian graph transformer. Same parameter budget and k-frame
     temporal attention, but sees raw xyz vectors without O(3) symmetry. Tests whether built-in E(3) symmetry is
     needed for 10^4-step stability.
  2. `TrajCastBaseline` (`arch: trajcast`): O(3)-equivariant Clebsch-Gordan message-passing baseline following
     TrajCast (Thiemann et al., arXiv:2503.23794, 2025): Markovian 1-frame conditioning (`markovian=True`),
     `e3nn` spherical harmonics + channel-wise Clebsch-Gordan tensor products (`o3.TensorProduct`), gated
     nonlinearities, and deterministic regression head by default (combine with CSVR switch S6 at rollout time
     to match published TrajCast inference, or leave S6 off for pure NVE head-to-head comparison). Setting
     `markovian=False` or `head='flow'` also isolates the CG-tensor-product vs eSCN SO(2)-convolution ablation.
"""

import math

from e3nn import o3
import torch
import torch.nn as nn

from equitraj import so3
from equitraj.layers import (
    MLP,
    EquivariantLinear,
    EquivariantRMSNorm,
    FeedForward,
    Gate,
    TemporalAttention,
    envelope,
    gaussian_rbf,
)
from equitraj.model import EquiTraj, build_graph_plain, timestep_embedding
from equitraj.pair_masks import bond_hop_matrix
from equitraj.prior import RepulsionPrior


def segment_softmax(logits, index, n):
    """Softmax of logits (E, H) over the edges that share the same index (destination node)."""
    mx = torch.full((n, logits.shape[1]), -float("inf"), dtype=logits.dtype, device=logits.device)
    mx = mx.scatter_reduce(0, index[:, None].expand_as(logits), logits, "amax", include_self=True)
    mx_safe = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    ex = torch.where(torch.isfinite(logits), torch.exp(logits - mx_safe[index]), torch.zeros_like(logits))
    den = torch.zeros((n, logits.shape[1]), dtype=logits.dtype, device=logits.device).index_add_(0, index, ex)
    return ex / (den[index] + 1e-12)


def smooth_segment_softmax(logits, env, index, n):
    """Envelope-weighted softmax with a unit sink so weights go smoothly to 0 at the cutoff."""
    mx = torch.zeros((n, logits.shape[1]), dtype=logits.dtype, device=logits.device)
    mx = mx.scatter_reduce(0, index[:, None].expand_as(logits), logits, "amax", include_self=True)
    ex = torch.where(torch.isfinite(logits), torch.exp(logits - mx[index]), torch.zeros_like(logits))
    w = env[:, None] * ex
    den = torch.exp(-mx) + torch.zeros_like(mx).index_add_(0, index, w)
    return w / den[index]


class PlainBlock(nn.Module):
    """Graph attention with raw edge vectors as edge features, then a feed-forward layer. Both with residuals and LayerNorm."""

    def __init__(self, c, heads, n_rbf, cutoff, smooth=False):
        super().__init__()
        self.c, self.heads, self.cutoff, self.n_rbf, self.smooth = c, heads, cutoff, n_rbf, smooth
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
        logits = (q * k).sum(-1) / math.sqrt(hd)
        logits = logits.masked_fill((d >= self.cutoff)[:, None], -float("inf"))
        if self.smooth:
            alpha = smooth_segment_softmax(logits, envelope(d, self.cutoff), dst, h.shape[0])
            msg = (alpha[..., None] * v).reshape(-1, self.c)
        else:
            alpha = segment_softmax(logits, dst, h.shape[0])  # (E, H)
            msg = (alpha[..., None] * v * envelope(d, self.cutoff)[:, None, None]).reshape(-1, self.c)
        h = self.norm1(h + self.o(torch.zeros_like(h).index_add_(0, dst, msg)))
        return self.norm2(h + self.ff(h))


class PlainTraj(EquiTraj):
    def __init__(self, n_types, channels=128, heads=8, n_rbf=32, cutoff=5.0, k=3, n_spatial=2, n_fusion=2, n_denoiser=2,
                 head="flow", use_velocity=True, neighbor_method="brute_force", target="residual", stride_fs=1.0,
                 lmax=None, parity=None, smooth_attention=False, **_unused):
        nn.Module.__init__(self)  # not EquiTraj.__init__: none of the equivariant modules are wanted
        self.lmax, self.c, self.cutoff, self.k, self.n_rbf = 0, channels, cutoff, k, n_rbf
        self.head_type, self.use_velocity, self.neighbor_method = head, use_velocity, neighbor_method
        self.target, self.stride_fs = target, stride_fs
        self.target_scaling = "element_std"
        self.geometry_aware = False
        self.prior = None
        self.use_charges = False
        self.n_ff_types = 0
        self.max_bond_hops = 0
        self.edge_dim = 0
        self.stride_conditioning = False
        self.register_buffer("_chem_charges", None, persistent=False)
        self.register_buffer("_chem_ff_types", None, persistent=False)
        self.register_buffer("_chem_bond_hops", None, persistent=False)
        c = channels
        for name in ["dx_std", "v_std", "out_dx_std", "out_v_std"]:
            self.register_buffer(name, torch.ones(n_types))
        self.type_embed = nn.Embedding(n_types, c)
        self.vec_in = MLP([6, c, c])  # raw (velocity, displacement)
        self.spatial = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff, smooth_attention) for _ in range(n_spatial)])
        self.frame_embed = nn.Parameter(torch.zeros(k, c))
        self.temporal = nn.MultiheadAttention(c, heads, batch_first=True)
        self.fusion = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff, smooth_attention) for _ in range(n_fusion)])
        if head == "regression":
            self.readout = MLP([c, c, 6])
        else:
            self.y_in = nn.Linear(6, c)
            self.tau_in = MLP([c, c, c])
            self.denoiser = nn.ModuleList([PlainBlock(c, heads, n_rbf, cutoff, smooth_attention) for _ in range(n_denoiser)])
            self.readout = MLP([c, c, 6])

    def encode(self, pos, vel, types, box, graphs=None, **_unused):
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
        graph = build_graph_plain(pos.reshape(b * k, n, 3), box_f, self.cutoff, self.neighbor_method) if graphs is None else graphs[0]
        h = self.type_embed(tf) + self.vec_in(torch.cat([v_norm, d_norm], -1))
        for blk in self.spatial:
            h = blk(h, graph)
        h = h.view(b, k, n, self.c).permute(0, 2, 1, 3).reshape(b * n, k, self.c) + self.frame_embed
        h = h[:, -1] + self.temporal(h[:, -1:], h, h, need_weights=False)[0][:, 0]  # newest frame attends to all k frames

        graph_last = build_graph_plain(pos[:, -1], box, self.cutoff, self.neighbor_method) if graphs is None else graphs[1]
        for blk in self.fusion:
            h = blk(h, graph_last)
        return h, graph_last

    def regress(self, cond):
        return self.readout(cond).view(-1, 2, 3)

    def denoise(self, cond, graph, y_tau, tau, candidate_graph_fn=None):
        if candidate_graph_fn is not None:
            graph = candidate_graph_fn(y_tau, tau)
        h = cond + self.y_in(y_tau.reshape(-1, 6)) + self.tau_in(timestep_embedding(tau, self.c))
        for blk in self.denoiser:
            h = blk(h, graph)
        return self.readout(h).view(-1, 2, 3)


def _to_e3nn_irreps(x, lmax):
    """Convert standard layout (N, S, C) -> e3nn flat Irreps layout (N, C * S) for natural parity irreps."""
    n, _, c = x.shape
    parts = []
    for l in range(lmax + 1):
        parts.append(x[:, l * l:(l + 1) ** 2, :].transpose(1, 2).reshape(n, c * (2 * l + 1)))
    return torch.cat(parts, dim=-1)


def _from_e3nn_irreps(z, lmax, c):
    """Convert e3nn flat Irreps layout (N, C * S) -> standard layout (N, S, C)."""
    n = z.shape[0]
    parts = []
    offset = 0
    for l in range(lmax + 1):
        dim = c * (2 * l + 1)
        parts.append(z[:, offset:offset + dim].view(n, c, 2 * l + 1).transpose(1, 2))
        offset += dim
    return torch.cat(parts, dim=1)


class CGBlock(nn.Module):
    """O(3)-equivariant Clebsch-Gordan message-passing block (`e3nn` TensorProduct + Gate + FeedForward).

    Each directed edge forms a channel-wise Clebsch-Gordan tensor product between source features and edge
    spherical harmonics Y_lm(r_ij / |r_ij|), weighted by a smooth radial MLP * cutoff envelope, summed over
    neighbors, normalized by average coordination scale, mixed across channels per irrep, and passed through
    an equivariant Gate + residual FeedForward.
    """

    def __init__(self, lmax, c, n_rbf, cutoff, edge_dim=0, avg_num_neighbors=16.0):
        super().__init__()
        self.lmax, self.c, self.n_rbf, self.cutoff = lmax, c, n_rbf, cutoff
        self.edge_dim = int(edge_dim)
        self.inv_sqrt_deg = 1.0 / math.sqrt(max(float(avg_num_neighbors), 1.0))
        feat_irreps = o3.Irreps([(c, (l, (-1) ** l)) for l in range(lmax + 1)])
        sh_irreps = o3.Irreps([(1, (l, (-1) ** l)) for l in range(lmax + 1)])
        instructions = []
        irreps_out = []
        for i_in, (_, ir_in) in enumerate(feat_irreps):
            for i_sh, (_, ir_sh) in enumerate(sh_irreps):
                for ir_out in ir_in * ir_sh:
                    if ir_out.l <= lmax and ir_out.p == (-1) ** ir_out.l:
                        i_out = len(irreps_out)
                        irreps_out.append((c, ir_out))
                        instructions.append((i_in, i_sh, i_out, "uvu", True))
        tp_out_irreps = o3.Irreps(irreps_out)
        self.sh_irreps = sh_irreps
        self.tp_out_ls = [ir.l for _, ir in tp_out_irreps]
        self.tp = o3.TensorProduct(feat_irreps, sh_irreps, tp_out_irreps, instructions,
                                   shared_weights=False, internal_weights=False)
        self.norm1 = EquivariantRMSNorm(lmax, c)
        self.pre = EquivariantLinear(lmax, c, c)
        self.radial = MLP([n_rbf + self.edge_dim, 64, self.tp.weight_numel])
        self.mix = nn.ModuleList([
            nn.Linear(sum(1 for l_val in self.tp_out_ls if l_val == l) * c, c, bias=(l == 0))
            for l in range(lmax + 1)
        ])
        self.gate = Gate(lmax, c)
        self.out = EquivariantLinear(lmax, c, c)
        self.norm2 = EquivariantRMSNorm(lmax, c)
        self.ff = FeedForward(lmax, c, 2 * c)

    def _collect_by_l(self, z_tp, e, dtype):
        """Group multi-path TensorProduct output (E, dim) by angular degree l -> (E, S, C)."""
        buckets = [[] for _ in range(self.lmax + 1)]
        offset = 0
        for l_val in self.tp_out_ls:
            dim = self.c * (2 * l_val + 1)
            # shape (E, 2l+1, C)
            blk = z_tp[:, offset:offset + dim].view(e, self.c, 2 * l_val + 1).transpose(1, 2)
            buckets[l_val].append(blk)
            offset += dim
        out_parts = []
        for l in range(self.lmax + 1):
            cat_l = torch.cat(buckets[l], dim=-1).to(dtype)  # (E, 2l+1, n_paths * C)
            out_parts.append(self.mix[l](cat_l))
        return torch.cat(out_parts, dim=1)

    def forward(self, x, graph):
        src, dst, vec, dist = graph["src"], graph["dst"], graph["vec"], graph["dist"]
        e = src.shape[0]
        xn = self.pre(self.norm1(x))
        if e == 0:
            return x + self.ff(self.norm2(x))
        rbf = gaussian_rbf(dist, self.cutoff, self.n_rbf)
        if self.edge_dim > 0:
            edge_attr = graph.get("edge_attr")
            if edge_attr is None:
                edge_attr = rbf.new_zeros(e, self.edge_dim)
            rbf = torch.cat([rbf, edge_attr.to(rbf.dtype)], dim=-1)
        env = envelope(dist, self.cutoff)[:, None]
        w = self.radial(rbf) * env.to(rbf.dtype)

        # Compute spherical harmonics & CG tensor product in matching floating precision (float64 or float32).
        tp_buf = next(self.tp.buffers(), None)
        tp_dtype = x.dtype if tp_buf is None else tp_buf.dtype
        unit = vec.to(tp_dtype) / dist.to(tp_dtype).clamp_min(1e-12)[:, None]
        # e3nn spherical_harmonics orders xyz as (x, y, z) when normalize=False/True; match so3.rotate (x, y, z):
        sh = o3.spherical_harmonics(self.sh_irreps, unit, normalize=False, normalization="component")
        x_src_flat = _to_e3nn_irreps(xn[src].to(tp_dtype), self.lmax)
        tp_raw = self.tp(x_src_flat, sh, w.to(tp_dtype))
        msg = self._collect_by_l(tp_raw, e, x.dtype)
        agg = torch.zeros_like(x).index_add_(0, dst, msg) * self.inv_sqrt_deg
        x = x + self.out(self.gate(agg))
        return x + self.ff(self.norm2(x))


class TrajCastBaseline(EquiTraj):
    """O(3)-equivariant Clebsch-Gordan MPNN baseline (TrajCast style, Thiemann et al., 2025).

    Defaults to Markovian single-frame conditioning (`markovian=True`) and deterministic `head='regression'`.
    Accepts `(B, k, N, 3)` windows directly so training, rollout, and evaluation scripts run identically.
    """

    def __init__(self, n_types, lmax=2, channels=64, heads=8, n_rbf=32, cutoff=6.0, k=3,
                 n_spatial=2, n_fusion=2, n_denoiser=2, head="regression", parity=True,
                 use_velocity=True, neighbor_method="auto", target="residual", stride_fs=1.0,
                 repulsion_prior=None, n_ff_types=0, use_charges=False, max_bond_hops=0,
                 target_scaling="mass", markovian=True, avg_num_neighbors=16.0, **_unused):
        nn.Module.__init__(self)
        self.lmax, self.c, self.cutoff, self.k, self.n_rbf = lmax, channels, cutoff, k, n_rbf
        self.head_type, self.use_velocity, self.neighbor_method = head, use_velocity, neighbor_method
        self.target, self.stride_fs = target, float(stride_fs)
        self.target_scaling = target_scaling
        self.markovian = bool(markovian)
        self.geometry_aware = False
        self.stride_conditioning = False
        self.gradient_checkpointing = False
        self.n_ff_types = int(n_ff_types)
        self.use_charges = bool(use_charges)
        self.max_bond_hops = int(max_bond_hops)
        c = channels
        bond_dim = min(16, c) if self.max_bond_hops > 0 else 0
        self.edge_dim = (1 if self.use_charges else 0) + bond_dim

        for name in ["dx_std", "v_std", "out_dx_std", "out_v_std"]:
            self.register_buffer(name, torch.ones(n_types))
        if self.target_scaling == "mass":
            self.register_buffer("mass_dx_std", torch.tensor(1.0))
            self.register_buffer("mass_v_std", torch.tensor(1.0))
        self.register_buffer("_chem_charges", None, persistent=False)
        self.register_buffer("_chem_ff_types", None, persistent=False)
        self.register_buffer("_chem_bond_hops", None, persistent=False)

        self.type_embed = nn.Embedding(n_types, c)
        if self.n_ff_types > 0:
            self.ff_type_embed = nn.Embedding(self.n_ff_types, c)
        if self.use_charges:
            self.charge_in = MLP([1, c, c])
        if self.max_bond_hops > 0:
            self.bond_embed = nn.Embedding(self.max_bond_hops + 1, bond_dim)
        n_vec_in = 1 if self.markovian else 2
        self.vec_in = nn.Linear(n_vec_in, c, bias=False)
        self.scalar_in = MLP([c + (1 if self.markovian else 3), c, c])

        if self.markovian:
            n_layers = n_spatial + n_fusion
            self.mp_blocks = nn.ModuleList([
                CGBlock(lmax, c, n_rbf, cutoff, edge_dim=self.edge_dim, avg_num_neighbors=avg_num_neighbors)
                for _ in range(n_layers)
            ])
        else:
            self.spatial = nn.ModuleList([
                CGBlock(lmax, c, n_rbf, cutoff, edge_dim=self.edge_dim, avg_num_neighbors=avg_num_neighbors)
                for _ in range(n_spatial)
            ])
            self.temporal = TemporalAttention(lmax, c, heads, k_max=max(16, int(k)))
            self.fusion = nn.ModuleList([
                CGBlock(lmax, c, n_rbf, cutoff, edge_dim=self.edge_dim, avg_num_neighbors=avg_num_neighbors)
                for _ in range(n_fusion)
            ])

        if head == "regression":
            self.readout_norm = EquivariantRMSNorm(lmax, c)
            self.readout = EquivariantLinear(lmax, c, c)
            self.y_out = nn.Linear(c, 2, bias=False)
        else:
            self.y_in = nn.Linear(2, c, bias=False)
            self.denoise_scalar = MLP([c + 3, c, c])
            self.denoiser = nn.ModuleList([
                CGBlock(lmax, c, n_rbf, cutoff, edge_dim=self.edge_dim, avg_num_neighbors=avg_num_neighbors)
                for _ in range(n_denoiser)
            ])
            self.readout_norm = EquivariantRMSNorm(lmax, c)
            self.readout = EquivariantLinear(lmax, c, c)
            self.y_out = nn.Linear(c, 2, bias=False)

        self.prior = None
        if repulsion_prior and repulsion_prior.get("enabled", True):
            self.prior = RepulsionPrior(n_types, **{k_p: v_p for k_p, v_p in repulsion_prior.items() if k_p != "enabled"})

    def _apply(self, fn):
        # Keep CG TensorProduct buffers synced when .double() / .float() / .to(device) is called on the module.
        super()._apply(fn)
        return self

    def _embed_nodes(self, types_rep, v_norm, d_norm, ff_rep, q_rep):
        s = so3.n_components(self.lmax)
        t = self.type_embed(types_rep)
        if self.n_ff_types > 0 and ff_rep is not None:
            t = t + self.ff_type_embed(ff_rep)
        if self.use_charges and q_rep is not None:
            t = t + self.charge_in(q_rep[:, None]).to(t.dtype)
        if self.markovian:
            inv = v_norm.norm(dim=-1, keepdim=True)
            s0 = self.scalar_in(torch.cat([t, inv.to(t.dtype)], dim=-1))
            v1 = self.vec_in(v_norm[..., None])
        else:
            inv = torch.stack([v_norm.norm(dim=-1), d_norm.norm(dim=-1), (v_norm * d_norm).sum(-1)], -1)
            s0 = self.scalar_in(torch.cat([t, inv.to(t.dtype)], dim=-1))
            v1 = self.vec_in(torch.stack([v_norm, d_norm], dim=-1))
        x = torch.zeros(types_rep.shape[0], s, self.c, dtype=s0.dtype, device=v_norm.device)
        x[:, 0, :] = s0
        x[:, 1:4, :] = v1.to(x.dtype)
        return x

    def encode(self, pos, vel, types, box, graphs=None, *, charges=None, ff_types=None, bonds=None, stride_fs=None):
        b, k, n, _ = pos.shape
        q, ft, bh = self._resolve_chemistry(n, pos.device, pos.dtype, charges=charges, ff_types=ff_types, bonds=bonds)
        if self.markovian:
            tb = types.repeat(b)
            v_last = vel[:, -1].reshape(-1, 3) / self.v_std[tb][:, None]
            if not self.use_velocity:
                v_last = torch.zeros_like(v_last)
            graph_last = (build_graph_plain(pos[:, -1], box, self.cutoff, self.neighbor_method)
                          if graphs is None else graphs[1])
            graph_last = self._attach_edge_attr(graph_last, n, pos.dtype, charges=q, bond_hops=bh)
            x = self._embed_nodes(
                tb, v_last, None,
                None if ft is None else ft.repeat(b),
                None if q is None else q.repeat(b),
            )
            for blk in self.mp_blocks:
                x = blk(x, graph_last)
            return x, graph_last

        tf = types.repeat(b * k)
        v_norm = vel.reshape(-1, 3) / self.v_std[tf][:, None]
        if not self.use_velocity:
            v_norm = torch.zeros_like(v_norm)
        disp = torch.zeros_like(pos)
        disp[:, 1:] = pos[:, 1:] - pos[:, :-1]
        d_norm = disp.reshape(-1, 3) / self.dx_std[tf][:, None]
        box_f = None if box is None else box[:, None].expand(b, k, 3, 3).reshape(b * k, 3, 3)
        graph = (build_graph_plain(pos.reshape(b * k, n, 3), box_f, self.cutoff, self.neighbor_method)
                 if graphs is None else graphs[0])
        graph = self._attach_edge_attr(graph, n, pos.dtype, charges=q, bond_hops=bh)
        x = self._embed_nodes(
            tf, v_norm, d_norm,
            None if ft is None else ft.repeat(b * k),
            None if q is None else q.repeat(b * k),
        )
        for blk in self.spatial:
            x = blk(x, graph)
        x = self.temporal(x.view(b, k, n, *x.shape[1:])).reshape(b * n, -1, self.c)
        graph_last = (build_graph_plain(pos[:, -1], box, self.cutoff, self.neighbor_method)
                      if graphs is None else graphs[1])
        graph_last = self._attach_edge_attr(graph_last, n, pos.dtype, charges=q, bond_hops=bh)
        for blk in self.fusion:
            x = blk(x, graph_last)
        return x, graph_last

    def regress(self, cond):
        return self.y_out(self.readout(self.readout_norm(cond))[:, 1:4, :]).transpose(1, 2)

    def denoise(self, cond, graph, y_tau, tau, candidate_graph_fn=None):
        if candidate_graph_fn is not None:
            graph = candidate_graph_fn(y_tau, tau)
        x = cond.clone()
        x[:, 1:4, :] = x[:, 1:4, :] + self.y_in(y_tau.transpose(1, 2)).to(x.dtype)
        inv = torch.stack([y_tau[:, 0].norm(dim=-1), y_tau[:, 1].norm(dim=-1), (y_tau[:, 0] * y_tau[:, 1]).sum(-1)], -1)
        x[:, 0, :] = x[:, 0, :] + self.denoise_scalar(torch.cat([timestep_embedding(tau, self.c), inv], -1)).to(x.dtype)
        for blk in self.denoiser:
            x = blk(x, graph)
        return self.y_out(self.readout(self.readout_norm(x))[:, 1:4, :]).transpose(1, 2)
