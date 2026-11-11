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
from equitraj.pair_masks import bond_hop_matrix
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
    When candidate_graph_fn is supplied (geometry_aware=True), edges are rebuilt on the
    interpolated candidate frame x_t + tau * dx(y_tau).
    """

    def __init__(self, lmax, c, heads, n_rbf, cutoff, n_blocks, parity, edge_dim=0):
        super().__init__()
        self.c = c
        self.y_in = nn.Linear(2, c, bias=False)  # 2 input vectors -> c vector channels
        self.scalar_in = MLP([c + 3, c, c])
        self.blocks = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity, edge_dim=edge_dim) for _ in range(n_blocks)])
        self.readout = EquivariantLinear(lmax, c, c)
        self.y_out = nn.Linear(c, 2, bias=False)

    def forward(self, cond, graph, y_tau, tau_per_node, candidate_graph_fn=None):
        if candidate_graph_fn is not None:
            graph = candidate_graph_fn(y_tau, tau_per_node)
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
                 repulsion_prior=None, n_ff_types=0, use_charges=False, max_bond_hops=0,
                 geometry_aware=False, target_scaling="element_std", stride_conditioning=False):
        super().__init__()
        if isinstance(n_ff_types, bool) or int(n_ff_types) != n_ff_types or n_ff_types < 0:
            raise ValueError("n_ff_types must be a nonnegative integer")
        if isinstance(max_bond_hops, bool) or int(max_bond_hops) != max_bond_hops or max_bond_hops < 0:
            raise ValueError("max_bond_hops must be a nonnegative integer")
        if target_scaling not in ("element_std", "mass"):
            raise ValueError(f"target_scaling must be 'element_std' or 'mass', got {target_scaling!r}")
        if not math.isfinite(stride_fs) or stride_fs <= 0:
            raise ValueError("stride_fs must be finite and positive")
        self.lmax, self.c, self.cutoff, self.k = lmax, channels, cutoff, k
        self.n_rbf = n_rbf
        self.head_type = head
        self.use_velocity = use_velocity
        self.neighbor_method = neighbor_method
        self.target = target
        self.stride_fs = float(stride_fs)
        self.n_ff_types = int(n_ff_types)
        self.use_charges = bool(use_charges)
        self.max_bond_hops = int(max_bond_hops)
        self.geometry_aware = bool(geometry_aware)
        self.target_scaling = target_scaling
        self.stride_conditioning = bool(stride_conditioning)
        c = channels
        bond_dim = min(16, c) if self.max_bond_hops > 0 else 0
        self.edge_dim = (1 if self.use_charges else 0) + bond_dim

        # per-element scales, filled from training data
        self.register_buffer("dx_std", torch.ones(n_types))
        self.register_buffer("v_std", torch.ones(n_types))
        self.register_buffer("out_dx_std", torch.ones(n_types))
        self.register_buffer("out_v_std", torch.ones(n_types))
        if self.target_scaling == "mass":
            self.register_buffer("mass_dx_std", torch.tensor(1.0))
            self.register_buffer("mass_v_std", torch.tensor(1.0))

        # optional non-persistent chemistry context (kept out of state_dict)
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
        if self.stride_conditioning:
            self.stride_in = MLP([c, c, c])
        self.vec_in = nn.Linear(2, c, bias=False)  # (velocity, displacement) -> c vector channels
        self.scalar_in = MLP([c + 3, c, c])
        self.edge_radial = MLP([n_rbf + self.edge_dim, 64, (lmax + 1) * c])
        self.edge_type = nn.Linear(c, c, bias=False)
        self.spatial = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity, edge_dim=self.edge_dim) for _ in range(n_spatial)])
        self.temporal = TemporalAttention(lmax, c, heads)
        self.fusion = nn.ModuleList([Block(lmax, c, heads, n_rbf, cutoff, parity, edge_dim=self.edge_dim) for _ in range(n_fusion)])
        if head == "regression":
            self.readout = EquivariantLinear(lmax, c, c)
            self.y_out = nn.Linear(c, 2, bias=False)
        else:
            self.denoiser = Denoiser(lmax, c, heads, n_rbf, cutoff, n_denoiser, parity, edge_dim=self.edge_dim)
        # optional short-range repulsion applied to predictions (equitraj/prior.py); when disabled no
        # module or buffer is registered, so state dicts are the same as before
        self.prior = None
        if repulsion_prior and repulsion_prior.get("enabled", True):
            self.prior = RepulsionPrior(n_types, **{k: v for k, v in repulsion_prior.items() if k != "enabled"})

    def set_chemistry(self, *, charges=None, ff_types=None, bonds=None, n_atoms=None):
        """Attach optional per-particle chemistry and topology metadata for subsequent calls.

        Stored as non-persistent buffers so checkpoints stay clean while rollout() and
        FastRollout() inherit the chemistry automatically. Explicit keyword arguments on
        encode/loss/predict override these values.
        """
        device = self.dx_std.device
        dtype = self.dx_std.dtype
        if charges is not None:
            q = torch.as_tensor(charges, device=device, dtype=dtype)
            if q.ndim != 1 or not torch.isfinite(q).all():
                raise ValueError("charges must be a finite 1D tensor of shape (N,)")
            n_atoms = q.shape[0] if n_atoms is None else n_atoms
            if q.shape != (n_atoms,):
                raise ValueError("charges length does not match n_atoms")
            self.register_buffer("_chem_charges", q, persistent=False)
        else:
            self.register_buffer("_chem_charges", None, persistent=False)

        if ff_types is not None:
            ft = torch.as_tensor(ff_types, device=device)
            if ft.dtype == torch.bool or ft.is_floating_point() or ft.ndim != 1:
                raise ValueError("ff_types must be a 1D integer tensor of shape (N,)")
            ft = ft.to(dtype=torch.long)
            n_atoms = ft.shape[0] if n_atoms is None else n_atoms
            if ft.shape != (n_atoms,):
                raise ValueError("ff_types length does not match n_atoms")
            if self.n_ff_types > 0 and ((ft < 0).any() or (ft >= self.n_ff_types).any()):
                raise ValueError("ff_types indices out of range for n_ff_types")
            self.register_buffer("_chem_ff_types", ft, persistent=False)
        else:
            self.register_buffer("_chem_ff_types", None, persistent=False)

        if bonds is not None:
            if n_atoms is None:
                raise ValueError("n_atoms is required when setting bonds without charges or ff_types")
            hops = bond_hop_matrix(bonds, n_atoms, max_hops=max(1, self.max_bond_hops), device=device)
            self.register_buffer("_chem_bond_hops", hops, persistent=False)
        else:
            self.register_buffer("_chem_bond_hops", None, persistent=False)

    def _resolve_chemistry(self, n_atoms, device, dtype, charges=None, ff_types=None, bonds=None):
        q = self._chem_charges if charges is None else torch.as_tensor(charges, device=device, dtype=dtype)
        if self.use_charges:
            if q is None:
                raise ValueError("charges required when use_charges=True")
            q = q.to(device=device, dtype=dtype)
            if q.shape != (n_atoms,) or not torch.isfinite(q).all():
                raise ValueError("charges must be finite with shape (N,)")
        else:
            q = None

        ft = self._chem_ff_types if ff_types is None else torch.as_tensor(ff_types, device=device)
        if self.n_ff_types > 0:
            if ft is None:
                raise ValueError("ff_types required when n_ff_types > 0")
            if ft.dtype == torch.bool or ft.is_floating_point():
                raise ValueError("ff_types must contain integer indices")
            ft = ft.to(device=device, dtype=torch.long)
            if ft.shape != (n_atoms,) or (ft < 0).any() or (ft >= self.n_ff_types).any():
                raise ValueError("ff_types must have shape (N,) with values in [0, n_ff_types)")
        else:
            ft = None

        bh = None
        if self.max_bond_hops > 0:
            if bonds is not None:
                bh = bond_hop_matrix(bonds, n_atoms, max_hops=self.max_bond_hops, device=device)
            elif self._chem_bond_hops is not None:
                if self._chem_bond_hops.shape != (n_atoms, n_atoms):
                    raise ValueError("cached bond hops do not match n_atoms")
                bh = self._chem_bond_hops.to(device=device).clamp_max(self.max_bond_hops)
            else:
                bh = torch.zeros((n_atoms, n_atoms), dtype=torch.long, device=device)
        return q, ft, bh

    def _resolve_stride(self, stride_fs, b, dtype, device):
        if stride_fs is None:
            return None
        s = torch.as_tensor(stride_fs, dtype=dtype, device=device)
        if s.ndim == 0:
            s = s.expand(b)
        elif s.shape != (b,):
            raise ValueError("stride_fs must be a positive scalar or have shape (B,)")
        if not torch.isfinite(s).all() or (s <= 0).any():
            raise ValueError("stride_fs must be finite and positive")
        return s

    def _attach_edge_attr(self, graph, n_atoms, dtype, charges=None, bond_hops=None):
        if self.edge_dim == 0:
            return graph
        g = dict(graph)
        atom_src = g["src"] % n_atoms
        atom_dst = g["dst"] % n_atoms
        parts = []
        if self.use_charges:
            parts.append((charges[atom_src] * charges[atom_dst])[:, None].to(dtype))
        if self.max_bond_hops > 0:
            parts.append(self.bond_embed(bond_hops[atom_src, atom_dst]).to(dtype))
        g["edge_attr"] = torch.cat(parts, dim=-1)
        return g

    # ------------------------------------------------------------ encoder

    def embed(self, types, v_norm, d_norm, graph, n_nodes, *, ff_types=None, charges=None, stride_emb=None):
        """Initial node features (n_nodes, S, c) for all frames."""
        s = so3.n_components(self.lmax)
        x = torch.zeros(n_nodes, s, self.c, dtype=v_norm.dtype, device=v_norm.device)
        t = self.type_embed(types)
        if self.n_ff_types > 0 and ff_types is not None:
            t = t + self.ff_type_embed(ff_types)
        if self.use_charges and charges is not None:
            t = t + self.charge_in(charges[:, None])
        if stride_emb is not None:
            t = t + stride_emb
        inv = torch.stack([v_norm.norm(dim=-1), d_norm.norm(dim=-1), (v_norm * d_norm).sum(-1)], -1)
        x[:, 0, :] = self.scalar_in(torch.cat([t, inv], -1))
        x[:, 1:4, :] = self.vec_in(torch.stack([v_norm, d_norm], -1))  # (n, 3, 2) -> (n, 3, c)

        # "edge-degree" embedding: each edge adds a learned function of its length on the m=0
        # components in the edge frame, rotated back (= a function of the edge's spherical harmonics)
        e = graph["src"].shape[0]
        rbf = gaussian_rbf(graph["dist"], self.cutoff, self.n_rbf)
        if self.edge_dim > 0 and "edge_attr" in graph:
            rbf = torch.cat([rbf, graph["edge_attr"].to(rbf.dtype)], dim=-1)
        rad = self.edge_radial(rbf).view(e, self.lmax + 1, self.c)
        rad = rad * self.edge_type(t[graph["src"]])[:, None, :] * envelope(graph["dist"], self.cutoff)[:, None, None]
        ye = torch.zeros(e, s, self.c, dtype=x.dtype, device=x.device)
        ye[:, [so3.index(l, 0) for l in range(self.lmax + 1)], :] = rad.to(ye.dtype)
        agg = torch.zeros_like(x).index_add_(0, graph["dst"], so3.from_edge_frame(graph["wigner"], ye).to(x.dtype))
        return x + agg / 10.0

    def encode(self, pos, vel, types, box, graphs=None, *, charges=None, ff_types=None, bonds=None, stride_fs=None):
        """pos, vel: (B, k, N, 3) in Å and Å/fs, oldest frame first. types: (N,). box: (B, 3, 3) or None.

        Returns condition features (B*N, S, c) for the newest frame and that frame's graph.
        """
        b, k, n, _ = pos.shape
        q, ft, bh = self._resolve_chemistry(n, pos.device, pos.dtype, charges=charges, ff_types=ff_types, bonds=bonds)
        s_vec = self._resolve_stride(stride_fs, b, pos.dtype, pos.device)
        tf = types.repeat(b * k)
        v_norm = vel.reshape(-1, 3) / self.v_std[tf][:, None]
        if not self.use_velocity:
            v_norm = torch.zeros_like(v_norm)
        disp = torch.zeros_like(pos)
        disp[:, 1:] = pos[:, 1:] - pos[:, :-1]  # displacement since the previous frame (0 for the oldest)
        if s_vec is not None:
            disp = disp / (s_vec / self.stride_fs).view(b, 1, 1, 1)
        d_norm = disp.reshape(-1, 3) / self.dx_std[tf][:, None]

        stride_emb = None
        if self.stride_conditioning:
            s_cond = s_vec if s_vec is not None else torch.full((b,), self.stride_fs, dtype=pos.dtype, device=pos.device)
            se = self.stride_in(timestep_embedding(s_cond / 100.0, self.c))
            stride_emb = se[:, None, None, :].expand(b, k, n, self.c).reshape(b * k * n, self.c)

        box_f = None if box is None else box[:, None].expand(b, k, 3, 3).reshape(b * k, 3, 3)
        graph = (build_graph(pos.reshape(b * k, n, 3), box_f, self.cutoff, self.lmax, self.neighbor_method)
                 if graphs is None else graphs[0])
        graph = self._attach_edge_attr(graph, n, pos.dtype, charges=q, bond_hops=bh)
        x = self.embed(
            tf, v_norm, d_norm, graph, b * k * n,
            ff_types=None if ft is None else ft.repeat(b * k),
            charges=None if q is None else q.repeat(b * k),
            stride_emb=stride_emb,
        )
        for blk in self.spatial:
            x = blk(x, graph)
        x = self.temporal(x.view(b, k, n, *x.shape[1:]))  # (B, N, S, c)

        box_last = None if box is None else box
        graph_last = (build_graph(pos[:, -1], box_last, self.cutoff, self.lmax, self.neighbor_method)
                      if graphs is None else graphs[1])
        graph_last = self._attach_edge_attr(graph_last, n, pos.dtype, charges=q, bond_hops=bh)
        x = x.reshape(b * n, *x.shape[2:])
        for blk in self.fusion:
            x = blk(x, graph_last)
        return x, graph_last

    # ------------------------------------------------------------ scaling helpers

    def scale_up(self, y, types, b, masses=None):
        """(B*N, 2, 3) normalized outputs -> the two physical target vectors, each (B, N, 3)."""
        n = types.shape[0]
        if self.target_scaling == "mass":
            if masses is None:
                raise ValueError("masses required when target_scaling='mass'")
            inv_sqrt_m = masses.to(device=y.device, dtype=y.dtype).rsqrt().repeat(b)[:, None]
            a = (y[:, 0] * self.mass_dx_std.to(y.dtype) * inv_sqrt_m).view(b, n, 3)
            c = (y[:, 1] * self.mass_v_std.to(y.dtype) * inv_sqrt_m).view(b, n, 3)
            return a, c
        tb = types.repeat(b)
        a = (y[:, 0] * self.out_dx_std[tb][:, None]).view(b, n, 3)
        c = (y[:, 1] * self.out_v_std[tb][:, None]).view(b, n, 3)
        return a, c

    def scale_down(self, a, c, types, masses=None):
        b = a.shape[0]
        if self.target_scaling == "mass":
            if masses is None:
                raise ValueError("masses required when target_scaling='mass'")
            sqrt_m = masses.to(device=a.device, dtype=a.dtype).sqrt().repeat(b)[:, None]
            return torch.stack([a.reshape(-1, 3) * sqrt_m / self.mass_dx_std.to(a.dtype),
                                c.reshape(-1, 3) * sqrt_m / self.mass_v_std.to(a.dtype)], 1)
        tb = types.repeat(b)
        return torch.stack([a.reshape(-1, 3) / self.out_dx_std[tb][:, None],
                            c.reshape(-1, 3) / self.out_v_std[tb][:, None]], 1)

    def _coasting_dt(self, stride_fs, v_last):
        if stride_fs is None:
            return self.stride_fs
        s = self._resolve_stride(stride_fs, v_last.shape[0], v_last.dtype, v_last.device)
        return s[:, None, None]

    def targets_to_y(self, dx, v_new, v_last, types, masses=None, stride_fs=None):
        """Physical (Δx, v_new) -> normalized model outputs, given the last context velocity."""
        if self.target == "residual":
            dt = self._coasting_dt(stride_fs, v_last)
            return self.scale_down(dx - v_last * dt, v_new - v_last, types, masses=masses)
        return self.scale_down(dx, v_new, types, masses=masses)

    def y_to_targets(self, y, types, v_last, masses=None, stride_fs=None):
        """Normalized model outputs -> physical (Δx, v_new)."""
        a, c = self.scale_up(y, types, v_last.shape[0], masses=masses)
        if self.target == "residual":
            dt = self._coasting_dt(stride_fs, v_last)
            return a + v_last * dt, c + v_last
        return a, c

    def com_free_noise(self, b, types, masses, generator=None, like=None):
        """Standard normal noise in normalized units whose physical counterpart has no center-of-mass part."""
        n = types.shape[0]
        gen_device = like.device if generator is None else generator.device
        eps = torch.randn(b * n, 2, 3, generator=generator, dtype=like.dtype, device=gen_device).to(like.device)
        a, c = self.scale_up(eps, types, b, masses=masses)
        return self.scale_down(remove_com(a, masses), remove_com(c, masses), types, masses=masses)

    def _candidate_graph_builder(self, pos_last, v_last, types, box, masses, base_graph, q, bh, stride_fs):
        if not self.geometry_aware:
            return None
        b, n, _ = pos_last.shape

        def build_candidate(y_t, tau_n):
            if bool((tau_n == 0).all()):
                return base_graph
            dx_cand, _ = self.y_to_targets(y_t, types, v_last, masses=masses, stride_fs=stride_fs)
            pos_cand = pos_last + tau_n.view(b, n, 1) * dx_cand
            g = build_graph(pos_cand, box, self.cutoff, self.lmax, self.neighbor_method)
            return self._attach_edge_attr(g, n, pos_last.dtype, charges=q, bond_hops=bh)

        return build_candidate

    # ------------------------------------------------------------ training losses

    def loss(self, pos, vel, types, box, masses, dx_target, v_target, *,
             charges=None, ff_types=None, bonds=None, stride_fs=None):
        """MSE loss in normalized units. dx_target, v_target: (B, N, 3)."""
        b, _, n, _ = pos.shape
        q, _, bh = self._resolve_chemistry(n, pos.device, pos.dtype, charges=charges, ff_types=ff_types, bonds=bonds)
        cond, graph = self.encode(pos, vel, types, box, charges=charges, ff_types=ff_types,
                                  bonds=bonds, stride_fs=stride_fs)
        y = self.targets_to_y(dx_target, v_target, vel[:, -1], types, masses=masses, stride_fs=stride_fs)
        if self.head_type == "regression":
            pred = self.y_out(self.readout(cond)[:, 1:4, :]).transpose(1, 2)
            return ((pred - y) ** 2).mean()
        # flow matching: y_tau = (1 - tau) eps + tau y, target velocity y - eps
        eps = self.com_free_noise(b, types, masses, like=y)
        tau = torch.rand(b, dtype=y.dtype, device=y.device)
        tau_n = tau.repeat_interleave(types.shape[0])
        y_tau = (1 - tau_n)[:, None, None] * eps + tau_n[:, None, None] * y
        cand_fn = self._candidate_graph_builder(pos[:, -1], vel[:, -1], types, box, masses, graph, q, bh, stride_fs)
        pred = self.denoiser(cond, graph, y_tau, tau_n, candidate_graph_fn=cand_fn)
        return ((pred - (y - eps)) ** 2).mean()

    # ------------------------------------------------------------ prediction

    @torch.no_grad()
    def predict(self, pos, vel, types, box, masses, n_steps=10, noise=None, sampler="euler", *,
                charges=None, ff_types=None, bonds=None, stride_fs=None):
        """One autoregressive step. Returns (dx, v_new) each (B, N, 3), center-of-mass motion removed."""
        return self.predict_with_grad(pos, vel, types, box, masses, n_steps, noise, None, sampler,
                                      charges=charges, ff_types=ff_types, bonds=bonds, stride_fs=stride_fs)

    def predict_with_grad(self, pos, vel, types, box, masses, n_steps=10, noise=None, graphs=None,
                          sampler="euler", *, charges=None, ff_types=None, bonds=None, stride_fs=None):
        """Same as predict, but differentiable (used for unrolled training)."""
        if sampler not in ("euler", "heun"):
            raise ValueError(f"sampler must be 'euler' or 'heun', got {sampler!r}")
        if n_steps < 1:
            raise ValueError("n_steps must be positive")
        b, _, n, _ = pos.shape
        q, _, bh = self._resolve_chemistry(n, pos.device, pos.dtype, charges=charges, ff_types=ff_types, bonds=bonds)
        cond, graph = self.encode(pos, vel, types, box, graphs=graphs, charges=charges,
                                  ff_types=ff_types, bonds=bonds, stride_fs=stride_fs)
        if self.head_type == "regression":
            y = self.y_out(self.readout(cond)[:, 1:4, :]).transpose(1, 2)
        else:
            y = self.com_free_noise(b, types, masses, like=cond) if noise is None else noise
            cand_fn = self._candidate_graph_builder(pos[:, -1], vel[:, -1], types, box, masses, graph, q, bh, stride_fs)
            dt = 1.0 / n_steps
            for i in range(n_steps):  # Euler or 2nd-order Heun integration of the learned flow from tau=0 to 1
                tau0 = torch.full((b * n,), i * dt, dtype=cond.dtype, device=cond.device)
                v0 = self.denoiser(cond, graph, y, tau0, candidate_graph_fn=cand_fn)
                if sampler == "euler":
                    y = y + dt * v0
                else:
                    y_pred = y + dt * v0
                    tau1 = torch.full((b * n,), (i + 1) * dt, dtype=cond.dtype, device=cond.device)
                    v1 = self.denoiser(cond, graph, y_pred, tau1, candidate_graph_fn=cand_fn)
                    y = y + 0.5 * dt * (v0 + v1)
        dx, v = self.y_to_targets(y, types, vel[:, -1], masses=masses, stride_fs=stride_fs)
        dx, v = remove_com(dx, masses), remove_com(v, masses)
        if self.prior is not None:  # mass-weighted pair pushes: the center of mass stays fixed
            dx, v = self.prior(pos[:, -1], dx, v, types, box, masses)
        return dx, v
