# Chemistry conditioning and flow-matching options

All options below are **opt-in** and default to off, so existing configurations and `v0` checkpoints keep an identical `state_dict` and behavior.

## 1. Covalent bond hops and automatic pair exclusions (`equitraj/pair_masks.py`)

```python
from equitraj.pair_masks import bond_hop_matrix, exclusions_from_bonds

hops = bond_hop_matrix(bonds, n_atoms, max_hops=3)        # (N, N) LongTensor in {0, 1, 2, 3}
excluded = exclusions_from_bonds(bonds, n_atoms, max_hops=2)  # symmetric BoolTensor for 1-2 and 1-3 pairs
```

- `bond_hop_matrix` computes shortest-path covalent distances (`1` = 1–2 bond, `2` = 1–3 angle, `3` = 1–4 torsion, `0` = non-bonded or beyond `max_hops`).
- `exclusions_from_bonds` builds the symmetric `(N, N)` mask consumed by `RepulsionPrior`, `overlap_penalty`, `estimate_rmin`, `StabilityMonitor`, and `rollout`.

## 2. Force-field types, partial charges, and bond-topology conditioning (`equitraj/model.py`)

```yaml
model:
  n_ff_types: 6          # 0 (default): element types only
  use_charges: true      # false (default): no partial-charge node/edge features
  max_bond_hops: 3       # 0 (default): no covalent topological edge embeddings
```

- **Node features (`l = 0`):** `ff_types` (`LongTensor(N,)`) adds a force-field atom-type embedding to distinguish environments sharing an element (such as carbonyl vs. ether oxygen in carbonate solvents); `charges` (`Tensor(N,)` in `e`) passes through a scalar MLP.
- **Edge features (`l = 0`):** `use_charges` appends $q_i q_j$ to radial edge inputs; `max_bond_hops > 0` appends a learned topological hop embedding (`0..max_bond_hops`) in both `EquiTraj.embed` and `GraphAttention`. Edge contributions remain multiplied by the smooth cutoff `envelope`, preserving $C^2$ continuity across `cutoff`.
- **Attaching metadata for rollouts:** call `model.set_chemistry(charges=..., ff_types=..., bonds=..., n_atoms=N)` once before `rollout()` or `FastRollout()`. These tensors are stored as non-persistent buffers (moved with `.to(device)` and excluded from `state_dict`), or can be passed per call via `encode` / `loss` / `predict`.
- **Dataset loading (`equitraj/dataset.py`):** when HDF5 trajectories carry `pff_metadata` (`equitraj/topology.py`), `preflight` checks cross-trajectory agreement on `charges`, `ff_types`, `molecule_ids`, `bonds`, `constraints`, and `drude_parent_map`, and `TrajWindows` exposes `ds.charges`, `ds.ff_types`, `ds.ff_type_names`, and `ds.bonds`.

## 3. Geometry-aware denoiser and Heun flow sampler

```yaml
model:
  geometry_aware: true        # false (default): reuse newest-frame graph across flow steps
  target_scaling: mass        # element_std (default) or mass (FlashMD-style sqrt(m) scaling)
  stride_conditioning: true   # false (default): fixed stride model
```

- **Geometry-aware flow denoiser (`geometry_aware=True`):** at flow time $\tau \in (0, 1]$, `Denoiser` evaluates message passing on the interpolated candidate geometry $\mathbf{x}_t + \tau\,\Delta\mathbf{x}(\mathbf{y}_\tau)$ under minimum-image PBC while reusing $\mathbf{x}_t$ at $\tau = 0$.
- **Second-order Heun sampler (`sampler="heun"`):** inference-only option on `model.predict(..., sampler="heun")` and `FastRollout(..., sampler="heun")`. Applies predictor–corrector trapezoidal integration over $\tau \in [0, 1]$ without changing checkpoints.
- **Mass-scaled targets (`target_scaling="mass"`):** normalizes $(\Delta\mathbf{x}\sqrt{m},\ \Delta\mathbf{v}\sqrt{m})$ by global RMS buffers `mass_dx_std` and `mass_v_std` (`compute_stats` in `equitraj/dataset.py`), aligning target variance with kinetic-energy equipartition across mixed-mass species.
- **Stride conditioning (`stride_conditioning=True`):** embeds `stride_fs` into node scalars and rescales context displacements by `stride_fs / model.stride_fs` so a single network can be trained and queried at multiple time strides.
