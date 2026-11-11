"""Boolean pair masks and covalent bond-hop matrices.

Masks must be symmetric, have shape (N,N), and use the current particle order.
All systems share the mask. Molecules and atom types do not set exclusions
automatically unless `exclusions_from_bonds` is called explicitly.
"""
import torch


def exclusion_mask(excluded_pairs, n_atoms, device):
    if excluded_pairs is None:
        return torch.zeros((n_atoms, n_atoms), dtype=torch.bool, device=device)
    mask = torch.as_tensor(excluded_pairs, device=device)
    if mask.dtype != torch.bool or mask.shape != (n_atoms, n_atoms):
        raise ValueError("excluded_pairs must be a boolean (N,N) mask")
    if not torch.equal(mask, mask.T):
        raise ValueError("excluded_pairs must be symmetric")
    return mask


def bond_hop_matrix(bonds, n_atoms, max_hops=3, device=None):
    """Shortest topological bond distance (1..max_hops) for each atom pair, or 0 otherwise.

    bonds: None, empty, or integer index pairs of shape (B, 2) with 0 <= i != j < n_atoms.
    Returns a symmetric (n_atoms, n_atoms) LongTensor with zeros on the diagonal.
    """
    if isinstance(n_atoms, bool) or int(n_atoms) != n_atoms or n_atoms < 0:
        raise ValueError("n_atoms must be a nonnegative integer")
    if isinstance(max_hops, bool) or int(max_hops) != max_hops or max_hops < 1:
        raise ValueError("max_hops must be a positive integer")
    n_atoms, max_hops = int(n_atoms), int(max_hops)
    hops = torch.zeros((n_atoms, n_atoms), dtype=torch.long, device=device)
    if bonds is None or n_atoms == 0:
        return hops
    raw = torch.as_tensor(bonds)
    if raw.dtype == torch.bool:
        raise ValueError("bonds must contain integer particle indices")
    if raw.numel() == 0:
        return hops
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if raw.is_floating_point():
        if not torch.isfinite(raw).all() or not torch.equal(raw, raw.floor()):
            raise ValueError("bonds must contain finite integer particle indices")
    idx = raw.to(dtype=torch.long, device=device)
    if (idx < 0).any() or (idx >= n_atoms).any() or (idx[:, 0] == idx[:, 1]).any():
        raise ValueError("bond indices must satisfy 0 <= i != j < n_atoms")
    adj = torch.zeros((n_atoms, n_atoms), dtype=torch.bool, device=device)
    adj[idx[:, 0], idx[:, 1]] = True
    adj[idx[:, 1], idx[:, 0]] = True
    reached = torch.eye(n_atoms, dtype=torch.bool, device=device)
    frontier = adj & ~reached
    for hop in range(1, max_hops + 1):
        if not frontier.any():
            break
        hops[frontier] = hop
        reached = reached | frontier
        if hop < max_hops:
            frontier = ((frontier.float() @ adj.float()) > 0) & ~reached
    return hops


def exclusions_from_bonds(bonds, n_atoms, max_hops=2, device=None):
    """Symmetric boolean (N, N) exclusion mask for pairs within 1..max_hops covalent bonds."""
    return bond_hop_matrix(bonds, n_atoms, max_hops=max_hops, device=device) > 0

