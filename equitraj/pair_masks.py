"""Boolean pair masks: True means skip the pair.

Masks must be symmetric, have shape (N,N), and use the current particle order.
All systems share the mask. Molecules and atom types do not set exclusions.
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
