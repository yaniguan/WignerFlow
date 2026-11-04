"""Explicit symmetric boolean pair exclusions; True means ignore this pair.

Masks are particle-order-specific (N,N), shared across replicas. No exclusions
are inferred from molecule membership or chemical types.
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
