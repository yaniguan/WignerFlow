"""Neighbor lists with the minimum-image convention (PyTorch).

Both functions return (src, dst, vec) for one frame:
    src, dst: (E,) atom indices, one entry per directed edge src -> dst
    vec:      (E, 3) = position[src] - position[dst], wrapped to the minimum image
So `vec` points from the receiving atom (dst) to the sending atom (src).
Only relative vectors are ever passed to the network, never absolute positions.

box: (3, 3) matrix whose rows are the box vectors, or None for no periodic boundaries.
The minimum image is taken in fractional coordinates; this is exact as long as the
cutoff is below half of the smallest box width (checked below).
"""

import torch


def _check_box(box, cutoff):
    # width of the box perpendicular to each pair of box vectors = volume / |a x b|
    a, b, c = box[0], box[1], box[2]
    volume = torch.abs(torch.dot(a, torch.linalg.cross(b, c)))
    widths = volume / torch.stack([
        torch.linalg.norm(torch.linalg.cross(b, c)),
        torch.linalg.norm(torch.linalg.cross(c, a)),
        torch.linalg.norm(torch.linalg.cross(a, b)),
    ])
    if cutoff >= 0.5 * widths.min():
        raise ValueError(f"cutoff {cutoff} must be below half the smallest box width {widths.min().item():.3f}")


def minimum_image(dr, box):
    """Wrap displacement vectors dr (..., 3) into the minimum image of `box` (None: no change)."""
    if box is None:
        return dr
    frac = dr @ torch.linalg.inv(box)
    frac = frac - torch.round(frac)
    return frac @ box


def brute_force(pos, box, cutoff):
    """O(N^2) neighbor list. Fast enough on a GPU for ~1000 atoms."""
    if box is not None:
        _check_box(box, cutoff)
    n = pos.shape[0]
    dr = minimum_image(pos[:, None, :] - pos[None, :, :], box)  # dr[i, j] = pos[i] - pos[j]
    dist = torch.linalg.norm(dr, dim=-1)
    mask = (dist < cutoff) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    src, dst = torch.nonzero(mask, as_tuple=True)
    return src, dst, dr[src, dst]


def cell_list(pos, box, cutoff):
    """O(N) cell-list neighbor list for orthorhombic boxes.

    Atoms are binned into cells at least `cutoff` wide; each atom only checks the 27
    surrounding cells. Needs at least 3 cells along every axis (otherwise a cell would be
    visited twice); falls back to brute force when the box is too small or not periodic.
    """
    if box is None or not torch.allclose(box, torch.diag(torch.diag(box))):
        return brute_force(pos, box, cutoff)
    lengths = torch.diag(box)
    n_cells = torch.floor(lengths / cutoff).long()
    if (n_cells < 3).any():
        return brute_force(pos, box, cutoff)
    _check_box(box, cutoff)

    n = pos.shape[0]
    device = pos.device
    frac = torch.remainder(pos / lengths, 1.0)
    cell_xyz = torch.clamp((frac * n_cells).long(), max=n_cells - 1)  # guard against frac == 1.0 by rounding
    cell_id = (cell_xyz[:, 0] * n_cells[1] + cell_xyz[:, 1]) * n_cells[2] + cell_xyz[:, 2]
    n_total = int(n_cells.prod())

    # Table of atoms per cell, padded with -1: table[cell, slot]
    order = torch.argsort(cell_id)
    counts = torch.bincount(cell_id, minlength=n_total)
    max_count = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(n, device=device) - starts[cell_id[order]]
    table = torch.full((n_total, max_count), -1, dtype=torch.long, device=device)
    table[cell_id[order], slot] = order

    # The 27 neighboring cells of each atom's cell (with periodic wrap)
    shifts = torch.stack(torch.meshgrid(*[torch.arange(-1, 2, device=device)] * 3, indexing="ij"), -1).reshape(-1, 3)
    nb_xyz = torch.remainder(cell_xyz[:, None, :] + shifts[None, :, :], n_cells)
    nb_id = (nb_xyz[..., 0] * n_cells[1] + nb_xyz[..., 1]) * n_cells[2] + nb_xyz[..., 2]  # (n, 27)
    candidates = table[nb_id].reshape(n, -1)  # (n, 27 * max_count)

    dst = torch.arange(n, device=device)[:, None].expand_as(candidates)
    valid = (candidates >= 0) & (candidates != dst)
    src, dst = candidates[valid], dst[valid]
    dr = minimum_image(pos[src] - pos[dst], box)
    keep = torch.linalg.norm(dr, dim=-1) < cutoff
    return src[keep], dst[keep], dr[keep]
