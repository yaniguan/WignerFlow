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


def brute_force(pos, box, cutoff, *, _validate=True):
    """O(N^2) neighbor list. Fast enough on a GPU for ~1000 atoms."""
    if box is not None and _validate:
        _check_box(box, cutoff)
    n = pos.shape[0]
    dr = minimum_image(pos[:, None, :] - pos[None, :, :], box)  # dr[i, j] = pos[i] - pos[j]
    dist = torch.linalg.norm(dr, dim=-1)
    mask = (dist < cutoff) & ~torch.eye(n, dtype=torch.bool, device=pos.device)
    src, dst = torch.nonzero(mask, as_tuple=True)
    return src, dst, dr[src, dst]


def cell_list(pos, box, cutoff, *, _validate=True):
    """Vectorised sparse cell search, including no-PBC and fractional triclinic PBC.

    Sorting costs O(N log N); candidate work is O(N + E) at bounded density.
    No loops over atoms or dense tables proportional to empty volume are used.
    Small periodic boxes use the reference pair search to avoid duplicate images.
    Uses the same fractional minimum-image convention as brute_force.
    """
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    n, device = pos.shape[0], pos.device
    if n == 0:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty, pos[:0]
    if box is None:
        xyz = torch.floor((pos - pos.amin(0)) / cutoff).long() + 1
        dims = xyz.amax(0) + 2
    else:
        if _validate:
            _check_box(box, cutoff)
        inv = torch.linalg.inv(box)
        widths = 1 / torch.linalg.norm(inv, dim=0)
        dims = torch.floor(widths / cutoff).long()
        if bool((dims < 3).any()):
            return brute_force(pos, box, cutoff, _validate=_validate)
        xyz = torch.floor(torch.remainder(pos @ inv, 1) * dims).long()
        xyz = torch.minimum(xyz, dims - 1)
    shifts = torch.cartesian_prod(*[torch.arange(-1, 2, device=device)] * 3)
    nb = xyz[:, None] + shifts
    if box is not None:
        nb = torch.remainder(nb, dims)
    def ids(x):
        return (x[..., 0] * dims[1] + x[..., 1]) * dims[2] + x[..., 2]
    sorted_ids, order = torch.sort(ids(xyz))
    queries = ids(nb).reshape(-1)
    lo = torch.searchsorted(sorted_ids, queries)
    hi = torch.searchsorted(sorted_ids, queries, right=True)
    counts = hi - lo
    query = torch.repeat_interleave(torch.arange(queries.numel(), device=device), counts)
    starts = torch.cumsum(counts, 0) - counts
    offset = torch.arange(query.numel(), device=device) - starts[query]
    src = order[lo[query] + offset]
    dst = query // 27
    dr = minimum_image(pos[src] - pos[dst], box)
    keep = (src != dst) & (dr.square().sum(-1) < cutoff ** 2)
    return src[keep], dst[keep], dr[keep]


class VerletList:
    """Reusable directed pairs at cutoff+skin; displacement checked on every call.

    rebuild_every optionally imposes an additional maximum age (in calls).
    Box changes and shape/device/dtype changes always invalidate the cache.
    Reference displacement is unwrapped, conservatively detecting box crossings.
    """
    def __init__(self, cutoff, skin=1.0, rebuild_every=None):
        if cutoff <= 0 or skin < 0 or (rebuild_every is not None and rebuild_every < 1):
            raise ValueError("invalid cutoff, skin or rebuild interval")
        self.cutoff, self.skin, self.rebuild_every = cutoff, skin, rebuild_every
        self.reference = None
        self.box = None
        self.age = 0
        self.rebuilds = 0

    def __call__(self, pos, box=None):
        if box is not None:
            _check_box(box, self.cutoff)
        changed = (self.reference is None or self.reference.shape != pos.shape
                   or self.reference.device != pos.device or self.reference.dtype != pos.dtype)
        changed = changed or ((box is None) != (self.box is None))
        if not changed and box is not None:
            changed = not torch.equal(box, self.box)
        if not changed:
            changed = bool(((pos - self.reference).square().sum(-1) > (self.skin / 2) ** 2).any())
        if self.rebuild_every is not None and self.age >= self.rebuild_every:
            changed = True
        if changed:
            self.src, self.dst, _ = cell_list(pos, box, self.cutoff + self.skin, _validate=False)
            self.reference = pos.detach().clone()
            self.box = None if box is None else box.detach().clone()
            self.age = 0
            self.rebuilds += 1
        self.age += 1
        dr = minimum_image(pos[self.src] - pos[self.dst], box)
        keep = dr.square().sum(-1) < self.cutoff ** 2
        return self.src[keep], self.dst[keep], dr[keep]
