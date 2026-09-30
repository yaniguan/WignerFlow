"""Rotations of equivariant features (Wigner-D matrices), using e3nn's conventions.

Feature layout used everywhere: x has shape (..., S, C) with S = (lmax+1)^2 and the
components ordered l = 0, 1, ..., lmax, and within each l, m = -l, ..., l.
Index of (l, m) is l*l + l + m.

e3nn uses the y axis as the "pole". So in this file the edge frame is the frame where
the edge vector points along +y. Two facts are used by the SO(2) convolution:
  * after rotating into the edge frame, the spherical harmonics of the edge are nonzero
    only at m = 0;
  * a rotation about the y axis only mixes the components (l, m) and (l, -m).
"""

import torch
from e3nn import o3


def n_components(lmax):
    return (lmax + 1) ** 2


def index(l, m):
    return l * l + l + m


def edge_angles(vec):
    """Euler angles (alpha, beta) such that R(alpha, beta, 0) @ e_y = vec / |vec| (same as e3nn.o3.xyz_to_angles).

    beta uses atan2(sqrt(x^2 + z^2), y) instead of acos(y): acos loses precision when the edge is
    close to the y axis (error ~ sqrt(float eps), about 3e-4 in float32), which breaks equivariance.
    The tiny constant keeps the gradient finite when an edge points exactly along y.
    """
    x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
    alpha = torch.atan2(x, z)
    beta = torch.atan2(torch.sqrt(x * x + z * z + 1e-30), y)
    return alpha, beta


def rotation_about_y(l, theta):
    """Wigner-D of a rotation by theta about the y axis (the pole), written out explicitly.

    It only couples (l, -m) and (l, +m): cos(m theta) on the diagonal, +-sin(m theta) off it.
    theta: (E,) -> (E, 2l+1, 2l+1).
    """
    d = torch.zeros(theta.shape[0], 2 * l + 1, 2 * l + 1, dtype=theta.dtype, device=theta.device)
    d[:, l, l] = 1.0
    for m in range(1, l + 1):
        c, s = torch.cos(m * theta), torch.sin(m * theta)
        lo, hi = l - m, l + m
        d[:, lo, lo] = c
        d[:, lo, hi] = s
        d[:, hi, lo] = -s
        d[:, hi, hi] = c
    return d


_Y_TO_X = {}


def y_to_x(l, dtype, device):
    """Wigner-D of the fixed rotation that maps the y axis onto the x axis (-90 deg about z). Cached.

    A rotation about x by beta equals this matrix times a rotation about y by beta times its transpose.
    """
    key = (l, dtype, device)
    if key not in _Y_TO_X:
        m = torch.tensor([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)
        irrep = o3.Irrep(l, (-1) ** l)
        _Y_TO_X[key] = irrep.D_from_matrix(m).to(dtype=dtype, device=device)
    return _Y_TO_X[key]


def edge_wigner(vec, lmax):
    """Block-diagonal Wigner-D matrices D (E, S, S) for the rotation R(alpha, beta, 0) = Ry(alpha) Rx(beta) of each edge.

    to_edge_frame(x) = D^T x  puts the edge along +y;  from_edge_frame(y) = D y  undoes it.
    Built from explicit rotations about y (no matrix exponentials), so it is fast on GPU.
    """
    alpha, beta = edge_angles(vec)
    s = n_components(lmax)
    d = torch.zeros(vec.shape[0], s, s, dtype=vec.dtype, device=vec.device)
    d[:, 0, 0] = 1.0  # l = 0 is invariant
    for l in range(1, lmax + 1):
        a, b = l * l, (l + 1) * (l + 1)
        j = y_to_x(l, vec.dtype, vec.device)
        rx = j @ rotation_about_y(l, beta) @ j.T
        d[:, a:b, a:b] = rotation_about_y(l, alpha) @ rx
    return d


def to_edge_frame(d, x):
    """x: (E, S, C) in the global frame -> edge frame."""
    return torch.einsum("eij,eic->ejc", d, x)


def from_edge_frame(d, y):
    """y: (E, S, C) in the edge frame -> global frame."""
    return torch.einsum("eij,ejc->eic", d, y)


def rotate(x, rotation, lmax):
    """Apply a 3x3 rotation or reflection matrix to features x (..., S, C) with natural parity (-1)^l.

    Used only in tests. In e3nn (>= 0.5) the l=1 components are ordered (x, y, z), so an
    xyz vector can be used directly as an l=1 feature, and D for l=1 equals the matrix itself.
    """
    irreps = o3.Irreps([(1, (l, (-1) ** l)) for l in range(lmax + 1)])
    d = irreps.D_from_matrix(rotation.to(x.dtype))
    return torch.einsum("ij,...jc->...ic", d, x)
