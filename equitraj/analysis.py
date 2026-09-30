"""Analysis of trajectories (ground truth and model rollouts).

All inputs are in Å and fs (convert with equitraj.units first). Energies in kJ/mol.
Only orthorhombic boxes are supported for now (argon and Li+/EC are cubic).
"""

import numpy as np

from equitraj.units import KB, per_fs_to_wavenumber


def box_lengths(box):
    """Diagonal of a (3, 3) box matrix. Fails for non-orthorhombic boxes."""
    box = np.asarray(box)
    assert np.allclose(box - np.diag(np.diag(box)), 0.0, atol=1e-6), "only orthorhombic boxes are supported"
    return np.diag(box)


def minimum_image(dr, lengths):
    """Wrap displacement vectors into [-L/2, L/2). lengths=None means no periodic box."""
    if lengths is None:
        return dr
    return dr - lengths * np.round(dr / lengths)


# ---------------------------------------------------------------- structure

def rdf(positions, box, idx_a, idx_b, r_max, n_bins=200):
    """Radial distribution function g_ab(r).

    positions: (n_frames, n_atoms, 3) in Å. box: (n_frames, 3, 3) in Å, or None for no PBC.
    For a periodic box, r_max must be at most half the box length.
    Returns (r, g, rho_b) where rho_b is the number density of species b in 1/Å^3.
    """
    idx_a, idx_b = np.asarray(idx_a), np.asarray(idx_b)
    same = np.array_equal(idx_a, idx_b)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    hist = np.zeros(n_bins)
    volume = 0.0
    for t in range(len(positions)):
        lengths = None if box is None else box_lengths(box[t])
        dr = positions[t, idx_a][:, None, :] - positions[t, idx_b][None, :, :]
        d = np.linalg.norm(minimum_image(dr, lengths), axis=-1)
        if same:
            d = d[~np.eye(len(idx_a), dtype=bool)]  # drop i == j
        hist += np.histogram(d, bins=edges)[0]
        volume += np.prod(lengths) if lengths is not None else np.nan
    n_frames = len(positions)
    r = 0.5 * (edges[1:] + edges[:-1])
    if box is None:
        # No volume in vacuum, so no g(r): return the pair-distance density p(r), which integrates to 1.
        return r, hist / (hist.sum() * (edges[1] - edges[0])), np.nan
    volume /= n_frames
    n_b = len(idx_b) - 1 if same else len(idx_b)
    rho_b = n_b / volume
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    g = hist / (n_frames * len(idx_a) * rho_b * shell)
    return r, g, rho_b


def first_peak(r, g):
    """Position and height of the highest peak, refined with a parabola through 3 points."""
    i = int(np.argmax(g))
    if 0 < i < len(g) - 1:
        y0, y1, y2 = g[i - 1], g[i], g[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
        dr = r[1] - r[0]
        return r[i] + shift * dr, y1 - 0.25 * (y0 - y2) * shift
    return r[i], g[i]


def first_minimum_after_peak(r, g):
    """Position of the first minimum after the highest peak (used as the shell cutoff)."""
    i = int(np.argmax(g))
    j = i + 1
    while j < len(g) - 1 and not (g[j] <= g[j - 1] and g[j] <= g[j + 1]):
        j += 1
    return r[j]


def coordination_number(r, g, rho_b, r_cut):
    """n(r_cut) = 4 pi rho_b * integral_0^r_cut g(r) r^2 dr."""
    mask = r <= r_cut
    dr = r[1] - r[0]
    return 4.0 * np.pi * rho_b * np.sum(g[mask] * r[mask] ** 2) * dr


def rdf_l1(r, g1, g2):
    """L1 distance between two RDFs on the same grid (units of Å)."""
    return np.sum(np.abs(g1 - g2)) * (r[1] - r[0])


# ---------------------------------------------------------------- dynamics

def vacf(velocities, max_lag, masses=None):
    """Velocity autocorrelation <v(0).v(t)>, averaged over atoms and time origins.

    velocities: (n_times, n_atoms, 3). With masses given, each atom is weighted by its mass.
    Returns the unnormalized VACF (Å^2/fs^2, mass-weighted if masses given).
    """
    v = velocities if masses is None else velocities * np.sqrt(masses)[None, :, None]
    n, n_atoms, _ = v.shape
    f = np.fft.rfft(v.reshape(n, -1), n=2 * n, axis=0)
    ac = np.fft.irfft(f * np.conj(f), axis=0)[:max_lag]
    ac = ac.sum(axis=1) / n_atoms  # sum over xyz, mean over atoms -> <v.v>
    return ac / (n - np.arange(max_lag))


def vdos(vacf_values, dt_fs):
    """Vibrational density of states from a VACF sampled every dt_fs.

    Uses a Hann window on the one-sided VACF. Returns (wavenumber in cm^-1, spectrum normalized to max 1).
    The highest resolvable wavenumber is the Nyquist limit 1/(2 dt).
    """
    c = vacf_values / vacf_values[0]
    window = np.hanning(2 * len(c))[len(c):]
    spec = np.abs(np.fft.rfft(c * window, n=4 * len(c)))
    freq = np.fft.rfftfreq(4 * len(c), d=dt_fs)  # 1/fs
    return per_fs_to_wavenumber(freq), spec / spec.max()


def msd(positions, lags):
    """Mean squared displacement for the given lags (in frames), averaged over atoms and origins.

    positions must be unwrapped: (n_times, n_atoms, 3).
    """
    out = np.zeros(len(lags))
    for k, lag in enumerate(lags):
        d = positions[lag:] - positions[:-lag] if lag > 0 else np.zeros_like(positions)
        out[k] = np.mean(np.sum(d**2, axis=-1))
    return out


def diffusion_coefficient(t_fs, msd_values, t_min_fs, t_max_fs):
    """Einstein relation D = slope / 6, fitted on [t_min, t_max]. Returns D in Å^2/fs and cm^2/s."""
    mask = (t_fs >= t_min_fs) & (t_fs <= t_max_fs)
    slope = np.polyfit(t_fs[mask], msd_values[mask], 1)[0]
    d = slope / 6.0
    return d, d * 0.1  # 1 Å^2/fs = 1e-16 cm^2 / 1e-15 s = 0.1 cm^2/s


# ---------------------------------------------------------------- energy

def dihedral(p0, p1, p2, p3):
    """Dihedral angle in degrees for arrays of points (..., 3)."""
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=-1, keepdims=True)
    v = b0 - np.sum(b0 * b1, -1, keepdims=True) * b1
    w = b2 - np.sum(b2 * b1, -1, keepdims=True) * b1
    x = np.sum(v * w, -1)
    y = np.sum(np.cross(b1, v) * w, -1)
    return np.degrees(np.arctan2(y, x))


def momenta(positions, velocities, masses):
    """Total linear momentum (T, 3) and angular momentum about the center of mass (T, 3), in amu Å/fs and amu Å^2/fs."""
    m = masses[None, :, None]
    p = (m * velocities).sum(axis=1)
    com = (m * positions).sum(axis=1) / masses.sum()
    rel = positions - com[:, None, :]
    l = (m * np.cross(rel, velocities)).sum(axis=1)
    return p, l


def n_dof(n_atoms, n_constraints, com_removed=True):
    return 3 * n_atoms - n_constraints - (3 if com_removed else 0)


def temperature(kinetic_energy, dof):
    return 2.0 * np.asarray(kinetic_energy) / (dof * KB)


def energy_drift(time_ps, total_energy, dof):
    """Linear drift of the total energy in kJ/mol per ns per degree of freedom."""
    slope_per_ps = np.polyfit(time_ps, total_energy, 1)[0]
    return slope_per_ps * 1000.0 / dof
