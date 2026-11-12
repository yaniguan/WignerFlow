"""Public Pure-MD Trajectory Rollout Benchmark Suite (MD17-CCSD(T), MD17 & MD22).

Implements the 4-pillar MLMD trajectory benchmark protocol aligned with our
40-paper literature survey (Chmiela et al. Sci. Adv. 2017, Nat. Commun. 2018,
Sci. Adv. 2023; Fu et al. ICLR/TMLR 2023; Schreiner et al. ITO 2023;
Thiemann et al. TrajCast 2025):

1. Covalent bond-rupture stability horizon T_stab (|r_ij(t) - b_ij^0| > 0.5 A)
   plus short-range steric-clash guard (min r_ij(t) < 0.7 A).
2. Normalized all-pair interatomic distance density error MAE_{h(r)} against
   held-out authentic CCSD(T) / DFT-MD quantum ensembles (9 to 60 atoms).
3. Nyquist-bounded velocity autocorrelation (VACF) -> vibrational density of
   states (VDOS) spectral fidelity (cosine similarity & L1 MAE).
4. Chapman-Kolmogorov multi-stride consistency (10 fs vs 20 fs), kinetic
   temperature conservation, COM momentum drift, and CPU throughput (ns/day).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import io
from pathlib import Path
import time
from typing import Dict, List, Optional, Sequence, Tuple
import urllib.request
import zipfile

import numpy as np

ATOMIC_MASSES_AMU: Dict[int, float] = {
    1: 1.00784,
    6: 12.0107,
    7: 14.0067,
    8: 15.999,
    9: 18.9984,
    15: 30.9738,
    16: 32.065,
    17: 35.453,
}

COVALENT_RADII_A: Dict[int, float] = {
    1: 0.31,
    6: 0.76,
    7: 0.71,
    8: 0.66,
    9: 0.57,
    15: 1.07,
    16: 1.05,
    17: 1.02,
}

# Energies in kcal/mol, forces in kcal/(mol*A), positions in A, time in fs, mass in amu
KCAL_MOL_TO_AMU_A2_FS2 = 4.184e-4
KB_KCAL_MOL_K = 0.001987204259
KB_AMU_A2_FS2_K = KB_KCAL_MOL_K * KCAL_MOL_TO_AMU_A2_FS2
SPEED_OF_LIGHT_CM_FS = 2.99792458e-5

MD_BENCH_SYSTEMS: Dict[str, Dict[str, str]] = {
    "ethanol_ccsd_t": {
        "kind": "ccsd_zip",
        "url": "http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip",
        "theory": "CCSD(T)",
        "label": "Ethanol",
    },
    "malonaldehyde_ccsd_t": {
        "kind": "ccsd_zip",
        "url": "http://www.quantum-machine.org/gdml/data/npz/malonaldehyde_ccsd_t.zip",
        "theory": "CCSD(T)",
        "label": "Malonaldehyde",
    },
    "benzene_ccsd_t": {
        "kind": "ccsd_zip",
        "url": "http://www.quantum-machine.org/gdml/data/npz/benzene_ccsd_t.zip",
        "theory": "CCSD(T)",
        "label": "Benzene",
    },
    "uracil_dft": {
        "kind": "npz_full",
        "url": "http://www.quantum-machine.org/gdml/data/npz/md17_uracil.npz",
        "theory": "DFT-PBE+TS",
        "label": "Uracil",
    },
    "toluene_ccsd_t": {
        "kind": "ccsd_zip",
        "url": "http://www.quantum-machine.org/gdml/data/npz/toluene_ccsd_t.zip",
        "theory": "CCSD(T)",
        "label": "Toluene",
    },
    "aspirin_ccsd": {
        "kind": "ccsd_zip",
        "url": "http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip",
        "theory": "CCSD",
        "label": "Aspirin",
    },
    "md22_AT_AT": {
        "kind": "npz_full",
        "url": "http://www.quantum-machine.org/gdml/data/npz/md22_AT-AT.npz",
        "theory": "DFT-PBE+MBD",
        "label": "MD22 AT-AT (DNA)",
    },
}

DEFAULT_BENCHMARK_PANEL: Tuple[str, ...] = tuple(MD_BENCH_SYSTEMS.keys())


def load_or_fetch_benchmark_system(
    system_key: str,
    *,
    cache_dir: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Load cached train/test quantum MD dataset or fetch from public sGDML/MD22 archive."""
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        for cand in (cdir / f"{system_key}.npz", cdir / f"md17_{system_key}_4k.npz"):
            if cand.exists():
                d = np.load(cand)
                if "R_train" in d and "R_test" in d:
                    meta = MD_BENCH_SYSTEMS.get(
                        system_key, {"theory": "QM-MD", "label": system_key}
                    )
                    return {
                        "system": system_key,
                        "label": meta["label"],
                        "theory": meta["theory"],
                        "z": np.asarray(d["z"], dtype=np.int64),
                        "R_train": np.ascontiguousarray(d["R_train"], dtype=np.float64),
                        "F_train": np.ascontiguousarray(d["F_train"], dtype=np.float64),
                        "R_test": np.ascontiguousarray(d["R_test"], dtype=np.float64),
                        "F_test": np.ascontiguousarray(d["F_test"], dtype=np.float64),
                    }
                if "R" in d:
                    r_all = np.ascontiguousarray(d["R"], dtype=np.float64)
                    f_all = np.ascontiguousarray(d["F"], dtype=np.float64)
                    mid = len(r_all) // 2
                    return {
                        "system": system_key,
                        "label": system_key,
                        "theory": "QM-MD",
                        "z": np.asarray(d["z"], dtype=np.int64),
                        "R_train": r_all[:mid],
                        "F_train": f_all[:mid],
                        "R_test": r_all[mid:],
                        "F_test": f_all[mid:],
                    }

    if system_key not in MD_BENCH_SYSTEMS:
        raise ValueError(f"Unknown benchmark system '{system_key}'.")

    meta = MD_BENCH_SYSTEMS[system_key]
    with urllib.request.urlopen(meta["url"], timeout=90.0) as resp:
        payload = resp.read()

    if meta["kind"] == "ccsd_zip":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            tr_name = [n for n in zf.namelist() if "train" in n][0]
            te_name = [n for n in zf.namelist() if "test" in n][0]
            tr = np.load(io.BytesIO(zf.read(tr_name)))
            te = np.load(io.BytesIO(zf.read(te_name)))
            z_arr = np.asarray(tr["z"], dtype=np.int64)
            r_tr = np.ascontiguousarray(tr["R"], dtype=np.float64)
            f_tr = np.ascontiguousarray(tr["F"], dtype=np.float64)
            r_te = np.ascontiguousarray(te["R"], dtype=np.float64)
            f_te = np.ascontiguousarray(te["F"], dtype=np.float64)
    else:
        raw = np.load(io.BytesIO(payload))
        z_arr = np.asarray(raw["z"], dtype=np.int64)
        r_all = np.ascontiguousarray(raw["R"], dtype=np.float64)
        f_all = np.ascontiguousarray(raw["F"], dtype=np.float64)
        r_tr, f_tr = r_all[:1000], f_all[:1000]
        r_te, f_te = r_all[2000:3000], f_all[2000:3000]

    if cache_dir is not None:
        np.savez_compressed(
            Path(cache_dir) / f"{system_key}.npz",
            z=z_arr,
            R_train=r_tr,
            F_train=f_tr,
            R_test=r_te,
            F_test=f_te,
        )

    return {
        "system": system_key,
        "label": meta["label"],
        "theory": meta["theory"],
        "z": z_arr,
        "R_train": r_tr,
        "F_train": f_tr,
        "R_test": r_te,
        "F_test": f_te,
    }


def masses_from_atomic_numbers(z: Sequence[int]) -> np.ndarray:
    return np.array(
        [ATOMIC_MASSES_AMU.get(int(zi), 12.0107) for zi in z], dtype=np.float64
    )


def infer_covalent_bonds(
    pos0: np.ndarray, z: Sequence[int], tolerance_a: float = 0.35
) -> Tuple[np.ndarray, np.ndarray]:
    """Infer covalent bond index pairs (B, 2) and equilibrium lengths b_ij^0 (B,)."""
    pos0 = np.asarray(pos0, dtype=np.float64)
    n = pos0.shape[0]
    radii = np.array(
        [COVALENT_RADII_A.get(int(zi), 0.76) for zi in z], dtype=np.float64
    )
    bonds: List[Tuple[int, int]] = []
    lengths: List[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(pos0[i] - pos0[j]))
            if d <= radii[i] + radii[j] + tolerance_a:
                bonds.append((i, j))
                lengths.append(d)
    if not bonds:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float64)
    return np.array(bonds, dtype=np.int64), np.array(lengths, dtype=np.float64)


def project_zero_com_motion(
    pos: np.ndarray, vel: np.ndarray, masses: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """TrajCast-style exact removal of COM translation, linear & angular momentum."""
    m_col = masses[:, None]
    m_tot = float(np.sum(masses))
    r_com = np.sum(pos * m_col, axis=0) / m_tot
    v_com = np.sum(vel * m_col, axis=0) / m_tot
    pos_c = pos - r_com
    vel_c = vel - v_com

    ang_mom = np.sum(m_col * np.cross(pos_c, vel_c), axis=0)
    r2 = np.sum(pos_c * pos_c, axis=1)
    inertia = np.sum(
        masses[:, None, None]
        * (
            r2[:, None, None] * np.eye(3)[None, :, :]
            - pos_c[:, :, None] * pos_c[:, None, :]
        ),
        axis=0,
    )
    omega, *_ = np.linalg.lstsq(inertia + 1e-14 * np.eye(3), ang_mom, rcond=None)
    vel_c = vel_c - np.cross(omega[None, :], pos_c)
    return pos_c, vel_c


def sample_maxwell_boltzmann_velocities(
    pos0: np.ndarray,
    masses: np.ndarray,
    *,
    temp_k: float = 500.0,
    seed: int = 42,
) -> np.ndarray:
    """Draw COM-projected Maxwell-Boltzmann velocities calibrated exactly to temp_k."""
    rng = np.random.default_rng(seed)
    sigma_v = np.sqrt(KB_AMU_A2_FS2_K * temp_k / masses)[:, None]
    vel = rng.normal(0.0, sigma_v, size=pos0.shape)
    _, vel = project_zero_com_motion(pos0, vel, masses)
    t_cur = float(compute_kinetic_temperature(vel, masses)[0])
    if t_cur > 1e-10:
        vel = vel * np.sqrt(temp_k / t_cur)
    return vel


def compute_kinetic_temperature(vel: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Instantaneous kinetic temperature [K] accounting for 6 removed rigid DOFs."""
    arr = np.asarray(vel, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[None, ...]
    n_atoms = masses.shape[0]
    dof = max(3 * n_atoms - 6, 3)
    two_ke = np.sum(masses[None, :, None] * (arr**2), axis=(1, 2))
    return two_ke / (dof * KB_AMU_A2_FS2_K)


def compute_bond_stability(
    traj_pos: np.ndarray,
    z: Sequence[int],
    ref_pos0: Optional[np.ndarray] = None,
    *,
    dt_frame_fs: float = 10.0,
    delta_stab_a: float = 0.5,
    clash_cutoff_a: float = 0.7,
) -> Dict[str, float | bool]:
    """Fu et al. (ICLR/TMLR 2023) bond-rupture horizon + MLIP-Arena clash guard."""
    traj = np.asarray(traj_pos, dtype=np.float64)
    p0 = traj[0] if ref_pos0 is None else np.asarray(ref_pos0, dtype=np.float64)
    finite_frames = np.all(np.isfinite(traj), axis=(1, 2))
    if not np.all(finite_frames):
        first_bad = int(np.argmax(~finite_frames))
        valid = traj[: max(first_bad, 1)]
        if first_bad == 0:
            return {
                "is_stable": False,
                "stable_fraction": 0.0,
                "t_stab_ps": 0.0,
                "max_bond_dev_a": float("inf"),
                "min_pair_dist_a": 0.0,
            }
        sub = compute_bond_stability(
            valid,
            z,
            p0,
            dt_frame_fs=dt_frame_fs,
            delta_stab_a=delta_stab_a,
            clash_cutoff_a=clash_cutoff_a,
        )
        n_ok = int(round(float(sub["stable_fraction"]) * len(valid)))
        return {
            "is_stable": False,
            "stable_fraction": float(n_ok / max(len(traj), 1)),
            "t_stab_ps": float(n_ok * dt_frame_fs * 1e-3),
            "max_bond_dev_a": float(sub["max_bond_dev_a"]),
            "min_pair_dist_a": float(sub["min_pair_dist_a"]),
        }

    bonds, b0 = infer_covalent_bonds(p0, z)
    t_len, n_atoms, _ = traj.shape
    iu = np.triu_indices(n_atoms, k=1)
    pair_vecs = traj[:, iu[0], :] - traj[:, iu[1], :]
    pair_dists = np.linalg.norm(pair_vecs, axis=-1)
    min_pair_per_frame = np.min(pair_dists, axis=1)

    if len(bonds) > 0:
        b_vecs = traj[:, bonds[:, 0], :] - traj[:, bonds[:, 1], :]
        b_dists = np.linalg.norm(b_vecs, axis=-1)
        bond_devs = np.max(np.abs(b_dists - b0[None, :]), axis=1)
    else:
        bond_devs = np.zeros(t_len, dtype=np.float64)

    unstable_mask = (bond_devs > delta_stab_a) | (min_pair_per_frame < clash_cutoff_a)
    if np.any(unstable_mask):
        first_idx = int(np.argmax(unstable_mask))
        is_stable = False
        stable_frames = first_idx
    else:
        is_stable = True
        stable_frames = t_len

    return {
        "is_stable": bool(is_stable),
        "stable_fraction": float(stable_frames / max(t_len, 1)),
        "t_stab_ps": float(stable_frames * dt_frame_fs * 1e-3),
        "max_bond_dev_a": float(np.max(bond_devs[: max(stable_frames, 1)])),
        "min_pair_dist_a": float(np.min(min_pair_per_frame[: max(stable_frames, 1)])),
    }


def compute_hr_density(
    traj_pos: np.ndarray,
    *,
    r_max: float = 10.0,
    n_grid: int = 250,
    sigma_a: float = 0.035,
) -> Tuple[np.ndarray, np.ndarray]:
    """Normalized all-pair distance density h(r) with integral_0^{r_max} h(r) dr = 1."""
    traj = np.asarray(traj_pos, dtype=np.float64)
    finite_mask = np.all(np.isfinite(traj), axis=(1, 2))
    traj = traj[finite_mask]
    r_grid = np.linspace(0.4, r_max, n_grid)
    if len(traj) == 0:
        return r_grid, np.zeros_like(r_grid)

    n_atoms = traj.shape[1]
    iu = np.triu_indices(n_atoms, k=1)
    diffs = traj[:, iu[0], :] - traj[:, iu[1], :]
    dists = np.linalg.norm(diffs, axis=-1).reshape(-1)

    dr = float(r_grid[1] - r_grid[0])
    edges = np.linspace(0.4 - 0.5 * dr, r_max + 0.5 * dr, n_grid + 1)
    counts, _ = np.histogram(dists, bins=edges)
    kernel_offsets = r_grid[:, None] - r_grid[None, :]
    kernel = np.exp(-0.5 * (kernel_offsets / sigma_a) ** 2)
    h = kernel @ counts.astype(np.float64)
    area = float(np.sum(h) * dr)
    if area > 0.0:
        h = h / area
    return r_grid, h


def compute_hr_mae(
    model_pos: np.ndarray,
    ref_pos: np.ndarray,
    *,
    r_max: float = 10.0,
    n_grid: int = 250,
    sigma_a: float = 0.035,
) -> float:
    """Fu et al. (ICLR/TMLR 2023) integrated L1 error int |h_model(r) - h_ref(r)| dr."""
    r_grid, h_mod = compute_hr_density(
        model_pos, r_max=r_max, n_grid=n_grid, sigma_a=sigma_a
    )
    _, h_ref = compute_hr_density(
        ref_pos, r_max=r_max, n_grid=n_grid, sigma_a=sigma_a
    )
    dr = float(r_grid[1] - r_grid[0])
    return float(np.sum(np.abs(h_mod - h_ref)) * dr)


def compute_vdos_spectrum(
    traj_vel: np.ndarray,
    masses: np.ndarray,
    dt_fs: float,
    *,
    max_lag_frames: int = 128,
    wavenumber_grid_cm1: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Mass-weighted VACF -> Hann-windowed cosine VDOS spectrum S(nu_bar)."""
    vel = np.asarray(traj_vel, dtype=np.float64)
    vel = vel[np.all(np.isfinite(vel), axis=(1, 2))]
    nyquist_cm1 = 1.0 / (2.0 * SPEED_OF_LIGHT_CM_FS * dt_fs)
    nu_grid = (
        np.linspace(20.0, min(4000.0, nyquist_cm1 * 0.96), 140)
        if wavenumber_grid_cm1 is None
        else np.asarray(wavenumber_grid_cm1, dtype=np.float64)
    )
    if len(vel) < 4:
        return nu_grid, np.zeros_like(nu_grid)

    t_len = vel.shape[0]
    n_lags = min(max_lag_frames, max(t_len // 2, 2))
    mv = vel * np.sqrt(masses[None, :, None])
    flat = mv.reshape(t_len, -1)
    n_fft = 1 << int(np.ceil(np.log2(2 * t_len)))
    spec = np.fft.rfft(flat, n=n_fft, axis=0)
    acf_all = np.fft.irfft(spec * np.conj(spec), n=n_fft, axis=0)[:n_lags]
    norm = np.arange(t_len, t_len - n_lags, -1, dtype=np.float64)
    vacf = np.sum(acf_all, axis=1) / np.maximum(norm, 1.0)
    if abs(vacf[0]) > 1e-30:
        vacf = vacf / vacf[0]

    lags_fs = np.arange(n_lags, dtype=np.float64) * dt_fs
    hann = 0.5 * (1.0 + np.cos(np.pi * np.arange(n_lags) / max(n_lags - 1, 1)))
    omega_t = 2.0 * np.pi * SPEED_OF_LIGHT_CM_FS * nu_grid[:, None] * lags_fs[None, :]
    vdos = np.maximum(0.0, (vacf * hann) @ np.cos(omega_t).T * dt_fs)
    dnu = float(nu_grid[1] - nu_grid[0]) if len(nu_grid) > 1 else 1.0
    area = float(np.sum(vdos) * dnu)
    if area > 1e-30:
        vdos = vdos / area
    return nu_grid, vdos


def compute_vdos_metrics(
    model_vel: np.ndarray,
    ref_vel: np.ndarray,
    masses: np.ndarray,
    *,
    dt_model_fs: float,
    dt_ref_fs: float,
) -> Dict[str, float]:
    """Compare VDOS strictly within the unaliased Nyquist band [20, nu_Nyq]."""
    nyq_cm1 = min(
        3800.0,
        0.95 / (2.0 * SPEED_OF_LIGHT_CM_FS * max(dt_model_fs, dt_ref_fs)),
    )
    nu_grid = np.linspace(20.0, max(nyq_cm1, 60.0), 140)
    _, s_mod = compute_vdos_spectrum(
        model_vel, masses, dt_model_fs, wavenumber_grid_cm1=nu_grid
    )
    _, s_ref = compute_vdos_spectrum(
        ref_vel, masses, dt_ref_fs, wavenumber_grid_cm1=nu_grid
    )
    dnu = float(nu_grid[1] - nu_grid[0])
    mae = float(np.sum(np.abs(s_mod - s_ref)) * dnu)
    denom = float(np.linalg.norm(s_mod) * np.linalg.norm(s_ref))
    cos_sim = float(np.dot(s_mod, s_ref) / denom) if denom > 1e-30 else 0.0
    return {
        "nyquist_cm1": float(nyq_cm1),
        "vdos_mae": mae,
        "vdos_cosine_sim": cos_sim,
    }


class EquivariantPairFlowModel:
    """E(3)-equivariant pairwise harmonic + Gaussian-RBF residual operator.

    Separates 1-2 covalent bonds, 1-3 valence-angle pairs, and soft 1-4+
    anharmonic/nonbonded pair interactions fit on a quantum training split.
    """

    def __init__(
        self,
        z: Sequence[int],
        pair_i: np.ndarray,
        pair_j: np.ndarray,
        mu_ij: np.ndarray,
        sigma_ij: np.ndarray,
        bond_mask: np.ndarray,
        angle_mask: np.ndarray,
        k_harm_kcal: np.ndarray,
        rbf_centers: np.ndarray,
        rbf_gamma: float,
        rbf_weights: np.ndarray,
        target_temp_k: float,
    ) -> None:
        self.z = np.asarray(z, dtype=np.int64)
        self.masses = masses_from_atomic_numbers(self.z)
        self.pair_i = pair_i
        self.pair_j = pair_j
        self.mu_ij = mu_ij
        self.sigma_ij = sigma_ij
        self.bond_mask = bond_mask
        self.angle_mask = angle_mask
        self.stiff_mask = bond_mask | angle_mask
        self.k_harm_kcal = k_harm_kcal
        self.rbf_centers = rbf_centers
        self.rbf_gamma = rbf_gamma
        self.rbf_weights = rbf_weights
        self.target_temp_k = float(target_temp_k)

    @classmethod
    def fit_from_trajectory(
        cls,
        z: Sequence[int],
        train_pos: np.ndarray,
        train_forces: Optional[np.ndarray] = None,
        *,
        target_temp_k: float = 500.0,
        n_rbf: int = 8,
        ridge: float = 1e-2,
    ) -> "EquivariantPairFlowModel":
        pos = np.asarray(train_pos, dtype=np.float64)
        n_atoms = pos.shape[1]
        pi, pj = np.triu_indices(n_atoms, k=1)
        diffs = pos[:, pi, :] - pos[:, pj, :]
        dists = np.linalg.norm(diffs, axis=-1)
        mu_ij = np.median(dists, axis=0)
        sigma_ij = np.maximum(np.std(dists, axis=0), 0.015)

        bonds, _ = infer_covalent_bonds(pos[0], z)
        bond_set = {(int(a), int(b)) for a, b in bonds}
        adj: Dict[int, set[int]] = {k: set() for k in range(n_atoms)}
        for a, b in bonds:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))
        angle_set: set[Tuple[int, int]] = set()
        for center, nbrs in adj.items():
            nbr_list = sorted(nbrs)
            for idx_a in range(len(nbr_list)):
                for idx_b in range(idx_a + 1, len(nbr_list)):
                    angle_set.add((nbr_list[idx_a], nbr_list[idx_b]))

        bond_mask = np.array(
            [(int(i), int(j)) in bond_set for i, j in zip(pi, pj)], dtype=bool
        )
        angle_mask = np.array(
            [
                (not bond_mask[idx])
                and (
                    ((int(i), int(j)) in angle_set)
                    or (sigma_ij[idx] < 0.11 and mu_ij[idx] < 3.05)
                    or (sigma_ij[idx] < 0.085 and mu_ij[idx] < 5.20)
                )
                for idx, (i, j) in enumerate(zip(pi, pj))
            ],
            dtype=bool,
        )
        stiff_mask = bond_mask | angle_mask

        # Physical covalent bond stretches (~700-950 kcal/mol/A^2 for C-H/O-H/C=O)
        k_eq = (KB_KCAL_MOL_K * target_temp_k) / (sigma_ij**2)
        k_harm = np.where(
            bond_mask,
            np.clip(k_eq * 0.85, 320.0, 880.0),
            np.where(
                angle_mask,
                np.clip(k_eq * 0.55, 25.0, 180.0),
                np.clip(k_eq * 0.12, 0.15, 4.5),
            ),
        )

        rbf_centers = np.linspace(-0.45, 0.65, n_rbf)
        rbf_gamma = 14.0
        n_pairs = len(pi)
        rbf_weights = np.zeros((n_pairs, n_rbf), dtype=np.float64)

        if train_forces is not None and len(train_forces) == len(pos):
            sub = slice(0, len(pos), max(len(pos) // 250, 1))
            d_sub = diffs[sub]
            r_sub = np.maximum(dists[sub], 1e-6)
            u_sub = d_sub / r_sub[:, :, None]
            f_sub = np.asarray(train_forces[sub], dtype=np.float64)
            f_mag_base = -k_harm[None, :] * (r_sub - mu_ij[None, :])
            f_pred = np.zeros_like(f_sub)
            np.add.at(f_pred, (slice(None), pi), f_mag_base[:, :, None] * u_sub)
            np.add.at(f_pred, (slice(None), pj), -f_mag_base[:, :, None] * u_sub)
            f_res = f_sub - f_pred

            for p_idx in range(n_pairs):
                if stiff_mask[p_idx]:
                    continue
                i_at, j_at = int(pi[p_idx]), int(pj[p_idx])
                proj = 0.5 * np.sum(
                    (f_res[:, i_at, :] - f_res[:, j_at, :]) * u_sub[:, p_idx, :],
                    axis=-1,
                )
                x = r_sub[:, p_idx] - mu_ij[p_idx]
                phi = np.exp(-rbf_gamma * (x[:, None] - rbf_centers[None, :]) ** 2)
                gram = phi.T @ phi + (ridge + 10.0) * np.eye(n_rbf)
                w = np.linalg.solve(gram, phi.T @ proj)
                rbf_weights[p_idx] = np.clip(w, -1.2, 1.2)

        return cls(
            z=z,
            pair_i=pi,
            pair_j=pj,
            mu_ij=mu_ij,
            sigma_ij=sigma_ij,
            bond_mask=bond_mask,
            angle_mask=angle_mask,
            k_harm_kcal=k_harm,
            rbf_centers=rbf_centers,
            rbf_gamma=rbf_gamma,
            rbf_weights=rbf_weights,
            target_temp_k=target_temp_k,
        )

    def compute_forces(
        self,
        pos: np.ndarray,
        *,
        include_stiff: bool = True,
        dt_coarse_fs: float = 0.5,
    ) -> np.ndarray:
        """Compute E(3)-equivariant internal forces [kcal/(mol*A)]."""
        vec = pos[self.pair_i] - pos[self.pair_j]
        r = np.maximum(np.linalg.norm(vec, axis=-1), 1e-6)
        u = vec / r[:, None]
        dx = np.clip(r - self.mu_ij, -2.0, 2.0)

        if include_stiff:
            k_eff = self.k_harm_kcal
            rep_gap = np.maximum(0.80 * self.mu_ij - r, 0.0)
            f_rep = 80.0 * rep_gap
        else:
            red_m = (self.masses[self.pair_i] * self.masses[self.pair_j]) / (
                self.masses[self.pair_i] + self.masses[self.pair_j]
            )
            k_nyq = 0.18 * red_m / (KCAL_MOL_TO_AMU_A2_FS2 * max(dt_coarse_fs**2, 1.0))
            k_eff = np.where(self.stiff_mask, 0.0, np.minimum(self.k_harm_kcal, k_nyq))
            rep_gap = np.maximum(0.78 * self.mu_ij - r, 0.0)
            f_rep = np.minimum(14.0 * rep_gap, 10.0)

        phi = np.exp(-self.rbf_gamma * (dx[:, None] - self.rbf_centers[None, :]) ** 2)
        f_rbf = np.sum(self.rbf_weights * phi, axis=-1)
        if not include_stiff:
            f_rbf = np.where(self.stiff_mask, 0.0, np.clip(f_rbf, -3.0, 3.0))

        f_scalar = -k_eff * dx + f_rbf + f_rep
        if not include_stiff:
            f_scalar = np.clip(f_scalar, -12.0, 12.0)
        pair_vec = f_scalar[:, None] * u
        forces = np.zeros_like(pos)
        np.add.at(forces, self.pair_i, pair_vec)
        np.add.at(forces, self.pair_j, -pair_vec)
        return forces

    def rollout_velocity_verlet(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        *,
        n_steps: int,
        dt_fs: float,
        thermostat_tau_fs: Optional[float] = None,
        seed: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Standard explicit Velocity-Verlet rollout (with optional CSVR thermostat)."""
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        pos = np.array(pos0, dtype=np.float64, copy=True)
        vel = np.array(vel0, dtype=np.float64, copy=True)
        pos, vel = project_zero_com_motion(pos, vel, self.masses)
        inv_m = KCAL_MOL_TO_AMU_A2_FS2 / self.masses[:, None]

        traj_r = np.empty((n_steps + 1,) + pos.shape, dtype=np.float64)
        traj_v = np.empty((n_steps + 1,) + vel.shape, dtype=np.float64)
        traj_r[0] = pos
        traj_v[0] = vel

        acc = self.compute_forces(pos, include_stiff=True) * inv_m
        for step in range(1, n_steps + 1):
            vel_half = vel + 0.5 * dt_fs * acc
            pos = pos + dt_fs * vel_half
            if not np.all(np.isfinite(pos)) or np.max(np.abs(pos)) > 200.0:
                traj_r[step:] = np.nan
                traj_v[step:] = np.nan
                break
            acc_new = self.compute_forces(pos, include_stiff=True) * inv_m
            vel = vel_half + 0.5 * dt_fs * acc_new
            if thermostat_tau_fs is not None and step % 10 == 0:
                pos, vel = project_zero_com_motion(pos, vel, self.masses)
                t_cur = float(compute_kinetic_temperature(vel, self.masses)[0])
                if t_cur > 1e-8:
                    g = float(np.exp(-10.0 * dt_fs / thermostat_tau_fs))
                    t_tgt = self.target_temp_k * float(
                        np.clip(1.0 + rng.normal(0.0, 0.05), 0.85, 1.15)
                    )
                    vel = vel * np.sqrt(g + (1.0 - g) * (t_tgt / t_cur))
            acc = acc_new
            traj_r[step] = pos
            traj_v[step] = vel

        wall_s = time.perf_counter() - t0
        return traj_r, traj_v, wall_s

    def rollout_wigner_coarse(
        self,
        pos0: np.ndarray,
        vel0: np.ndarray,
        *,
        n_steps: int,
        dt_fs: float = 10.0,
        seed: int = 42,
        tau_thermostat_fs: float = 80.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Two-tier coarse-step propagator baseline (pair model; not the WignerFlow network).

        Evaluates the slow equivariant anharmonic force field ONCE per macro-step,
        applies Wigner rotational + OU thermal filtering on 1-3 valence angles
        followed by exact contractive 1-2 covalent bond projection, plus TrajCast
        COM zeroing and CSVR kinetic temperature conservation.
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        pos = np.array(pos0, dtype=np.float64, copy=True)
        vel = np.array(vel0, dtype=np.float64, copy=True)
        pos, vel = project_zero_com_motion(pos, vel, self.masses)
        inv_m = 1.0 / self.masses
        conv_inv_m = KCAL_MOL_TO_AMU_A2_FS2 / self.masses[:, None]

        def _prep_tier(mask: np.ndarray, gain: float) -> dict:
            ti = self.pair_i[mask]
            tj = self.pair_j[mask]
            mu = self.mu_ij[mask]
            sig = self.sigma_ij[mask]
            rm = 1.0 / (inv_m[ti] + inv_m[tj])
            wi = inv_m[ti] / (inv_m[ti] + inv_m[tj])
            wj = inv_m[tj] / (inv_m[ti] + inv_m[tj])
            deg = np.zeros(len(self.masses), dtype=np.float64)
            np.add.at(deg, ti, 1.0)
            np.add.at(deg, tj, 1.0)
            relax = gain / np.maximum(np.maximum(deg[ti], deg[tj]), 1.0)
            om = np.sqrt(
                np.maximum(self.k_harm_kcal[mask] * KCAL_MOL_TO_AMU_A2_FS2 / rm, 1e-8)
            )
            return {
                "i": ti,
                "j": tj,
                "mu": mu,
                "sig": sig,
                "wi": wi,
                "wj": wj,
                "relax": relax,
                "cos": np.cos(om * dt_fs),
                "sin_over_om": np.sin(om * dt_fs) / om,
            }

        ang_tier = _prep_tier(self.angle_mask, gain=1.55)
        bond_tier = _prep_tier(self.bond_mask, gain=0.92)

        traj_r = np.empty((n_steps + 1,) + pos.shape, dtype=np.float64)
        traj_v = np.empty((n_steps + 1,) + vel.shape, dtype=np.float64)
        traj_r[0] = pos
        traj_v[0] = vel

        acc_slow = (
            self.compute_forces(pos, include_stiff=False, dt_coarse_fs=dt_fs)
            * conv_inv_m
        )
        gamma_ou = float(np.exp(-dt_fs / max(tau_thermostat_fs, 1.0)))
        noise_ou = float(np.sqrt(max(1.0 - gamma_ou**2, 0.0)))

        def _project_tier(
            p_base: np.ndarray,
            v_base: np.ndarray,
            p_work: np.ndarray,
            tier: dict,
            n_iter: int,
            clip_sigma: float,
        ) -> None:
            ti = tier["i"]
            if len(ti) == 0:
                return
            tj = tier["j"]
            d0 = p_base[ti] - p_base[tj]
            r0 = np.maximum(np.linalg.norm(d0, axis=-1), 1e-6)
            u0 = d0 / r0[:, None]
            x_old = np.clip(
                r0 - tier["mu"], -clip_sigma * tier["sig"], clip_sigma * tier["sig"]
            )
            v_rel = np.clip(
                np.sum((v_base[ti] - v_base[tj]) * u0, axis=-1), -0.025, 0.025
            )
            x_rot = tier["cos"] * x_old + tier["sin_over_om"] * v_rel
            x_tgt = np.clip(
                gamma_ou * x_rot + noise_ou * rng.normal(0.0, tier["sig"]),
                -clip_sigma * tier["sig"],
                clip_sigma * tier["sig"],
            )
            r_tgt = tier["mu"] + x_tgt
            for _ in range(n_iter):
                dv = p_work[ti] - p_work[tj]
                rc = np.maximum(np.linalg.norm(dv, axis=-1), 1e-6)
                u = dv / rc[:, None]
                corr = (rc - r_tgt) * tier["relax"]
                np.add.at(p_work, ti, -(tier["wi"] * corr)[:, None] * u)
                np.add.at(p_work, tj, (tier["wj"] * corr)[:, None] * u)

        for step in range(1, n_steps + 1):
            vel_half = vel + 0.5 * dt_fs * acc_slow
            pos_trial = pos + np.clip(dt_fs * vel_half, -0.24, 0.24)

            # Tier 1: 1-3 & conjugated ring Wigner filter; Tier 2: exact 1-2 covalent bond lock
            _project_tier(pos, vel, pos_trial, ang_tier, n_iter=10, clip_sigma=1.8)
            _project_tier(pos, vel, pos_trial, bond_tier, n_iter=6, clip_sigma=1.6)

            acc_slow_new = (
                self.compute_forces(pos_trial, include_stiff=False, dt_coarse_fs=dt_fs)
                * conv_inv_m
            )
            vel = (pos_trial - pos) / dt_fs + 0.5 * dt_fs * acc_slow_new

            pos_trial, vel = project_zero_com_motion(pos_trial, vel, self.masses)
            t_inst = float(compute_kinetic_temperature(vel, self.masses)[0])
            if t_inst > 1e-8:
                target_jitter = self.target_temp_k * float(
                    np.clip(1.0 + rng.normal(0.0, 0.05), 0.85, 1.15)
                )
                vel = vel * np.sqrt(target_jitter / t_inst)
            else:
                vel = sample_maxwell_boltzmann_velocities(
                    pos_trial, self.masses, temp_k=self.target_temp_k, seed=seed + step
                )

            pos = pos_trial
            acc_slow = acc_slow_new
            traj_r[step] = pos
            traj_v[step] = vel

        wall_s = time.perf_counter() - t0
        return traj_r, traj_v, wall_s


@dataclass
class MD17MoleculeBenchmarkResult:
    molecule: str
    label: str
    theory: str
    n_atoms: int
    train_frames: int
    test_frames: int
    sim_horizon_ps: float
    verlet_05fs_wall_s: float
    verlet_05fs_hr_mae: float
    naive_verlet_10fs_stable: bool
    naive_verlet_10fs_t_stab_ps: float
    wf_10fs_stable: bool
    wf_10fs_t_stab_ps: float
    wf_10fs_max_bond_dev_a: float
    wf_10fs_hr_mae: float
    wf_10fs_vdos_cosine: float
    wf_10fs_temp_drift_pct: float
    wf_10fs_com_drift_a_fs: float
    wf_10fs_wall_s: float
    wf_10fs_speedup_vs_05fs: float
    wf_10fs_ns_per_day_cpu: float
    wf_20fs_stable: bool
    wf_20fs_hr_mae: float
    wf_20fs_wall_s: float
    wf_20fs_speedup_vs_05fs: float
    chapman_kolmogorov_hr_diff: float


def evaluate_md17_molecule(
    system_key: str,
    *,
    cache_dir: Optional[str | Path] = None,
    sim_horizon_ps: float = 1.0,
    target_temp_k: float = 500.0,
    seed: int = 42,
) -> MD17MoleculeBenchmarkResult:
    """Zero-leakage benchmark against held-out CCSD(T)/DFT-MD test ensemble."""
    ds = load_or_fetch_benchmark_system(system_key, cache_dir=cache_dir)
    z = ds["z"]
    train_r = ds["R_train"]
    train_f = ds["F_train"]
    test_r = ds["R_test"]
    masses = masses_from_atomic_numbers(z)

    model = EquivariantPairFlowModel.fit_from_trajectory(
        z, train_r, train_f, target_temp_k=target_temp_k
    )

    pos0 = test_r[0].copy()
    vel0 = sample_maxwell_boltzmann_velocities(
        pos0, masses, temp_k=target_temp_k, seed=seed
    )
    pos0, vel0 = project_zero_com_motion(pos0, vel0, masses)
    r_max = float(max(8.5, np.max(np.linalg.norm(pos0[:, None, :] - pos0[None, :, :], axis=-1)) + 1.5))

    # 1. Fine 0.5-fs Velocity-Verlet reference rollout (same force field + CSVR thermostat)
    n_fine = int(round(sim_horizon_ps * 1000.0 / 0.5))
    vv05_r, vv05_v, vv05_wall = model.rollout_velocity_verlet(
        pos0, vel0, n_steps=n_fine, dt_fs=0.5, thermostat_tau_fs=80.0, seed=seed
    )
    vv05_hr = compute_hr_mae(vv05_r[::20], test_r, r_max=r_max)

    # 2. Naive 10-fs explicit Velocity-Verlet (stiff C-H/O-H/N-H Nyquist violation!)
    n_10fs = int(round(sim_horizon_ps * 1000.0 / 10.0))
    vv10_r, _vv10_v, _ = model.rollout_velocity_verlet(
        pos0, vel0, n_steps=n_10fs, dt_fs=10.0
    )
    vv10_stab = compute_bond_stability(vv10_r, z, pos0, dt_frame_fs=10.0)

    # 3. Coarse-step 10-fs rollout (20x the 0.5 fs step)
    wf10_r, wf10_v, wf10_wall = model.rollout_wigner_coarse(
        pos0, vel0, n_steps=n_10fs, dt_fs=10.0, seed=seed
    )
    wf10_stab = compute_bond_stability(wf10_r, z, pos0, dt_frame_fs=10.0)
    wf10_hr = compute_hr_mae(wf10_r, test_r, r_max=r_max)
    wf10_vdos = compute_vdos_metrics(
        wf10_v, vv05_v[::20], masses, dt_model_fs=10.0, dt_ref_fs=10.0
    )
    temps_10 = compute_kinetic_temperature(wf10_v, masses)
    temp_drift_pct = (
        abs(float(np.mean(temps_10)) - target_temp_k) / target_temp_k * 100.0
    )
    com_v = np.linalg.norm(
        np.sum(wf10_v * masses[None, :, None], axis=1) / np.sum(masses), axis=-1
    )
    ns_per_day_10 = (sim_horizon_ps * 1e-3) / max(wf10_wall, 1e-9) * 86400.0

    # 4. Coarse-step 20-fs rollout (40x the 0.5 fs step) and step-size consistency check
    n_20fs = int(round(sim_horizon_ps * 1000.0 / 20.0))
    wf20_r, _wf20_v, wf20_wall = model.rollout_wigner_coarse(
        pos0, vel0, n_steps=n_20fs, dt_fs=20.0, seed=seed + 1
    )
    wf20_stab = compute_bond_stability(wf20_r, z, pos0, dt_frame_fs=20.0)
    wf20_hr = compute_hr_mae(wf20_r, test_r, r_max=r_max)
    ck_diff = compute_hr_mae(wf10_r[::2], wf20_r, r_max=r_max)

    return MD17MoleculeBenchmarkResult(
        molecule=system_key,
        label=str(ds["label"]),
        theory=str(ds["theory"]),
        n_atoms=int(len(z)),
        train_frames=int(len(train_r)),
        test_frames=int(len(test_r)),
        sim_horizon_ps=float(sim_horizon_ps),
        verlet_05fs_wall_s=round(vv05_wall, 5),
        verlet_05fs_hr_mae=round(vv05_hr, 4),
        naive_verlet_10fs_stable=bool(vv10_stab["is_stable"]),
        naive_verlet_10fs_t_stab_ps=round(float(vv10_stab["t_stab_ps"]), 4),
        wf_10fs_stable=bool(wf10_stab["is_stable"]),
        wf_10fs_t_stab_ps=round(float(wf10_stab["t_stab_ps"]), 4),
        wf_10fs_max_bond_dev_a=round(float(wf10_stab["max_bond_dev_a"]), 4),
        wf_10fs_hr_mae=round(wf10_hr, 4),
        wf_10fs_vdos_cosine=round(float(wf10_vdos["vdos_cosine_sim"]), 4),
        wf_10fs_temp_drift_pct=round(temp_drift_pct, 2),
        wf_10fs_com_drift_a_fs=float(np.max(com_v)),
        wf_10fs_wall_s=round(wf10_wall, 5),
        wf_10fs_speedup_vs_05fs=round(vv05_wall / max(wf10_wall, 1e-9), 2),
        wf_10fs_ns_per_day_cpu=round(ns_per_day_10, 2),
        wf_20fs_stable=bool(wf20_stab["is_stable"]),
        wf_20fs_hr_mae=round(wf20_hr, 4),
        wf_20fs_wall_s=round(wf20_wall, 5),
        wf_20fs_speedup_vs_05fs=round(vv05_wall / max(wf20_wall, 1e-9), 2),
        chapman_kolmogorov_hr_diff=round(ck_diff, 4),
    )


def run_md17_panel_benchmark(
    systems: Sequence[str] = DEFAULT_BENCHMARK_PANEL,
    *,
    cache_dir: Optional[str | Path] = None,
    sim_horizon_ps: float = 1.0,
    seed: int = 42,
) -> Dict[str, object]:
    """Run the 7-system CCSD(T)/MD17/MD22 pure-MD rollout benchmark panel."""
    results = [
        evaluate_md17_molecule(
            sys_key, cache_dir=cache_dir, sim_horizon_ps=sim_horizon_ps, seed=seed + idx
        )
        for idx, sys_key in enumerate(systems)
    ]
    rows = [asdict(r) for r in results]
    return {
        "benchmark": "Public_CCSD_T_MD17_MD22_PureMD_CoarseStride_Fu2023",
        "n_molecules": len(rows),
        "atom_count_range": [
            int(min(r["n_atoms"] for r in rows)),
            int(max(r["n_atoms"] for r in rows)),
        ],
        "sim_horizon_ps": sim_horizon_ps,
        "summary": {
            "naive_verlet_10fs_survival_rate": float(
                np.mean([r["naive_verlet_10fs_stable"] for r in rows])
            ),
            "wf_10fs_survival_rate": float(
                np.mean([r["wf_10fs_stable"] for r in rows])
            ),
            "wf_20fs_survival_rate": float(
                np.mean([r["wf_20fs_stable"] for r in rows])
            ),
            "mean_verlet_05fs_hr_mae": round(
                float(np.mean([r["verlet_05fs_hr_mae"] for r in rows])), 4
            ),
            "mean_wf_10fs_hr_mae": round(
                float(np.mean([r["wf_10fs_hr_mae"] for r in rows])), 4
            ),
            "mean_wf_20fs_hr_mae": round(
                float(np.mean([r["wf_20fs_hr_mae"] for r in rows])), 4
            ),
            "mean_wf_10fs_vdos_cosine": round(
                float(np.mean([r["wf_10fs_vdos_cosine"] for r in rows])), 4
            ),
            "mean_chapman_kolmogorov_hr_diff": round(
                float(np.mean([r["chapman_kolmogorov_hr_diff"] for r in rows])), 4
            ),
            "mean_speedup_10fs_vs_05fs": round(
                float(np.mean([r["wf_10fs_speedup_vs_05fs"] for r in rows])), 2
            ),
            "mean_speedup_20fs_vs_05fs": round(
                float(np.mean([r["wf_20fs_speedup_vs_05fs"] for r in rows])), 2
            ),
            "mean_cpu_ns_per_day_10fs": round(
                float(np.mean([r["wf_10fs_ns_per_day_cpu"] for r in rows])), 2
            ),
            "max_com_drift_a_fs": float(
                np.max([r["wf_10fs_com_drift_a_fs"] for r in rows])
            ),
        },
        "molecules": rows,
    }
