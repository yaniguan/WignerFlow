"""Fine-Grained 3D Electrolyte Solvation Shell & Exchange Kinetics Analyzer.

Implements the multi-component solvation-structure and ion-transport formalism
synthesized in our 40-paper survey (Xu 2014; Rajput & Persson JACS 2015;
Zhang & Maginn JPCL 2015; Borodin et al. Acc. Chem. Res. 2017 & Joule 2020;
Yamada et al. Nat. Energy 2019; Cao et al. Nat. Energy 2019; France-Lanord &
Grossman PRL 2019; Self, Fong & Persson 2019-2021; Yu, Cui & Bao Nat. Energy
2020/2022; Xu, Borodin & Wang Nature 2023):

- Periodic minimum-image partial RDFs g_{Li,beta}(r), running CN(r), and
  automated first-minimum solvation cutoff r_c detection.
- Exact bipartite connected-component speciation:
    * Cation-centric: SSIP, CIP, AGG-I (finite dimers/trimers <=4 ions),
      AGG-net (extended >=5-ion clusters), and percolation order parameter P_inf.
    * Anion-centric Raman band proxy: Free, CIP (n_c=1), AGG-I (n_c=2), AGG-II (n_c>=3).
- Cluster Nernst-Einstein (cNE) ionicity sigma_cNE / sigma_NE and transference t_+^cNE.
- Primary-sheath additive competitive enrichment alpha_enr, preferential
  solvation free energy Delta G_pref (kJ/mol), LHCE diluent classification,
  and Yamada uncoordinated free-solvent fraction f_free.
- FFT-accelerated intermittent cage autocorrelation C_cage(t), mean exchange
  lifetime tau_ex (ps), vehicularity index Lambda_veh, and Goldilocks
  fast-charging / anion-derived SEI readiness scoring.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import math
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from equitraj.align import (
    SALTS_ATLAS,
    SOLVENTS_ATLAS,
    compute_formulation_descriptors,
    run_periodic_langevin_window,
    yeh_hummer_correction_cm2_s,
)
from equitraj.funnel import NOVEL_ADDITIVE_SPACE_100, decode_candidate_index
from equitraj.units import AMU_A2_PER_FS2_TO_KJ_PER_MOL, KB

GAS_CONSTANT_KJ_MOL_K = 8.314462618e-3
AVOGADRO = 6.02214076e23


def _pbc_distances(
    pos_a: np.ndarray, pos_b: np.ndarray, box_length: float
) -> np.ndarray:
    """Minimum-image distances for shapes (..., M, 3) and (..., K, 3) -> (..., M, K)."""
    dr = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    dr = dr - box_length * np.round(dr / box_length)
    return np.linalg.norm(dr, axis=-1)


def compute_partial_rdf_and_cn(
    traj_pos: np.ndarray,
    box_length: float,
    center_indices: Sequence[int],
    partner_indices: Sequence[int],
    *,
    r_max: float = 7.2,
    n_bins: int = 144,
    default_rc_a: float = 2.85,
) -> Dict[str, object]:
    """Compute partial RDF g(r), running coordination number CN(r), and first-minimum r_c."""
    pos = np.asarray(traj_pos, dtype=np.float64)
    c_idx = np.asarray(center_indices, dtype=np.int64)
    p_idx = np.asarray(partner_indices, dtype=np.int64)
    edges = np.linspace(0.0, r_max, n_bins + 1)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = float(edges[1] - edges[0])

    if len(c_idx) == 0 or len(p_idx) == 0 or len(pos) == 0:
        zeros = np.zeros_like(r_centers)
        return {
            "r_a": r_centers,
            "g_r": zeros,
            "cn_r": zeros,
            "r_cutoff_a": float(default_rc_a),
            "cn_at_rc": 0.0,
            "peak_r_a": 0.0,
            "peak_g": 0.0,
        }

    dists = _pbc_distances(pos[:, c_idx, :], pos[:, p_idx, :], float(box_length))
    # Mask self pairs if center and partner overlap
    if np.intersect1d(c_idx, p_idx).size > 0:
        same = c_idx[:, None] == p_idx[None, :]
        dists = np.where(same[None, :, :], np.inf, dists)

    counts, _ = np.histogram(dists.ravel(), bins=edges)
    vol = float(box_length) ** 3
    n_frames = len(pos)
    rho_partner = len(p_idx) / vol
    shell_vols = (4.0 / 3.0) * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    ideal = n_frames * len(c_idx) * rho_partner * np.maximum(shell_vols, 1e-12)
    g_r = counts.astype(np.float64) / np.maximum(ideal, 1e-12)
    cn_r = np.cumsum(counts.astype(np.float64)) / max(n_frames * len(c_idx), 1)

    # Smooth g(r) lightly with a 5-point kernel to detect first local minimum after primary peak
    kernel = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    g_smooth = np.convolve(g_r, kernel, mode="same")
    peak_mask = (r_centers >= 1.65) & (r_centers <= 2.55)
    if np.any(peak_mask) and np.max(g_smooth[peak_mask]) > 0.35:
        pk_local = int(np.argmax(g_smooth[peak_mask]))
        pk_idx = int(np.flatnonzero(peak_mask)[pk_local])
        peak_r = float(r_centers[pk_idx])
        peak_g = float(g_r[pk_idx])
        min_mask = (r_centers > max(peak_r + 0.30, 2.35)) & (r_centers <= 3.45)
        if np.any(min_mask):
            mn_local = int(np.argmin(g_smooth[min_mask]))
            rc_idx = int(np.flatnonzero(min_mask)[mn_local])
            rc = float(r_centers[rc_idx])
        else:
            rc = float(default_rc_a)
            rc_idx = int(np.argmin(np.abs(r_centers - rc)))
    else:
        rc = float(default_rc_a)
        rc_idx = int(np.argmin(np.abs(r_centers - rc)))
        peak_r = float(r_centers[int(np.argmax(g_r))]) if np.max(g_r) > 0 else 0.0
        peak_g = float(np.max(g_r))

    return {
        "r_a": r_centers,
        "g_r": g_r,
        "cn_r": cn_r,
        "r_cutoff_a": round(rc, 3),
        "cn_at_rc": float(cn_r[rc_idx]),
        "peak_r_a": round(peak_r, 3),
        "peak_g": round(peak_g, 3),
    }


def compute_bipartite_ion_speciation(
    traj_pos: np.ndarray,
    box_length: float,
    cation_indices: Sequence[int],
    anion_donor_groups: Sequence[Sequence[int]],
    *,
    r_cutoff_a: float = 2.95,
    dt_frame_ps: float = 0.05,
) -> Dict[str, float]:
    """Exact bipartite connected-component speciation (SSIP / CIP / AGG-I / AGG-net) & cNE."""
    pos = np.asarray(traj_pos, dtype=np.float64)
    c_idx = np.asarray(cation_indices, dtype=np.int64)
    n_cat = len(c_idx)
    n_an = len(anion_donor_groups)
    n_frames = len(pos)
    if n_cat == 0 or n_an == 0 or n_frames == 0:
        return {
            "ssip_frac": 1.0,
            "cip_frac": 0.0,
            "agg_dimer_frac": 0.0,
            "agg_network_frac": 0.0,
            "agg_total_frac": 0.0,
            "percolation_p_inf": 0.0,
            "anion_free_frac": 1.0,
            "anion_cip_frac": 0.0,
            "anion_agg1_frac": 0.0,
            "anion_agg2_frac": 0.0,
            "mean_anion_cn": 0.0,
            "cne_ionicity": 1.0,
            "cne_t_plus": 0.5,
        }

    # Flatten anion donor atoms for vectorized distance evaluation
    flat_donors: List[int] = []
    an_slices: List[Tuple[int, int]] = []
    for grp in anion_donor_groups:
        s = len(flat_donors)
        flat_donors.extend(int(x) for x in grp)
        an_slices.append((s, len(flat_donors)))
    flat_arr = np.asarray(flat_donors, dtype=np.int64)

    dists = _pbc_distances(pos[:, c_idx, :], pos[:, flat_arr, :], float(box_length))
    # Adjacency A[t, i, a] = 1 iff min_{k in anion a} d(Li_i, k) <= r_cutoff_a
    adj = np.zeros((n_frames, n_cat, n_an), dtype=bool)
    for a_idx, (s, e) in enumerate(an_slices):
        adj[:, :, a_idx] = np.min(dists[:, :, s:e], axis=-1) <= r_cutoff_a

    ssip_cnt = 0
    cip_cnt = 0
    agg_dim_cnt = 0
    agg_net_cnt = 0
    p_inf_acc = 0.0

    # Anion-centric Raman speciation counts (Borodin 2017)
    an_deg = np.sum(adj, axis=1)  # (T, N_an)
    an_free = float(np.mean(an_deg == 0))
    an_cip = float(np.mean(an_deg == 1))
    an_agg1 = float(np.mean(an_deg == 2))
    an_agg2 = float(np.mean(an_deg >= 3))
    mean_acn = float(np.mean(np.sum(adj, axis=2)))

    # Cluster Nernst-Einstein (France-Lanord & Grossman PRL 2019):
    # Weight cluster diffusivities by hydrodynamic 1 / R_cluster ~ (i + j)^(-1/3)
    cne_num_sigma = 0.0
    cne_num_tplus = 0.0
    ne_denom = 0.0

    for t in range(n_frames):
        a_mat = adj[t]
        visited_c = np.zeros(n_cat, dtype=bool)
        visited_a = np.zeros(n_an, dtype=bool)
        max_comp_ions = 0

        # Traverse components starting from cations
        for ci in range(n_cat):
            if visited_c[ci]:
                continue
            comp_c: List[int] = []
            comp_a: List[int] = []
            q: deque[Tuple[str, int]] = deque([("c", ci)])
            visited_c[ci] = True
            while q:
                kind, idx = q.popleft()
                if kind == "c":
                    comp_c.append(idx)
                    for nb_a in np.flatnonzero(a_mat[idx]):
                        if not visited_a[nb_a]:
                            visited_a[nb_a] = True
                            q.append(("a", int(nb_a)))
                else:
                    comp_a.append(idx)
                    for nb_c in np.flatnonzero(a_mat[:, idx]):
                        if not visited_c[nb_c]:
                            visited_c[nb_c] = True
                            q.append(("c", int(nb_c)))

            nc_k = len(comp_c)
            na_k = len(comp_a)
            tot_ions = nc_k + na_k
            max_comp_ions = max(max_comp_ions, tot_ions)

            if na_k == 0 and nc_k == 1:
                ssip_cnt += 1
            elif nc_k == 1 and na_k == 1:
                cip_cnt += 1
            elif tot_ions <= 4:
                agg_dim_cnt += nc_k
            else:
                agg_net_cnt += nc_k

            z_net = nc_k - na_k
            d_rel = (tot_ions) ** (-1.0 / 3.0)
            cne_num_sigma += (z_net**2) * d_rel
            cne_num_tplus += nc_k * z_net * d_rel

        # Count remaining free anions (i=0, j=1 -> z_net = -1)
        n_unvisited_a = int(np.sum(~visited_a))
        if n_unvisited_a > 0:
            max_comp_ions = max(max_comp_ions, 1)
            # Model assumption (not fitted to data): free anions get a relative mobility
            # weight of 1.25 versus solvated Li+.
            cne_num_sigma += n_unvisited_a * 1.25
        ne_denom += n_cat * 1.0 + n_an * 1.25
        p_inf_acc += max_comp_ions / float(n_cat + n_an)

    tot_cat_obs = float(n_frames * n_cat)
    ionicity_cne = float(np.clip(cne_num_sigma / max(ne_denom, 1e-12), 0.05, 1.0))
    t_plus_cne = float(
        np.clip(cne_num_tplus / max(cne_num_sigma, 1e-12), -0.25, 0.92)
    )
    ssip_f = ssip_cnt / tot_cat_obs
    cip_f = cip_cnt / tot_cat_obs
    agg_d_f = agg_dim_cnt / tot_cat_obs
    agg_n_f = agg_net_cnt / tot_cat_obs

    return {
        "ssip_frac": round(ssip_f, 4),
        "cip_frac": round(cip_f, 4),
        "agg_dimer_frac": round(agg_d_f, 4),
        "agg_network_frac": round(agg_n_f, 4),
        "agg_total_frac": round(agg_d_f + agg_n_f, 4),
        "percolation_p_inf": round(p_inf_acc / n_frames, 4),
        "anion_free_frac": round(an_free, 4),
        "anion_cip_frac": round(an_cip, 4),
        "anion_agg1_frac": round(an_agg1, 4),
        "anion_agg2_frac": round(an_agg2, 4),
        "mean_anion_cn": round(mean_acn, 4),
        "cne_ionicity": round(ionicity_cne, 4),
        "cne_t_plus": round(t_plus_cne, 4),
    }


def compute_additive_enrichment_and_free_solvent(
    cn_backbone: float,
    cn_additive: float,
    *,
    n_li: int,
    n_backbone_mols: int,
    n_additive_mols: int,
    temperature_k: float = 298.15,
) -> Dict[str, object]:
    """Compute additive enrichment alpha_enr, Delta G_pref (kJ/mol), role & free-solvent f_free."""
    n_solv_tot = max(n_backbone_mols + n_additive_mols, 1)
    cn_solv_tot = max(cn_backbone + cn_additive, 1e-8)
    x_add_bulk = n_additive_mols / float(n_solv_tot)
    x_add_shell = cn_additive / cn_solv_tot

    if n_additive_mols > 0 and x_add_bulk > 1e-6:
        alpha_enr = x_add_shell / x_add_bulk
        ratio = ((cn_additive + 1e-4) / max(n_additive_mols, 1)) / (
            (cn_backbone + 1e-4) / max(n_backbone_mols, 1)
        )
        dg_pref = -GAS_CONSTANT_KJ_MOL_K * temperature_k * math.log(max(ratio, 1e-6))
    else:
        alpha_enr = 0.0
        dg_pref = 0.0

    if cn_additive < 0.06:
        detected_role = "lhce_noncoordinating_diluent"
    elif alpha_enr >= 1.15:
        detected_role = "competitive_inner_sheath_sei_former"
    elif alpha_enr >= 0.45:
        detected_role = "weakly_solvating_cosolvent"
    else:
        detected_role = "outer_shell_fluidity_modulator"

    f_free = float(
        np.clip(1.0 - (n_li * (cn_backbone + cn_additive)) / float(n_solv_tot), 0.0, 1.0)
    )
    return {
        "x_additive_bulk": round(x_add_bulk, 4),
        "x_additive_shell": round(x_add_shell, 4),
        "additive_enrichment_alpha": round(float(alpha_enr), 4),
        "preferential_dg_kj_mol": round(float(dg_pref), 3),
        "free_solvent_frac": round(f_free, 4),
        "detected_solvation_role": detected_role,
    }


def compute_cage_exchange_kinetics(
    traj_pos: np.ndarray,
    box_length: float,
    cation_indices: Sequence[int],
    donor_indices: Sequence[int],
    *,
    dt_frame_ps: float = 0.25,
    r_cutoff_a: float = 2.85,
    d_li_cm2_s: float = 4.0e-6,
) -> Dict[str, float]:
    """FFT intermittent cage correlation C_cage(t), residence lifetime tau_ex & vehicularity."""
    pos = np.asarray(traj_pos, dtype=np.float64)
    c_idx = np.asarray(cation_indices, dtype=np.int64)
    d_idx = np.asarray(donor_indices, dtype=np.int64)
    t_len = len(pos)
    if t_len < 4 or len(c_idx) == 0 or len(d_idx) == 0:
        return {
            "tau_exchange_ps": 0.0,
            "c_cage_end": 0.0,
            "vehicularity_index": 0.0,
        }

    dists = _pbc_distances(pos[:, c_idx, :], pos[:, d_idx, :], float(box_length))
    h = (dists <= r_cutoff_a).astype(np.float64).reshape(t_len, -1)
    mean_h = float(np.sum(h) / t_len)
    if mean_h <= 1e-12:
        return {
            "tau_exchange_ps": 0.0,
            "c_cage_end": 0.0,
            "vehicularity_index": 0.0,
        }

    n_lags = max(t_len // 2, 2)
    n_fft = 1 << int(np.ceil(np.log2(2 * t_len)))
    spec = np.fft.rfft(h, n=n_fft, axis=0)
    acf = np.fft.irfft(spec * np.conj(spec), n=n_fft, axis=0)[:n_lags]
    norm = np.arange(t_len, t_len - n_lags, -1, dtype=np.float64)
    c_cage = np.sum(acf, axis=1) / (np.maximum(norm, 1.0) * max(mean_h, 1e-12))
    c_cage = np.clip(c_cage / max(c_cage[0], 1e-12), 0.0, 1.0)

    # Fit exponential tail if C_cage hasn't fully decayed within window
    t_grid = np.arange(n_lags, dtype=np.float64) * dt_frame_ps
    trapz_part = float(np.trapezoid(c_cage, t_grid)) if hasattr(np, "trapezoid") else float(np.trapz(c_cage, t_grid))
    c_end = float(c_cage[-1])
    t_end = float(t_grid[-1])
    if 0.04 < c_end < 0.985 and t_end > 0.0:
        tau_fit = -t_end / math.log(c_end)
        tau_ex = float(np.clip(trapz_part + c_end * tau_fit, 0.5, 1500.0))
    elif c_end >= 0.985:
        tau_ex = float(min(1500.0, t_end * 40.0))
    else:
        tau_ex = max(trapz_part, dt_frame_ps)

    # Diffusion length prior to exchange: L_ex [A] = sqrt(6 * D [A^2/ps] * tau_ex [ps])
    # 1 cm^2/s = 1e16 A^2 / 1e12 ps = 1e4 A^2/ps
    d_a2_ps = max(float(d_li_cm2_s), 0.0) * 1e4
    l_ex_a = math.sqrt(6.0 * d_a2_ps * tau_ex)
    vehicularity = l_ex_a / max(r_cutoff_a, 1e-6)

    return {
        "tau_exchange_ps": round(tau_ex, 2),
        "c_cage_end": round(c_end, 4),
        "vehicularity_index": round(vehicularity, 3),
    }


def score_fast_charging_sei_readiness(
    speciation: Dict[str, float],
    enrichment: Dict[str, object],
    kinetics: Dict[str, float],
    *,
    fluorine_index: float = 0.0,
) -> Dict[str, object]:
    """Classify the solvation regime with our heuristic target ranges (not values from one paper)."""
    cip = float(speciation["cip_frac"])
    agg_tot = float(speciation["agg_total_frac"])
    agg_net = float(speciation["agg_network_frac"])
    p_inf = float(speciation["percolation_p_inf"])
    ionicity = float(speciation["cne_ionicity"])
    tau_ps = float(kinetics["tau_exchange_ps"])
    is_lhce = enrichment["detected_solvation_role"] == "lhce_noncoordinating_diluent"

    # Ideal CIP target: 15% - 45% (peak around 28%)
    cip_score = max(0.0, 1.0 - abs(cip - 0.28) / 0.28)
    agg_limit = 0.45 if is_lhce else 0.28
    agg_penalty = max(0.0, (agg_tot - agg_limit) * 2.0) + max(0.0, (p_inf - 0.35) * 2.5)
    kin_score = 1.0 if tau_ps <= 250.0 else max(0.0, 1.0 - (tau_ps - 250.0) / 500.0)
    sei_propensity = float(
        np.clip(
            0.55 * min((cip + 0.7 * agg_tot) / 0.35, 1.2)
            + 0.45 * min(fluorine_index / 0.12, 1.2),
            0.0,
            1.0,
        )
    )
    readiness = float(
        np.clip(
            40.0 * cip_score
            + 25.0 * ionicity
            + 20.0 * kin_score
            + 15.0 * sei_propensity
            - 30.0 * agg_penalty,
            0.0,
            100.0,
        )
    )

    if 0.14 <= cip <= 0.48 and agg_net <= 0.15 and p_inf < 0.35:
        regime = "GOLDILOCKS_FAST_CHARGE_ANION_SEI"
    elif cip < 0.14 and agg_tot < 0.10:
        regime = "SSIP_SOLVENT_DOMINATED"
    elif is_lhce and (cip + agg_tot) >= 0.25:
        regime = "LHCE_LOCAL_CLUSTER_SEI"
    else:
        regime = "AGGREGATE_ENRICHED_WSE"

    return {
        "sei_inorganic_propensity": round(sei_propensity, 3),
        "fast_charge_sei_readiness_100": round(readiness, 1),
        "solvation_regime": regime,
    }


def build_multicomponent_solvation_box(
    desc: Dict[str, object],
    *,
    n_ion_pairs: int = 8,
    seed: int = 42,
) -> Dict[str, object]:
    """Construct a periodic 3D multi-component electrolyte box distinguishing backbone vs additive."""
    rng = np.random.default_rng(seed)
    c_m = float(desc["c_molar"])
    vol_a3 = (1660.539067 / c_m) * n_ion_pairs
    box_l = max(15.0, float(vol_a3 ** (1.0 / 3.0)))
    vol_eff_a3 = box_l**3

    solv_mass_g = (vol_eff_a3 * 1e-24) * float(desc["rho_solv_g_cm3"])
    n_solv = int(
        np.clip(
            round((solv_mass_g / float(desc["avg_mw_solv"])) * AVOGADRO),
            6 * n_ion_pairs,
            16 * n_ion_pairs,
        )
    )

    wa = float(desc.get("additive_wt_frac", 0.0))
    add_info = None
    if desc.get("additive_id") is not None or desc.get("additive_name"):
        target_id = str(desc.get("additive_id", ""))
        target_nm = str(desc.get("additive_name", ""))
        matches = [
            a
            for a in NOVEL_ADDITIVE_SPACE_100
            if a.get("add_id") == target_id or a.get("name") == target_nm
        ]
        add_info = matches[0] if matches else None

    if add_info is not None and wa > 0.0:
        # Convert weight fraction to molecule count
        bb_parts = str(desc["backbone"]).split(":")
        s1 = SOLVENTS_ATLAS.get(bb_parts[0], SOLVENTS_ATLAS["EC"])
        s2 = SOLVENTS_ATLAS.get(bb_parts[-1], s1)
        w_bb = max(1.0 - wa, 0.05)
        m_bb = 0.5 * w_bb / s1["mw"] + 0.5 * w_bb / s2["mw"]
        m_add = wa / add_info["mw"]
        x_add = m_add / (m_bb + m_add)
        n_add = max(1, int(round(n_solv * x_add)))
        dn_add = float(add_info["dn"])
        eps_add = float(add_info["eps_r"])
        dn_bb = 0.5 * (s1["dn"] + s2["dn"])
        eps_bb = 0.5 * (s1["eps_r"] + s2["eps_r"])
    else:
        n_add = 0
        dn_add, eps_add = 0.0, 2.0
        dn_bb = float(desc["dn_mix"])
        eps_bb = float(desc["eps_r_mix"])
    n_bb = max(1, n_solv - n_add)

    n_total_mols = 2 * n_ion_pairs + n_bb + n_add
    grid_k = max(3, int(math.ceil(n_total_mols ** (1.0 / 3.0))))
    axes = (np.arange(grid_k, dtype=np.float64) + 0.5) * (box_l / grid_k)
    gx, gy, gz = np.meshgrid(axes, axes, axes, indexing="ij")
    centers = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    perm = rng.permutation(len(centers))[:n_total_mols]
    mol_centers = centers[perm] + rng.uniform(-0.14, 0.14, size=(n_total_mols, 3))

    pos_list, masses, charges, sig, eps = [], [], [], [], []
    types, mol_ids = [], []
    bonds, bond_r0, bond_k = [], [], []
    excl = set()
    cat_idx: List[int] = []
    anion_groups: List[List[int]] = []
    anion_donors_flat: List[int] = []
    bb_o_idx: List[int] = []
    add_o_idx: List[int] = []

    q_scale = 0.82  # Electronic continuum polarizability screening (Madrid/ECC convention)

    # 1. Li+ cations
    for m in range(n_ion_pairs):
        cat_idx.append(len(pos_list))
        pos_list.append(mol_centers[m] % box_l)
        masses.append(6.941)
        charges.append(+1.0 * q_scale)
        sig.append(1.92)
        eps.append(0.09)
        types.append(0)
        mol_ids.append(m)

    # 2. Multi-site anions (central core + 4 coordinating O/F sites)
    tet = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.92, 0.92, 0.92],
            [0.92, -0.92, -0.92],
            [-0.92, 0.92, -0.92],
            [-0.92, -0.92, 0.92],
        ],
        dtype=np.float64,
    )
    r_bond_an = max(1.48, 0.62 * float(desc["r_anion_A"]))
    tet[1:] = tet[1:] / np.linalg.norm(tet[1]) * r_bond_an
    m_an_tot = max(
        85.0,
        SALTS_ATLAS.get(str(desc["salt"]), SALTS_ATLAS["LiPF6"])["mw"] - 6.941,
    )
    an_donor_sigma = max(2.58, 2.98 - 0.14 * (float(desc["salt_assoc"]) - 1.0))

    for m in range(n_ion_pairs):
        mc = mol_centers[n_ion_pairs + m]
        base = len(pos_list)
        mid = n_ion_pairs + m
        grp: List[int] = []
        for s_idx, off in enumerate(tet):
            pos_list.append((mc + off) % box_l)
            mol_ids.append(mid)
            if s_idx == 0:
                masses.append(0.36 * m_an_tot)
                charges.append(+0.60 * q_scale)
                sig.append(3.35)
                eps.append(0.55)
                types.append(1)
            else:
                idx_cur = base + s_idx
                grp.append(idx_cur)
                anion_donors_flat.append(idx_cur)
                masses.append(0.16 * m_an_tot)
                charges.append(-0.40 * q_scale)
                sig.append(an_donor_sigma)
                eps.append(0.24)
                types.append(2)
                bonds.append((base, idx_cur))
                bond_r0.append(r_bond_an)
                bond_k.append(1500.0)
        anion_groups.append(grp)
        for a in range(5):
            for b in range(a + 1, 5):
                excl.add((base + a, base + b))

    # 3. Dipolar 3-site backbone & additive molecules with DN-calibrated donor charges
    solv_geom = np.array(
        [[-1.18, 0.32, 0.0], [0.0, 0.0, 0.0], [1.21, 0.0, 0.0]], dtype=np.float64
    )

    def _append_solvent_mol(mc: np.ndarray, mid: int, dn_val: float, eps_val: float, mw_val: float, is_add: bool) -> None:
        q_o = -float(np.clip(0.34 + 0.0135 * dn_val + 0.0020 * eps_val, 0.35, 0.74))
        site_q = [-0.15 * q_o, -0.85 * q_o, q_o]
        site_m = [0.38 * mw_val, 0.34 * mw_val, 0.28 * mw_val]
        # Weakly solvating fluorinated additives have slightly larger steric CF3/CHF2 tail
        o_sig = 2.96 if (is_add and dn_val < 13.0) else 2.86
        site_sig = [3.48 if is_add else 3.38, 3.25, o_sig]
        site_t = [3, 3, 4]

        q_rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q_rot) < 0:
            q_rot[:, 0] = -q_rot[:, 0]
        coords = mc + solv_geom @ q_rot.T
        base = len(pos_list)
        if is_add:
            add_o_idx.append(base + 2)
        else:
            bb_o_idx.append(base + 2)

        for s_i in range(3):
            pos_list.append(coords[s_i] % box_l)
            masses.append(site_m[s_i])
            charges.append(site_q[s_i])
            sig.append(site_sig[s_i])
            eps.append(0.36 if s_i == 2 else 0.26)
            types.append(site_t[s_i])
            mol_ids.append(mid)
        bonds.extend([(base, base + 1), (base + 1, base + 2), (base, base + 2)])
        bond_r0.extend([1.22, 1.21, 2.39])
        bond_k.extend([1400.0, 1600.0, 600.0])
        for a in range(3):
            for b in range(a + 1, 3):
                excl.add((base + a, base + b))

    mw_bb = max(41.0, float(desc["avg_mw_solv"]))
    mw_add = float(add_info["mw"]) if add_info is not None else mw_bb
    cursor = 2 * n_ion_pairs
    for b_m in range(n_bb):
        _append_solvent_mol(mol_centers[cursor + b_m], cursor + b_m, dn_bb, eps_bb, mw_bb, False)
    cursor += n_bb
    for a_m in range(n_add):
        _append_solvent_mol(mol_centers[cursor + a_m], cursor + a_m, dn_add, eps_add, mw_add, True)

    pos_arr = np.asarray(pos_list, dtype=np.float64)
    mass_arr = np.asarray(masses, dtype=np.float64)
    temp_k = float(desc["temperature_k"])
    v_std = np.sqrt((KB * temp_k / AMU_A2_PER_FS2_TO_KJ_PER_MOL) / mass_arr)
    vel_arr = rng.normal(size=pos_arr.shape) * v_std[:, None]
    vel_arr -= (vel_arr * mass_arr[:, None]).sum(axis=0) / mass_arr.sum()

    eta_est = float(desc["eta0_mpa_s"]) * math.exp(
        float(desc["salt_b_visc"]) * c_m + 0.14 * (c_m**2)
    )
    friction_ps = float(np.clip(16.0 * eta_est, 5.0, 320.0))

    return {
        "positions": pos_arr,
        "velocities": vel_arr,
        "masses": mass_arr,
        "charges": np.asarray(charges, dtype=np.float64),
        "sigma_a": np.asarray(sig, dtype=np.float64),
        "eps_kj": np.asarray(eps, dtype=np.float64),
        "types": np.asarray(types, dtype=np.int64),
        "molecule_ids": np.asarray(mol_ids, dtype=np.int64),
        "bonds": np.asarray(bonds, dtype=np.int64),
        "bond_r0": np.asarray(bond_r0, dtype=np.float64),
        "bond_k": np.asarray(bond_k, dtype=np.float64),
        "excluded_pairs": excl,
        "cation_indices": np.asarray(cat_idx, dtype=np.int64),
        "anion_donor_groups": anion_groups,
        "anion_donor_indices": np.asarray(anion_donors_flat, dtype=np.int64),
        "backbone_o_indices": np.asarray(bb_o_idx, dtype=np.int64),
        "additive_o_indices": np.asarray(add_o_idx, dtype=np.int64),
        "n_backbone_mols": int(n_bb),
        "n_additive_mols": int(n_add),
        "box_length": float(box_l),
        "dielectric": float(max(1.45, float(desc["eps_inf_mix"]) * 1.15)),
        "friction_per_ps": friction_ps,
        "viscosity_mpa_s": float(eta_est),
        "temperature_k": temp_k,
    }


@dataclass
class SolvationCandidateReport:
    rank: int
    candidate_id: int
    formula: str
    salt: str
    backbone: str
    additive_name: str
    c_molar: float
    box_length_a: float
    n_atoms: int
    sigma_298K_mS_cm: float
    sigma_253K_mS_cm: float
    rc_li_solv_a: float
    rc_li_anion_a: float
    cn_backbone: float
    cn_additive: float
    cn_anion: float
    cn_total: float
    ssip_frac: float
    cip_frac: float
    agg_dimer_frac: float
    agg_network_frac: float
    percolation_p_inf: float
    anion_free_frac: float
    anion_cip_frac: float
    cne_ionicity: float
    cne_t_plus: float
    additive_enrichment_alpha: float
    preferential_dg_kj_mol: float
    free_solvent_frac: float
    detected_solvation_role: str
    tau_exchange_ps: float
    vehicularity_index: float
    sei_inorganic_propensity: float
    fast_charge_sei_readiness_100: float
    solvation_regime: str
    wall_time_s: float


def analyze_candidate_solvation_3d(
    desc: Dict[str, object],
    *,
    rank: int = 1,
    sigma_298k: float = 0.0,
    sigma_253k: float = 0.0,
    n_ion_pairs: int = 8,
    equil_frames: int = 10,
    prod_frames: int = 28,
    substeps: int = 8,
    dt_fs: float = 2.5,
    seed: int = 42,
) -> Tuple[SolvationCandidateReport, Dict[str, np.ndarray]]:
    """Run 3D periodic solvation equilibration + production and extract full shell physics."""
    t0 = time.perf_counter()
    box = build_multicomponent_solvation_box(desc, n_ion_pairs=n_ion_pairs, seed=seed)
    total_frames = equil_frames + prod_frames
    traj_pos, _traj_vel = run_periodic_langevin_window(
        box, n_frames=total_frames, substeps=substeps, dt_fs=dt_fs, seed=seed
    )
    prod_pos = traj_pos[equil_frames:]
    dt_frame_ps = (substeps * dt_fs) * 1e-3
    box_l = float(box["box_length"])

    rdf_bb = compute_partial_rdf_and_cn(
        prod_pos,
        box_l,
        box["cation_indices"],
        box["backbone_o_indices"],
        default_rc_a=2.85,
    )
    rdf_add = compute_partial_rdf_and_cn(
        prod_pos,
        box_l,
        box["cation_indices"],
        box["additive_o_indices"],
        default_rc_a=2.90,
    )
    rdf_an = compute_partial_rdf_and_cn(
        prod_pos,
        box_l,
        box["cation_indices"],
        box["anion_donor_indices"],
        default_rc_a=2.95,
    )

    rc_an = float(np.clip(rdf_an["r_cutoff_a"], 2.75, 3.20))
    rc_solv = float(np.clip(rdf_bb["r_cutoff_a"], 2.65, 3.10))

    spec = compute_bipartite_ion_speciation(
        prod_pos,
        box_l,
        box["cation_indices"],
        box["anion_donor_groups"],
        r_cutoff_a=rc_an,
        dt_frame_ps=dt_frame_ps,
    )
    enr = compute_additive_enrichment_and_free_solvent(
        float(rdf_bb["cn_at_rc"]),
        float(rdf_add["cn_at_rc"]),
        n_li=len(box["cation_indices"]),
        n_backbone_mols=box["n_backbone_mols"],
        n_additive_mols=box["n_additive_mols"],
        temperature_k=float(box["temperature_k"]),
    )

    # Estimate Yeh-Hummer-corrected Li+ self-diffusivity for vehicularity index
    all_solv_donors = np.concatenate([box["backbone_o_indices"], box["additive_o_indices"]])
    d_yh = yeh_hummer_correction_cm2_s(
        box_l, float(box["temperature_k"]), float(box["viscosity_mpa_s"])
    )
    d_li_est = max(2.2e-6 / max(float(box["viscosity_mpa_s"]), 0.25), d_yh * 1.6)
    kin = compute_cage_exchange_kinetics(
        prod_pos,
        box_l,
        box["cation_indices"],
        all_solv_donors,
        dt_frame_ps=dt_frame_ps,
        r_cutoff_a=rc_solv,
        d_li_cm2_s=d_li_est,
    )

    f_idx = float(desc.get("fluorine_index", 0.0))
    ready = score_fast_charging_sei_readiness(spec, enr, kin, fluorine_index=f_idx)
    cn_tot = (
        float(rdf_bb["cn_at_rc"])
        + float(rdf_add["cn_at_rc"])
        + float(spec["mean_anion_cn"])
    )
    wall_s = time.perf_counter() - t0

    rep = SolvationCandidateReport(
        rank=int(rank),
        candidate_id=int(desc.get("candidate_id", -1)),
        formula=str(desc.get("formula_label", desc.get("formula", "Baseline"))),
        salt=str(desc.get("salt_label", desc.get("salt", "LiPF6"))),
        backbone=str(desc.get("backbone", "EC:DMC")),
        additive_name=str(desc.get("additive_name", "None")),
        c_molar=float(desc["c_molar"]),
        box_length_a=round(box_l, 2),
        n_atoms=int(len(box["positions"])),
        sigma_298K_mS_cm=round(float(sigma_298k), 2),
        sigma_253K_mS_cm=round(float(sigma_253k), 2),
        rc_li_solv_a=round(rc_solv, 3),
        rc_li_anion_a=round(rc_an, 3),
        cn_backbone=round(float(rdf_bb["cn_at_rc"]), 3),
        cn_additive=round(float(rdf_add["cn_at_rc"]), 3),
        cn_anion=round(float(spec["mean_anion_cn"]), 3),
        cn_total=round(cn_tot, 3),
        ssip_frac=spec["ssip_frac"],
        cip_frac=spec["cip_frac"],
        agg_dimer_frac=spec["agg_dimer_frac"],
        agg_network_frac=spec["agg_network_frac"],
        percolation_p_inf=spec["percolation_p_inf"],
        anion_free_frac=spec["anion_free_frac"],
        anion_cip_frac=spec["anion_cip_frac"],
        cne_ionicity=spec["cne_ionicity"],
        cne_t_plus=spec["cne_t_plus"],
        additive_enrichment_alpha=float(enr["additive_enrichment_alpha"]),
        preferential_dg_kj_mol=float(enr["preferential_dg_kj_mol"]),
        free_solvent_frac=float(enr["free_solvent_frac"]),
        detected_solvation_role=str(enr["detected_solvation_role"]),
        tau_exchange_ps=kin["tau_exchange_ps"],
        vehicularity_index=kin["vehicularity_index"],
        sei_inorganic_propensity=float(ready["sei_inorganic_propensity"]),
        fast_charge_sei_readiness_100=float(ready["fast_charge_sei_readiness_100"]),
        solvation_regime=str(ready["solvation_regime"]),
        wall_time_s=round(wall_s, 4),
    )
    curves = {
        "r_a": rdf_bb["r_a"],
        "g_li_backbone": rdf_bb["g_r"],
        "g_li_additive": rdf_add["g_r"],
        "g_li_anion": rdf_an["g_r"],
        "cn_li_backbone": rdf_bb["cn_r"],
        "cn_li_additive": rdf_add["cn_r"],
        "cn_li_anion": rdf_an["cn_r"],
    }
    return rep, curves


def build_commercial_ec_dmc_baseline() -> Dict[str, object]:
    """Standard 1.0 M LiPF6 in EC:DMC (1:1 w/w) commercial baseline descriptor."""
    desc = compute_formulation_descriptors(
        salt="LiPF6",
        solvent_ratios={"EC": 0.5, "DMC": 0.5},
        ratio_type="w",
        c_val=1.0,
        c_units="mol/L",
        temperature_k=298.15,
    )
    desc.update(
        {
            "candidate_id": -1,
            "formula_label": "1.0M LiPF6 in EC:DMC (1:1 Commercial Baseline)",
            "salt_label": "LiPF6",
            "backbone": "EC:DMC",
            "additive_name": "None",
            "additive_wt_frac": 0.0,
            "fluorine_index": 0.0,
        }
    )
    return desc


def analyze_top15_funnel_candidates(
    top_candidates_json: Sequence[Dict[str, object]],
    *,
    n_ion_pairs: int = 8,
    prod_frames: int = 28,
    seed: int = 100,
) -> Tuple[Dict[str, object], Dict[str, Dict[str, np.ndarray]]]:
    """Run fine-grained 3D solvation-shell audit across all Top 15 candidates + commercial baseline."""
    t0 = time.perf_counter()
    reports: List[Dict[str, object]] = []
    rdf_curves: Dict[str, Dict[str, np.ndarray]] = {}

    # Commercial reference first (rank 0)
    base_desc = build_commercial_ec_dmc_baseline()
    base_rep, base_curves = analyze_candidate_solvation_3d(
        base_desc,
        rank=0,
        sigma_298k=11.20,  # hand-entered approximate value; CALiSol-23 lists 12.09 mS/cm at 297.9 K (doi 10.1016/j.jpowsour.2011.06.030)
        sigma_253k=2.85,  # hand-entered approximate value; CALiSol-23 lists 3.31 mS/cm at 253.4 K (same source)
        n_ion_pairs=n_ion_pairs,
        prod_frames=prod_frames,
        seed=seed,
    )
    rdf_curves["baseline_EC_DMC"] = base_curves

    for idx, item in enumerate(top_candidates_json[:15], start=1):
        cid = int(item["candidate_id"])
        desc = decode_candidate_index(cid)
        rep, curves = analyze_candidate_solvation_3d(
            desc,
            rank=idx,
            sigma_298k=float(item.get("sigma_298K_mS_cm", 0.0)),
            sigma_253k=float(item.get("sigma_253K_mS_cm", 0.0)),
            n_ion_pairs=n_ion_pairs,
            prod_frames=prod_frames,
            seed=seed + idx,
        )
        reports.append(asdict(rep))
        if idx <= 3:
            rdf_curves[f"rank_{idx}_{desc['backbone']}"] = curves

    goldilocks_hits = [
        r for r in reports if r["solvation_regime"] == "GOLDILOCKS_FAST_CHARGE_ANION_SEI"
    ]
    best_dual = max(
        reports,
        key=lambda r: 0.55 * r["fast_charge_sei_readiness_100"] + 1.15 * r["sigma_253K_mS_cm"],
    )

    summary = {
        "n_candidates_analyzed": len(reports),
        "wall_time_total_s": round(time.perf_counter() - t0, 3),
        "commercial_baseline": asdict(base_rep),
        "top15_summary": {
            "mean_cip_frac": round(float(np.mean([r["cip_frac"] for r in reports])), 4),
            "mean_ssip_frac": round(float(np.mean([r["ssip_frac"] for r in reports])), 4),
            "mean_agg_network_frac": round(
                float(np.mean([r["agg_network_frac"] for r in reports])), 4
            ),
            "max_percolation_p_inf": round(
                float(np.max([r["percolation_p_inf"] for r in reports])), 4
            ),
            "mean_tau_exchange_ps": round(
                float(np.mean([r["tau_exchange_ps"] for r in reports])), 2
            ),
            "n_goldilocks_fast_charge_sei": len(goldilocks_hits),
            "top_dual_conductivity_sei_champion": {
                "rank": best_dual["rank"],
                "formula": best_dual["formula"],
                "sigma_298K_mS_cm": best_dual["sigma_298K_mS_cm"],
                "sigma_253K_mS_cm": best_dual["sigma_253K_mS_cm"],
                "cip_frac": best_dual["cip_frac"],
                "fast_charge_sei_readiness_100": best_dual["fast_charge_sei_readiness_100"],
            },
        },
        "candidates": reports,
    }
    return summary, rdf_curves
