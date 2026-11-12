"""Unit tests for fine-grained electrolyte solvation & exchange kinetics (equitraj.solvation)."""

from __future__ import annotations

import numpy as np
import pytest

from equitraj.funnel import decode_candidate_index
from equitraj.solvation import (
    analyze_candidate_solvation_3d,
    compute_additive_enrichment_and_free_solvent,
    compute_bipartite_ion_speciation,
    compute_cage_exchange_kinetics,
    compute_partial_rdf_and_cn,
    score_fast_charging_sei_readiness,
)


def test_bipartite_speciation_ssip_cip_agg_partition() -> None:
    # Construct a 1-frame periodic box (L=20 A) with 4 Li+ and 4 anions:
    # Li0: isolated -> SSIP (1/4)
    # Li1: paired only with Anion0 -> CIP (1/4)
    # Li2 & Li3: both bridged to Anion1 & Anion2 -> AGG-I of 4 ions (2/4)
    # Anion3: uncoordinated free anion
    pos = np.array(
        [
            # Cations 0..3
            [2.0, 2.0, 2.0],
            [7.0, 7.0, 7.0],
            [13.0, 13.0, 13.0],
            [15.2, 13.0, 13.0],
            # Anion donors 4..7
            [7.0, 7.0, 9.1],  # Anion0 near Li1 (2.1 A)
            [14.1, 13.0, 14.0],  # Anion1 bridging Li2 & Li3
            [14.1, 13.0, 12.0],  # Anion2 bridging Li2 & Li3
            [18.5, 2.0, 2.0],  # Anion3 free
        ],
        dtype=np.float64,
    )[None, :, :]

    spec = compute_bipartite_ion_speciation(
        pos,
        box_length=20.0,
        cation_indices=[0, 1, 2, 3],
        anion_donor_groups=[[4], [5], [6], [7]],
        r_cutoff_a=2.8,
    )
    assert spec["ssip_frac"] == pytest.approx(0.25)
    assert spec["cip_frac"] == pytest.approx(0.25)
    assert spec["agg_dimer_frac"] == pytest.approx(0.50)
    assert spec["agg_network_frac"] == pytest.approx(0.0)
    assert (
        spec["ssip_frac"] + spec["cip_frac"] + spec["agg_total_frac"]
    ) == pytest.approx(1.0)
    assert spec["percolation_p_inf"] == pytest.approx(4.0 / 8.0)
    assert spec["anion_free_frac"] == pytest.approx(0.25)
    assert spec["anion_cip_frac"] == pytest.approx(0.25)
    assert spec["anion_agg1_frac"] == pytest.approx(0.50)
    # Neutral CIP and neutral 2+2 AGG-I reduce cNE ionicity well below 1.0
    assert 0.05 <= spec["cne_ionicity"] < 0.50


def test_rdf_enrichment_and_cage_exchange() -> None:
    rng = np.random.default_rng(3)
    box_l = 16.0
    n_frames = 32
    traj = np.zeros((n_frames, 6, 3), dtype=np.float64)
    traj[:, 0] = [8.0, 8.0, 8.0]  # Li+
    # Two persistent inner-shell oxygens at ~2.05 A, one exchanging oxygen
    traj[:, 1] = [10.05, 8.0, 8.0] + rng.normal(0.0, 0.04, size=(n_frames, 3))
    traj[:, 2] = [8.0, 10.05, 8.0] + rng.normal(0.0, 0.04, size=(n_frames, 3))
    traj[:16, 3] = [8.0, 8.0, 10.10]
    traj[16:, 3] = [8.0, 8.0, 13.50]  # leaves shell halfway
    traj[:, 4] = [3.0, 3.0, 3.0]
    traj[:, 5] = [13.0, 13.0, 3.0]

    rdf = compute_partial_rdf_and_cn(
        traj, box_l, center_indices=[0], partner_indices=[1, 2, 3, 4, 5]
    )
    assert 1.85 <= rdf["peak_r_a"] <= 2.25
    assert rdf["cn_at_rc"] == pytest.approx(2.5, abs=0.2)

    enr = compute_additive_enrichment_and_free_solvent(
        cn_backbone=2.8,
        cn_additive=1.2,
        n_li=4,
        n_backbone_mols=36,
        n_additive_mols=4,
    )
    assert enr["additive_enrichment_alpha"] == pytest.approx(3.0, abs=0.05)
    assert enr["preferential_dg_kj_mol"] < -2.0
    assert enr["detected_solvation_role"] == "competitive_inner_sheath_sei_former"

    kin = compute_cage_exchange_kinetics(
        traj, box_l, cation_indices=[0], donor_indices=[1, 2, 3], dt_frame_ps=0.5, r_cutoff_a=2.8
    )
    assert kin["tau_exchange_ps"] > 1.0
    assert kin["vehicularity_index"] > 0.0

    ready = score_fast_charging_sei_readiness(
        {
            "cip_frac": 0.28,
            "agg_total_frac": 0.12,
            "agg_network_frac": 0.0,
            "percolation_p_inf": 0.18,
            "cne_ionicity": 0.72,
        },
        enr,
        {"tau_exchange_ps": 45.0},
        fluorine_index=0.10,
    )
    assert ready["solvation_regime"] == "GOLDILOCKS_FAST_CHARGE_ANION_SEI"
    assert ready["fast_charge_sei_readiness_100"] >= 75.0


def test_end_to_end_3d_candidate_solvation() -> None:
    desc = decode_candidate_index(95526)
    rep, curves = analyze_candidate_solvation_3d(
        desc,
        rank=1,
        sigma_298k=35.65,
        sigma_253k=18.35,
        n_ion_pairs=4,
        equil_frames=4,
        prod_frames=8,
        substeps=5,
        seed=21,
    )
    assert (
        rep.ssip_frac + rep.cip_frac + rep.agg_dimer_frac + rep.agg_network_frac
    ) == pytest.approx(1.0, abs=1e-3)
    assert rep.n_atoms > 40
    assert len(curves["r_a"]) == len(curves["g_li_backbone"])
