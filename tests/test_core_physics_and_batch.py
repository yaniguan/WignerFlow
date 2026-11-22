"""Regression tests for batched PBC shapes, hybrid chunk parity, and analysis edge cases (#91)."""

import numpy as np
import pytest
import torch

pytest.importorskip("e3nn")

from equitraj import align, analysis, funnel, solvation
from equitraj.baselines import PlainTraj
from equitraj.energy import lj_potential
from equitraj.forcefield import EquiFF, verlet_rollout
from equitraj.hybrid import HybridRollout, chunked_rollout
from equitraj.model import EquiTraj, make_model, remove_com
from equitraj.pair_masks import exclusions_from_bonds
from equitraj.rollout import max_bond_deviation, rollout, temperature


def test_max_bond_deviation_batched_boxes_and_empty_bonds():
    # System 0 has box L=10 (atoms at 0.1 and 9.1 -> wrapped bond length 1.0, dev 0.0)
    # System 1 has box L=20 (atoms at 0.1 and 9.1 -> wrapped bond length 9.0, dev 8.0)
    pos = torch.tensor([
        [[0.1, 0.0, 0.0], [9.1, 0.0, 0.0], [0.1, 1.0, 0.0]],
        [[0.1, 0.0, 0.0], [9.1, 0.0, 0.0], [0.1, 1.0, 0.0]],
    ], dtype=torch.float64)
    boxes = torch.stack([torch.eye(3, dtype=torch.float64) * 10.0, torch.eye(3, dtype=torch.float64) * 20.0])
    bonds = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
    r0 = torch.tensor([1.0, 1.0], dtype=torch.float64)

    dev = max_bond_deviation(pos, bonds, r0, box=boxes)
    assert dev.shape == (2,)
    assert dev[0].item() == pytest.approx(0.0, abs=1e-9)
    assert dev[1].item() == pytest.approx(8.0, abs=1e-9)

    empty_dev = max_bond_deviation(pos, torch.zeros((0, 2), dtype=torch.long), torch.zeros(0), box=boxes)
    assert empty_dev.shape == (2,) and torch.all(empty_dev == 0)

    # Single-atom temperature returns finite 0 instead of NaN/Inf
    t_one = temperature(torch.ones(2, 1, 3), torch.tensor([40.0]))
    assert t_one.shape == (2,) and torch.isfinite(t_one).all()


def test_lj_potential_batched_per_frame_boxes():
    # Test both B == N (2 systems, 2 atoms) and B != N (3 systems, 2 atoms)
    pos2 = torch.tensor([
        [[0.0, 0.0, 0.0], [7.4, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [7.4, 0.0, 0.0]],
    ], dtype=torch.float64)
    boxes2 = torch.stack([torch.eye(3) * 14.0, torch.eye(3) * 30.0])
    u_batch = lj_potential(pos2, boxes2)
    u0 = lj_potential(pos2[0:1], boxes2[0])
    u1 = lj_potential(pos2[1:2], boxes2[1])
    assert u_batch.shape == (2,)
    assert u_batch[0].item() == pytest.approx(u0.item(), rel=1e-12)
    assert u_batch[1].item() == pytest.approx(u1.item(), rel=1e-12)
    assert abs(u_batch[0].item() - u_batch[1].item()) > 1e-3

    pos3 = torch.cat([pos2, pos2[:1]], dim=0)
    boxes3 = torch.cat([boxes2, boxes2[:1]], dim=0)
    u3 = lj_potential(pos3, boxes3)
    assert u3.shape == (3,) and torch.isfinite(u3).all()


def test_pbc_wrapped_context_frames_and_per_graph_remove_com():
    box = (torch.eye(3, dtype=torch.float64) * 14.0)[None]
    pos_unw = torch.tensor([[
        [[13.8, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[14.1, 1.0, 1.0], [2.1, 2.0, 2.0]],
        [[14.4, 1.0, 1.0], [2.2, 2.0, 2.0]],
    ]], dtype=torch.float64)
    pos_wrap = pos_unw % 14.0
    vel = torch.full_like(pos_unw, 0.01)
    types = torch.zeros(2, dtype=torch.long)
    masses = torch.tensor([40.0, 40.0], dtype=torch.float64)

    for arch in ("equi", "plain"):
        m = make_model(n_types=1, k=3, stride_fs=10.0, arch=arch, lmax=1, channels=8, heads=2,
                       n_rbf=8, cutoff=5.0, n_spatial=1, n_fusion=1, n_denoiser=1, head="regression").double().eval()
        dx_u, v_u = m.predict(pos_unw, vel, types, box, masses)
        dx_w, v_w = m.predict(pos_wrap, vel, types, box, masses)
        assert torch.allclose(dx_u, dx_w, atol=1e-10)
        assert torch.allclose(v_u, v_w, atol=1e-10)

    # Missing bonds when max_bond_hops > 0 raises a clear ValueError
    m_bonds = EquiTraj(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=5.0, k=3,
                       n_spatial=1, n_fusion=1, head="regression", max_bond_hops=2).double().eval()
    with pytest.raises(ValueError, match="bonds required"):
        m_bonds.predict(pos_unw, vel, types, box, masses)
    dx_ok, _ = m_bonds.predict(pos_unw, vel, types, box, masses, bonds=[])
    assert torch.isfinite(dx_ok).all()

    # Per-graph remove_com removes COM independently on flat (M, 3) node arrays
    vec_flat = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [-2.0, 1.0, 0.0], [-4.0, -1.0, 0.0]])
    m_flat = torch.tensor([1.0, 1.0, 2.0, 2.0])
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    cleaned = remove_com(vec_flat, m_flat, batch=batch, n_graphs=2)
    assert torch.allclose((cleaned[:2] * m_flat[:2, None]).sum(0), torch.zeros(3), atol=1e-6)
    assert torch.allclose((cleaned[2:] * m_flat[2:, None]).sum(0), torch.zeros(3), atol=1e-6)


def test_chunked_rollout_exact_parity_on_unaligned_chunks_and_hybrid_pbc_bonds(tmp_path):
    model = EquiTraj(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.5, k=2,
                     n_spatial=1, n_fusion=1, n_denoiser=1, head="flow", stride_fs=5.0).eval()
    # Force high velocities so t_max fires at step 5 (not at intermediate chunk boundary step 3!)
    pos = torch.tensor([[[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]]]])
    vel = torch.tensor([[[[0.08, 0.0, 0.0], [-0.08, 0.0, 0.0]], [[0.08, 0.0, 0.0], [-0.08, 0.0, 0.0]]]])
    types = torch.zeros(2, dtype=torch.long)
    masses = torch.tensor([40.0, 40.0])

    ref = rollout(model, pos, vel, types, None, masses, 8, sampling_steps=2, check_every=5,
                  t_max=10.0, generator=torch.Generator().manual_seed(11))
    state_file = tmp_path / "chunk_state.pt"
    chk = chunked_rollout(model, pos, vel, types, None, masses, total_steps=8, chunk_steps=3,
                          state_path=state_file, seed=11, sampling_steps=2, check_every=5, t_max=10.0)
    assert torch.equal(ref["crash_step"], chk["crash_step"])
    assert torch.equal(ref["positions"], chk["positions"])
    assert chk["diagnostics"]["crash_step"] == ref["diagnostics"]["crash_step"]

    # Resuming an already-crashed/finished state file preserves diagnostics instead of returning None
    chk_resumed = chunked_rollout(model, pos, vel, types, None, masses, total_steps=8, chunk_steps=3,
                                  state_path=state_file, seed=11, sampling_steps=2, check_every=5, t_max=10.0)
    assert chk_resumed["diagnostics"] is not None
    assert chk_resumed["diagnostics"]["crash_step"] == ref["diagnostics"]["crash_step"]

    # HybridRollout bond check wraps bonds across periodic boundaries without false trigger
    box = (torch.eye(3) * 10.0)[None]
    x_pbc = torch.tensor([[[0.1, 0.0, 0.0], [9.1, 0.0, 0.0]]], dtype=torch.float64)
    v_pbc = torch.zeros((1, 2, 3), dtype=torch.float32)
    hyb = HybridRollout(model, lambda x, v, b: (x, v), min_dist=0.5,
                        bonds=torch.tensor([[0, 1]]), bond_r0=torch.tensor([1.0]), max_bond_dev=0.2)
    reasons = hyb._detect_triggers(x_pbc, x_pbc, v_pbc, box, masses, unc=None, n_constraints=0)
    assert reasons == [None]


def test_analysis_edge_cases_and_plain_dummy_edges_and_verlet():
    # vacf clamps max_lag > len(v) instead of dividing by zero
    v = np.ones((4, 2, 3), dtype=np.float64)
    c = analysis.vacf(v, max_lag=10)
    assert len(c) == 4 and np.all(np.isfinite(c))

    # msd handles lag >= len(positions) without NaN
    p = np.zeros((2, 3, 3), dtype=np.float64)
    p[1] = 1.0
    m_vals = analysis.msd(p, [0, 1, 5])
    assert np.all(np.isfinite(m_vals)) and m_vals[2] == pytest.approx(3.0)

    # rdf excludes i == j even when idx_a and idx_b partially overlap, and handles single-ion subsets
    pos_rdf = np.array([[[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.0, 0.0, 0.0]]])
    box_rdf = np.broadcast_to(np.eye(3) * 12.0, (1, 3, 3))
    _, g_part, _ = analysis.rdf(pos_rdf, box_rdf, [0, 1], [1, 2], r_max=5.5, n_bins=20)
    assert g_part[0] == 0.0 and np.all(np.isfinite(g_part))
    _, g_single, _ = analysis.rdf(pos_rdf, box_rdf, [0], [0], r_max=5.5, n_bins=20)
    assert np.all(g_single == 0.0)
    _, g_vac_zero, _ = analysis.rdf(pos_rdf, None, [0], [2], r_max=1.0, n_bins=10)
    assert np.all(g_vac_zero == 0.0)

    # PlainBlock ignores padded dummy edges at dist == cutoff
    plain = PlainTraj(n_types=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, k=1, n_spatial=1, n_fusion=1,
                      head="regression", repulsion_prior={"enabled": False}).eval()
    h = torch.randn(2, 8)
    g_clean = {"src": torch.tensor([1]), "dst": torch.tensor([0]), "vec": torch.tensor([[2.0, 0.0, 0.0]]),
               "dist": torch.tensor([2.0])}
    g_padded = {
        "src": torch.tensor([1, 0, 0, 0]),
        "dst": torch.tensor([0, 0, 0, 0]),
        "vec": torch.tensor([[2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 0.0], [0.0, 4.0, 0.0]]),
        "dist": torch.tensor([2.0, 4.0, 4.0, 4.0]),
    }
    out_clean = plain.spatial[0](h, g_clean)
    out_padded = plain.spatial[0](h, g_padded)
    assert torch.allclose(out_clean, out_padded, atol=1e-6)

    # verlet_rollout accepts excluded_pairs and returns diagnostics
    ff = EquiFF(n_types=1, lmax=1, channels=8, heads=2, n_rbf=8, cutoff=4.0, n_spatial=1)
    x_bond = torch.tensor([[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0]]])
    excl = exclusions_from_bonds([[0, 1]], 2)
    vr = verlet_rollout(ff, x_bond, torch.zeros_like(x_bond), torch.zeros(2, dtype=torch.long), None,
                        torch.ones(2), n_steps=2, substeps=2, dt_fs=0.2, min_dist=1.2, excluded_pairs=excl)
    assert int(vr["crash_step"][0]) == -1 and "diagnostics" in vr

    # Empty candidate lists in align/solvation and small n_types validation in funnel
    aligner = align.CALiSolTransportAligner(n_heads=2, hidden=8)
    empty_pred = aligner.predict_records([])
    assert len(empty_pred["sigma_pred_mS_cm"]) == 0
    summary, _ = solvation.analyze_top15_funnel_candidates([], n_ion_pairs=2, prod_frames=4, seed=0)
    assert summary["n_candidates_analyzed"] == 0 and summary["top15_summary"]["top_dual_conductivity_sei_champion"] is None
    with pytest.raises(ValueError, match="at least 5 atom types"):
        funnel.run_1m_hierarchical_funnel(aligner, plain, n_candidates=20, tier2_keep=2, tier3_keep=1)
