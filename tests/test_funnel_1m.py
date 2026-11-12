"""Unit tests for the 1,000,000-candidate unknown electrolyte chemical space funnel (`equitraj/funnel.py`)."""
import pytest
import torch

from equitraj.align import CALiSolTransportAligner
from equitraj.funnel import (
    NOVEL_ADDITIVE_SPACE_100,
    decode_candidate_index,
    run_1m_hierarchical_funnel,
)
from equitraj.model import EquiTraj


def test_1m_bijection_and_hierarchical_funnel():
    assert len(NOVEL_ADDITIVE_SPACE_100) == 100
    d0 = decode_candidate_index(0)
    d_last = decode_candidate_index(999_999)
    assert d0["candidate_id"] == 0 and d_last["candidate_id"] == 999_999
    assert d0["formula_label"] != d_last["formula_label"]

    aligner = CALiSolTransportAligner(n_heads=3, hidden=16).float().eval()
    wf_mini = EquiTraj(
        n_types=5, channels=16, lmax=1, heads=2, n_rbf=12, cutoff=4.5, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="regression", stride_fs=25.0, use_charges=True,
    ).float().eval()
    wf_mini.dx_std.fill_(0.05)
    wf_mini.v_std.fill_(0.004)
    wf_mini.out_dx_std.fill_(0.01)
    wf_mini.out_v_std.fill_(0.001)

    out = run_1m_hierarchical_funnel(
        aligner, wf_mini, n_candidates=25_000, tier2_keep=10, tier3_keep=4, top_k=3, rollout_steps_tier3=8
    )
    assert out["n_chemical_space"] == 25_000
    assert len(out["top_candidates"]) == 3
    assert out["top_candidates"][0]["final_escore"] >= out["top_candidates"][-1]["final_escore"]


def test_novel_additive_smiles_rdkit_valid_and_unique():
    chem = pytest.importorskip("rdkit.Chem")
    canonical = set()
    for add in NOVEL_ADDITIVE_SPACE_100:
        mol = chem.MolFromSmiles(add["smiles"])
        assert mol is not None, f"Invalid SMILES for {add['name']}: {add['smiles']}"
        canonical.add(chem.MolToSmiles(mol))
    assert len(canonical) == 100
