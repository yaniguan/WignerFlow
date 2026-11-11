"""Unit tests for OpenForceField protein-ligand X-ray pocket MD benchmark (`equitraj/plb_bench.py`)."""
from pathlib import Path

import pytest

from equitraj.plb_bench import benchmark_binding_pose_and_potency, compute_pose_auroc


def test_compute_pose_auroc_exact_separation():
    natives = [{"comp_score": -2.8, "tau_ramd_ps": 1.15, "vdw_clashes": 0}]
    decoys = [{"comp_score": -0.4, "tau_ramd_ps": 0.65, "vdw_clashes": 22}]
    res = compute_pose_auroc(natives, decoys)
    assert res["pose_discrimination_auroc"] == 1.0
    assert res["comp_score_only_auroc"] == 1.0


def test_benchmark_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        benchmark_binding_pose_and_potency(tmp_path / "nonexistent")


def test_benchmark_on_synthetic_plb_fixture(tmp_path: Path):
    chem = pytest.importorskip("rdkit.Chem")
    allchem = pytest.importorskip("rdkit.Chem.AllChem")
    pytest.importorskip("yaml")

    t_dir = tmp_path / "cdk2"
    (t_dir / "01_protein" / "crd").mkdir(parents=True)
    (t_dir / "02_ligands").mkdir(parents=True)
    (t_dir / "00_data").mkdir(parents=True)

    # Create a clash-free 12-atom heavy pocket shell at radius ~3.8 A around origin
    pdb_lines = []
    coords = [
        (3.8, 0.0, 0.0, "O"), (-3.8, 0.0, 0.0, "N"), (0.0, 3.8, 0.0, "C"), (0.0, -3.8, 0.0, "C"),
        (0.0, 0.0, 3.8, "O"), (0.0, 0.0, -3.8, "C"), (2.8, 2.8, 0.0, "C"), (-2.8, -2.8, 0.0, "C"),
        (2.8, -2.8, 0.0, "O"), (-2.8, 2.8, 0.0, "N"),
    ]
    for idx, (x, y, z, el) in enumerate(coords, 1):
        pdb_lines.append(
            f"ATOM  {idx:5d}  {el:<3s} ALA A   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}"
        )
    pdb_lines.append("END")
    (t_dir / "01_protein" / "crd" / "protein.pdb").write_text("\n".join(pdb_lines), encoding="utf-8")

    mol = chem.AddHs(chem.MolFromSmiles("CCO"))
    mol.SetProp("_Name", "lig_test")
    allchem.EmbedMolecule(mol, randomSeed=7)
    w = chem.SDWriter(str(t_dir / "02_ligands" / "ligands.sdf"))
    w.write(mol)
    w.close()

    (t_dir / "00_data" / "ligands.yml").write_text(
        "lig_test:\n  measurement:\n    type: Ki\n    value: 25.0\n    unit: nM\n", encoding="utf-8"
    )

    out = benchmark_binding_pose_and_potency(tmp_path, targets=("cdk2",), max_ligands_per_target=1, n_frames=12)
    assert out["n_complexes_evaluated"] == 2
    assert out["n_native_crystal_poses"] == 1
    assert out["n_perturbed_decoys"] == 1
