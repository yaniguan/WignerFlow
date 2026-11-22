"""Offline tests for scripts/evaluate_molecule.py helpers and molecule YAML configs."""

from pathlib import Path
import importlib.util
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_eval_mol():
    spec = importlib.util.spec_from_file_location("evaluate_molecule", ROOT / "scripts" / "evaluate_molecule.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_find_bonds_and_bond_lengths():
    mod = _load_eval_mol()
    # C-C bond at 1.52 A, non-bonded O at 4.0 A
    x0 = np.array([[0.0, 0.0, 0.0], [1.52, 0.0, 0.0], [4.0, 0.0, 0.0]], dtype=np.float64)
    bonds = mod.find_bonds(x0, ["C", "C", "O"])
    assert bonds.tolist() == [[0, 1]]
    traj = np.stack([x0, x0 + np.array([[0.0, 0.0, 0.0], [0.03, 0.0, 0.0], [0.0, 0.0, 0.0]])], axis=0)
    bl = mod.bond_lengths(traj, bonds)
    assert bl.shape == (2, 1)
    assert np.allclose(bl[:, 0], [1.52, 1.55])


def test_js_divergence_and_molecule_configs():
    mod = _load_eval_mol()
    p = np.array([[0.5, 0.5], [0.0, 0.0]])
    q = np.array([[0.0, 0.0], [0.5, 0.5]])
    assert mod.js_divergence(p, p) == 0.0
    assert np.isclose(mod.js_divergence(p, q), np.log(2.0))
    for rel in ("configs/ala2/B_S3a.yaml", "configs/aspirin/B_S3a.yaml"):
        cfg = yaml.safe_load((ROOT / rel).read_text(encoding="utf-8"))
        assert cfg["k"] == 3 and cfg["stride_steps"] == 10
