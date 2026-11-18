"""Smoke and SHA-256 integrity tests of the committed checkpoints."""

import hashlib
import os
from pathlib import Path
import sys

import pytest
import torch

CKPT = "checkpoints/argon_flow_v0.pt"
CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def verify_sha256_files(directory: Path) -> list[str]:
    """Verify every *.sha256 file in `directory` against its target binary file."""
    sha_files = sorted(directory.glob("*.sha256"))
    assert sha_files, f"No .sha256 files found in {directory}"
    checked = []
    for sha_path in sha_files:
        line = sha_path.read_text(encoding="utf-8").strip()
        parts = line.split()
        assert len(parts) == 2, f"Malformed sha256 line in {sha_path}: {line!r}"
        expected_hex, filename = parts[0], parts[1].lstrip("*")
        target = directory / filename
        assert target.is_file(), f"{sha_path.name} references missing file {target}"
        actual_hex = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual_hex == expected_hex, (
            f"SHA-256 mismatch for {filename}: expected {expected_hex}, got {actual_hex}"
        )
        checked.append(filename)
    return checked


def test_checkpoint_sha256_files_match(tmp_path):
    checked = verify_sha256_files(CKPT_DIR)
    assert "argon_flow_v0.pt" in checked
    assert "argon_flow_v1_cpu_reduced.pt" in checked

    # Verify tamper detection fails immediately
    fake_pt = tmp_path / "dummy.pt"
    fake_pt.write_bytes(b"original-weights")
    good_hex = hashlib.sha256(b"original-weights").hexdigest()
    (tmp_path / "dummy.pt.sha256").write_text(f"{good_hex}  dummy.pt\n", encoding="utf-8")
    assert verify_sha256_files(tmp_path) == ["dummy.pt"]

    fake_pt.write_bytes(b"corrupted-weights")
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        verify_sha256_files(tmp_path)


@pytest.mark.skipif(not os.path.exists(CKPT), reason="release checkpoint not present")
def test_release_checkpoint_loads_and_steps():
    pytest.importorskip("e3nn")
    pytest.importorskip("matplotlib")  # demo.py imports equitraj.plots
    sys.path.insert(0, "scripts")
    from demo import load_release  # noqa: PLC0415
    from equitraj.rollout import rollout  # noqa: PLC0415

    torch.set_default_dtype(torch.float32)  # other tests switch to float64 and may leave it on
    model, ck = load_release(CKPT, "cpu")
    d = ck["demo"]
    out = rollout(model, d["pos"][None], d["vel"][None], ck["types"], d["box"][None], ck["masses"], 2, sampling_steps=2)
    assert out["positions"].shape[0] == 2
    assert torch.isfinite(out["positions"]).all()
