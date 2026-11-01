"""Smoke test of the v0 release checkpoint (skipped when the file is not present)."""

import os
import sys

import pytest
import torch

CKPT = "checkpoints/argon_flow_v0.pt"


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
