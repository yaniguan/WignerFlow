"""Tests for the benchmark helpers in equitraj/bench.py (CPU, small sizes)."""

import math

import pytest
import torch

from equitraj import bench

# tiny architecture so the model tests take seconds
TINY = dict(channels=16, heads=2, n_rbf=8, cutoff=4.0, n_spatial=1, n_fusion=1, n_denoiser=1)


# ---------------------------------------------------------------- pure helpers (no e3nn needed)

@pytest.mark.parametrize("reflect", [False, True])
def test_random_orthogonal(reflect):
    g = torch.Generator().manual_seed(1)
    for _ in range(5):
        r = bench.random_orthogonal(reflect, generator=g)
        assert torch.allclose(r @ r.T, torch.eye(3, dtype=torch.float64), atol=1e-12)
        assert abs(torch.det(r).item() - (-1.0 if reflect else 1.0)) < 1e-12


def test_make_liquid_shapes_density_and_frames():
    inp = bench.make_liquid(64, k=3, batch=2, stride_fs=50.0, dtype=torch.float64)
    assert inp["pos"].shape == (2, 3, 64, 3) and inp["vel"].shape == (2, 3, 64, 3)
    assert inp["types"].shape == (64,) and inp["masses"].shape == (64,)
    length = inp["box"][0, 0, 0].item()
    assert abs(64 / length**3 - bench.ARGON_DENSITY) < 1e-9
    assert 6.0 < 0.5 * length  # the B_S3a cutoff fits in the 64-atom box
    # each frame is the previous one moved by the new velocity times the stride
    assert torch.allclose(inp["pos"][:, 1:] - inp["pos"][:, :-1], inp["vel"][:, 1:] * 50.0, atol=1e-12)
    assert bench.make_liquid(10, periodic=False)["box"] is None


def test_transform_inputs_rotates_box_and_vectors():
    inp = bench.make_liquid(27, k=2, dtype=torch.float64)
    r = bench.random_orthogonal(True, generator=torch.Generator().manual_seed(0))
    out = bench.transform_inputs(inp, r, torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64))
    assert torch.allclose(out["vel"], inp["vel"] @ r.T)
    assert torch.allclose(out["box"], inp["box"] @ r.T)
    assert torch.allclose(out["pos"] - torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), inp["pos"] @ r.T)


def test_jump_stats_smooth_vs_step():
    r = 1.0 + 1e-3 * torch.arange(40, dtype=torch.float64)
    n_edges = (r > 1.0205).long()  # pretend the graph changes in the middle
    smooth = torch.stack([torch.sin(r), torch.cos(r)], -1)
    s = bench.jump_stats(smooth, n_edges)
    assert s["n_crossing_steps"] == 1 and s["crossing_to_max_smooth"] < 1.5
    step = smooth + 0.5 * n_edges[:, None].double()
    s = bench.jump_stats(step, n_edges)
    assert s["crossing_to_max_smooth"] > 100
    assert abs(s["max_crossing_step"] - math.hypot(*(step[21] - step[20]).tolist())) < 1e-12
    s = bench.jump_stats(smooth, torch.zeros(40, dtype=torch.long))
    assert s["n_crossing_steps"] == 0 and "max_crossing_step" not in s


def test_rss_helpers():
    cur = bench.current_rss_mb()
    if cur is None:
        pytest.skip("no /proc/self/statm")
    assert cur > 0 and bench.peak_rss_mb() >= 0.9 * cur


def test_cutoff_probe_only_one_edge_pair_crosses():
    base = bench.make_cutoff_probe(k=3)
    counts = [bench.count_edges(bench.place_probe(base, r), 6.0) for r in (5.9, 5.99, 6.01, 6.1)]
    assert counts[0] == counts[1] and counts[2] == counts[3]
    assert counts[1] - counts[2] == 2 * 3  # edges 0 -> probe and probe -> 0, in each of the 3 frames


# ---------------------------------------------------------------- with the model (needs e3nn)

@pytest.mark.parametrize("head", ["regression", "flow"])
@pytest.mark.parametrize("periodic", [False, True])
def test_equivariance_error_is_tiny_in_float64(head, periodic):
    pytest.importorskip("e3nn")
    model = bench.make_model(head, torch.float64, **TINY)
    inp = bench.make_liquid(27, k=3, batch=2, periodic=periodic, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    trials = [bench.equivariance_error(model, inp, bench.random_orthogonal(reflect, generator=g),
                                       torch.randn(3, generator=g, dtype=torch.float64), generator=g)
              for reflect in (False, True)]
    s = bench.summarize_trials(trials)
    assert s["dx"]["max_rel"] < 1e-10 and s["v"]["max_rel"] < 1e-10
    assert s["dx"]["mean_abs"] <= s["dx"]["max_abs"]


def test_equivariance_error_detects_broken_parity():
    """parity=False layers are SO(3)- but not O(3)-equivariant: the metric must see the reflection error."""
    pytest.importorskip("e3nn")
    model = bench.make_model("regression", torch.float64, parity=False, **TINY)
    inp = bench.make_liquid(27, k=3, periodic=False, dtype=torch.float64)
    g = torch.Generator().manual_seed(0)
    rot = bench.equivariance_error(model, inp, bench.random_orthogonal(False, generator=g))
    ref = bench.equivariance_error(model, inp, bench.random_orthogonal(True, generator=g))
    assert rot["dx"]["max_rel"] < 1e-10
    assert ref["dx"]["max_rel"] > 1e-6


def test_bench_config_in_process():
    pytest.importorskip("e3nn")
    r = bench.bench_config(27, k=2, head="flow", repeats=1, warmup=0, threads=1, sampling_steps=2, **TINY)
    assert r["finite"] and r["forward_s"] > 0 and r["backward_s"] > 0 and r["predict_s"] > 0
    assert r["edges_per_frame"] > 0 and r["params"] > 0 and r["peak_rss_mb"] > 0


def test_bench_config_isolated_reports_results_and_errors():
    pytest.importorskip("e3nn")
    ok = bench.bench_config_isolated(timeout_s=300, n_atoms=27, k=2, head="regression", repeats=1, warmup=0,
                                     threads=1, sampling_steps=1, **TINY)
    assert "error" not in ok, ok
    assert ok["forward_s"] > 0 and ok["peak_delta_mb"] is not None and ok["peak_delta_mb"] >= 0
    # a 27-atom argon box is 10.9 Å wide: a 6 Å cutoff is rejected, and the error comes back to the parent
    bad = bench.bench_config_isolated(timeout_s=300, n_atoms=27, k=2, repeats=1, warmup=0, threads=1,
                                      **dict(TINY, cutoff=6.0))
    assert "error" in bad and "cutoff" in bad["error"]


@pytest.mark.parametrize("head", ["regression", "flow"])
def test_scan_outputs_is_flat_beyond_cutoff(head):
    """Once the probe is outside the cutoff it is disconnected, so the outputs cannot depend on r."""
    pytest.importorskip("e3nn")
    model = bench.make_model(head, torch.float64, **dict(TINY, cutoff=6.0))
    base = bench.make_cutoff_probe()
    noise = model.com_free_noise(1, base["types"], base["masses"], like=base["pos"]) if head == "flow" else None
    radii = 6.0 + 1e-3 / 3 + 1e-3 * (torch.arange(12, dtype=torch.float64) - 8)
    outputs, n_edges = bench.scan_outputs(model, base, radii, noise=noise)
    assert outputs.shape[0] == 12 and torch.isfinite(outputs).all()
    s = bench.jump_stats(outputs, n_edges)
    assert s["n_crossing_steps"] == 1
    beyond = outputs[radii > 6.0]
    assert (beyond - beyond[0]).abs().max() < 1e-12
