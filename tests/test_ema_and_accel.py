"""Tests for EMA shadow weights, Verlet unroll training graph reuse, and TF32 matmul configuration."""

import copy
import io

import pytest
import torch

o3 = pytest.importorskip("e3nn.o3")

from equitraj.ema import ExponentialMovingAverage, configure_matmul_precision  # noqa: E402
from equitraj.model import EquiTraj  # noqa: E402
from equitraj.train import accumulate_gradients, training_loss  # noqa: E402

F32 = torch.float32
F64 = torch.float64


# ---------------------------------------------------------------- ExponentialMovingAverage

def test_ema_warmup_shadow_math_and_exception_safe_swap():
    lin = torch.nn.Linear(4, 2, dtype=F64)
    with torch.no_grad():
        lin.weight.fill_(1.0)
        lin.bias.fill_(0.0)

    ema = ExponentialMovingAverage(lin, decay=0.99, warmup=True)
    with torch.no_grad():
        lin.weight.fill_(3.0)
        lin.bias.fill_(-2.0)

    # First update (u = 1): effective decay = min(0.99, (1+1)/(10+1)) = 2/11
    ema.update(lin)
    d1 = 2.0 / 11.0
    expected_w = d1 * 1.0 + (1.0 - d1) * 3.0
    expected_b = d1 * 0.0 + (1.0 - d1) * (-2.0)
    assert abs(ema.effective_decay() - d1) < 1e-15
    assert torch.allclose(ema.shadow["weight"], torch.full_like(lin.weight, expected_w), atol=1e-12)
    assert torch.allclose(ema.shadow["bias"], torch.full_like(lin.bias, expected_b), atol=1e-12)

    # Context manager swaps in EMA weights and restores live weights even on exception
    with pytest.raises(RuntimeError, match="boom"):
        with ema.average_parameters(lin):
            assert torch.allclose(lin.weight, torch.full_like(lin.weight, expected_w), atol=1e-12)
            raise RuntimeError("boom")
    assert torch.allclose(lin.weight, torch.full_like(lin.weight, 3.0), atol=1e-12)
    assert torch.allclose(lin.bias, torch.full_like(lin.bias, -2.0), atol=1e-12)

    # Serialization round-trip (weights_only=True compatible)
    buf = io.BytesIO()
    torch.save(ema.state_dict(), buf)
    buf.seek(0)
    loaded_state = torch.load(buf, weights_only=True)

    lin2 = torch.nn.Linear(4, 2, dtype=F64)
    ema2 = ExponentialMovingAverage(lin2, decay=0.5, warmup=False)
    ema2.load_state_dict(loaded_state)
    assert ema2.decay == 0.99 and ema2.warmup is True and ema2.num_updates == 1
    ema2.copy_to(lin2)
    assert torch.allclose(lin2.weight, torch.full_like(lin2.weight, expected_w), atol=1e-12)


def test_ema_no_warmup_and_validation():
    for bad in (-0.1, 1.1):
        with pytest.raises(ValueError):
            ExponentialMovingAverage(torch.nn.Linear(2, 2, dtype=F32), decay=bad)

    lin = torch.nn.Linear(2, 2, bias=False, dtype=F32)
    with torch.no_grad():
        lin.weight.fill_(2.0)
    ema = ExponentialMovingAverage(lin, decay=0.75, warmup=False)
    with torch.no_grad():
        lin.weight.fill_(6.0)
    ema.update(lin)
    assert torch.allclose(ema.shadow["weight"], torch.full_like(lin.weight, 0.75 * 2.0 + 0.25 * 6.0))


# ---------------------------------------------------------------- matmul precision helper

def test_configure_matmul_precision_roundtrip():
    prev = torch.get_float32_matmul_precision()
    try:
        for mode in ("highest", "high", "medium"):
            info = configure_matmul_precision(mode)
            assert info["matmul_precision"] == mode
        info_none = configure_matmul_precision(None)
        assert info_none["matmul_precision"] == "medium"
        with pytest.raises(ValueError):
            configure_matmul_precision("invalid_mode")
    finally:
        configure_matmul_precision(prev)


# ---------------------------------------------------------------- Verlet unroll training parity

def make_unroll_batch(b=2, total_frames=7, periodic=True, seed=0):
    """27-atom jittered cubic lattice (L=12 Å so cutoff+skin < L/2) over k+m frames."""
    g = torch.Generator().manual_seed(seed)
    r = torch.arange(3, dtype=F64) * 4.0 + 2.0
    grid = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), -1).reshape(-1, 3)
    n = grid.shape[0]
    vel = torch.randn(b, total_frames, n, 3, generator=g, dtype=F64) * 2e-3
    pos0 = grid[None] + torch.randn(b, n, 3, generator=g, dtype=F64) * 0.1
    pos = pos0[:, None] + torch.cumsum(vel, dim=1) * 20.0
    box = (torch.eye(3, dtype=F64) * 12.0).expand(b, 3, 3).clone() if periodic else None
    types = torch.zeros(n, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=F64)
    return pos, vel, box, types, masses


@pytest.mark.parametrize("periodic", [False, True])
@pytest.mark.parametrize("backprop", ["truncated", "full"])
def test_verlet_unroll_training_loss_and_grads_match_bruteforce(periodic, backprop):
    torch.manual_seed(0)
    m_ref = EquiTraj(
        n_types=1, lmax=2, channels=12, heads=2, n_rbf=6, cutoff=4.2, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="regression", stride_fs=20.0,
    ).to(F64).train()
    m_ref.out_dx_std.fill_(1e-2)
    m_ref.out_v_std.fill_(1e-4)
    m_verlet = copy.deepcopy(m_ref)

    pos, vel, box, types, masses = make_unroll_batch(b=2, total_frames=7, periodic=periodic, seed=5)
    overlap_r0 = torch.full((1, 1), 3.6, dtype=F64)
    weights = [1.0, 1.5, 2.0, 2.5]

    loss_ref = training_loss(
        m_ref, pos, vel, box, types, masses, m=4, noise_sigma=0.0, backprop=backprop,
        unroll_sampling_steps=1, unroll_weights=weights, overlap_weight=0.05,
        overlap_r0=overlap_r0, verlet_skin=None,
    )
    loss_ref.backward()

    loss_v = training_loss(
        m_verlet, pos, vel, box, types, masses, m=4, noise_sigma=0.0, backprop=backprop,
        unroll_sampling_steps=1, unroll_weights=weights, overlap_weight=0.05,
        overlap_r0=overlap_r0, verlet_skin=1.0,
    )
    loss_v.backward()

    assert torch.allclose(loss_v, loss_ref, atol=1e-11, rtol=1e-11)
    for (name, p_ref), (_, p_v) in zip(m_ref.named_parameters(), m_verlet.named_parameters()):
        if p_ref.grad is not None:
            assert torch.allclose(p_v.grad, p_ref.grad, atol=1e-10, rtol=1e-10), name


def test_accumulate_gradients_with_verlet_skin_matches_full_batch():
    torch.manual_seed(2)
    m_full = EquiTraj(
        n_types=1, lmax=1, channels=8, heads=2, n_rbf=4, cutoff=4.2, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head="regression", stride_fs=20.0,
    ).to(F64).train()
    m_micro = copy.deepcopy(m_full)
    pos, vel, box, types, masses = make_unroll_batch(b=4, total_frames=6, periodic=True, seed=8)

    l_full = accumulate_gradients(
        m_full, pos, vel, box, types, masses, m=3, noise_sigma=0.0, backprop="truncated",
        unroll_sampling_steps=1, micro_batch_size=4, verlet_skin=1.0,
    )
    l_micro = accumulate_gradients(
        m_micro, pos, vel, box, types, masses, m=3, noise_sigma=0.0, backprop="truncated",
        unroll_sampling_steps=1, micro_batch_size=2, verlet_skin=1.0,
    )
    assert abs(l_full - l_micro) < 1e-12
    for (_, p1), (_, p2) in zip(m_full.named_parameters(), m_micro.named_parameters()):
        if p1.grad is not None:
            assert torch.allclose(p1.grad, p2.grad, atol=1e-11)
