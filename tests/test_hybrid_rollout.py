"""Tests for resumable chunked rollout, O(3)-invariant flow uncertainty, and hybrid physics fallback."""

import pytest
import torch

o3 = pytest.importorskip("e3nn.o3")

from equitraj.hybrid import (  # noqa: E402
    HybridRollout,
    OpenMMStepper,
    chunked_rollout,
    load_rollout_state,
    predict_with_uncertainty,
    save_rollout_state,
)
from equitraj.model import EquiTraj, remove_com  # noqa: E402
from equitraj.rollout import rollout  # noqa: E402

F32 = torch.float32
F64 = torch.float64


def random_orthogonal(seed, reflect=True):
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(3, 3, generator=g, dtype=F64))
    q = q * torch.sign(torch.diagonal(r))[None, :]
    if torch.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return -q if reflect else q


def lattice_inputs(b=2, k=3, spacing=3.2, dtype=F32, seed=0):
    g = torch.Generator().manual_seed(seed)
    coords = torch.arange(2, dtype=F64) * spacing
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), -1).reshape(-1, 3)
    n = grid.shape[0]
    vel = torch.randn(b, k, n, 3, generator=g, dtype=F64) * 1e-3
    pos = grid[None, None] + torch.cumsum(vel, dim=1) * 20.0
    types = torch.zeros(n, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    return pos.to(dtype), vel.to(dtype), types, masses


def make_tiny_model(head="flow", dtype=F32, seed=0):
    torch.manual_seed(seed)
    m = EquiTraj(
        n_types=1, lmax=2, channels=16, heads=2, n_rbf=8, cutoff=4.5, k=3,
        n_spatial=1, n_fusion=1, n_denoiser=1, head=head, stride_fs=20.0,
    ).to(dtype).eval()
    m.out_dx_std.fill_(2e-3)
    m.out_v_std.fill_(2e-5)
    return m


# ---------------------------------------------------------------- resumable chunked rollout

def test_chunked_rollout_matches_uninterrupted_rollout_bit_for_bit(tmp_path):
    model = make_tiny_model(head="flow", dtype=F32)
    pos, vel, types, masses = lattice_inputs(b=2, k=3, dtype=F32, seed=3)
    therm = {"t_target": 94.4, "tau_fs": 200.0, "dt_fs": 20.0}

    ref_gen = torch.Generator().manual_seed(123)
    ref = rollout(
        model, pos, vel, types, None, masses, n_steps=10, sampling_steps=2,
        save_every=3, check_every=4, thermostat=therm, generator=ref_gen,
    )

    state_file = tmp_path / "ckpt" / "rollout_state.pt"
    seen_chunks = []
    # Phase 1: run 5 steps in chunks of 2 (stops at step 5, mid-save_every=3 cycle)
    part1 = chunked_rollout(
        model, pos, vel, types, None, masses, total_steps=5, chunk_steps=2,
        state_path=state_file, seed=123, sampling_steps=2, save_every=3,
        check_every=4, thermostat=therm, on_chunk=seen_chunks.append,
    )
    assert part1["steps_done"] == 5 and part1["chunks_run"] == 3
    assert part1["resumed_from_step"] == 0 and len(seen_chunks) == 3
    assert state_file.exists()

    loaded = load_rollout_state(state_file)
    assert loaded["context_positions"].dtype == F64
    assert loaded["steps_done"] == 5

    # Phase 2: resume from disk checkpoint up to step 10 in chunks of 4
    part2 = chunked_rollout(
        model, pos, vel, types, None, masses, total_steps=10, chunk_steps=4,
        state_path=state_file, seed=999,  # ignored on resume because rng_state is loaded!
        sampling_steps=2, save_every=3, check_every=4, thermostat=therm,
    )
    assert part2["resumed_from_step"] == 5
    assert part2["steps_done"] == 10 and part2["chunks_run"] == 2
    assert torch.equal(part2["positions"], ref["positions"])
    assert torch.equal(part2["velocities"], ref["velocities"])


def test_save_and_load_rollout_state_atomic(tmp_path):
    p = tmp_path / "nested" / "state.pt"
    payload = {"steps_done": 7, "x": torch.ones(2, 3, dtype=F64)}
    save_rollout_state(payload, p)
    got = load_rollout_state(p)
    assert got["steps_done"] == 7 and torch.equal(got["x"], payload["x"])
    # No leftover temporary files
    assert list(p.parent.iterdir()) == [p]


# ---------------------------------------------------------------- flow ensemble uncertainty

@pytest.mark.parametrize("reflect", [False, True])
def test_predict_with_uncertainty_o3_symmetry_and_single_encode(reflect):
    model = make_tiny_model(head="flow", dtype=F64)
    pos, vel, types, masses = lattice_inputs(b=2, k=3, dtype=F64, seed=9)
    n_ens = 4
    g = torch.Generator().manual_seed(21)
    noises = [model.com_free_noise(2, types, masses, generator=g, like=pos) for _ in range(n_ens)]

    encode_calls = []
    orig_encode = model.encode

    def counted_encode(*args, **kwargs):
        encode_calls.append(1)
        return orig_encode(*args, **kwargs)

    model.encode = counted_encode
    dx, v, unc = predict_with_uncertainty(
        model, pos, vel, types, None, masses, n_ensembles=n_ens, n_steps=2, noises=noises
    )
    assert len(encode_calls) == 1  # backbone runs once regardless of n_ensembles!
    assert unc["dx_std"].shape == (2, pos.shape[2])
    assert (unc["dx_std_max"] > 0).all()

    rot = random_orthogonal(17 if reflect else 5, reflect=reflect)
    shift = torch.tensor([6.5, -2.2, 4.1], dtype=F64)
    noises_r = [ns @ rot.T for ns in noises]
    dx_r, v_r, unc_r = predict_with_uncertainty(
        model, pos @ rot.T + shift, vel @ rot.T, types, None, masses,
        n_ensembles=n_ens, n_steps=2, noises=noises_r,
    )
    assert torch.allclose(dx_r, dx @ rot.T, atol=1e-10)
    assert torch.allclose(v_r, v @ rot.T, atol=1e-10)
    assert torch.allclose(unc_r["dx_std"], unc["dx_std"], atol=1e-10)
    assert torch.allclose(unc_r["v_std"], unc["v_std"], atol=1e-10)
    assert torch.allclose(unc_r["dx_std_max"], unc["dx_std_max"], atol=1e-10)
    assert torch.einsum("n,bnd->bd", masses, dx).abs().max() < 1e-10
    assert torch.einsum("n,bnd->bd", masses, v).abs().max() < 1e-10

    # Regression head returns zero spread cleanly
    reg = make_tiny_model(head="regression", dtype=F64)
    _, _, unc_reg = predict_with_uncertainty(reg, pos, vel, types, None, masses, n_ensembles=3)
    assert (unc_reg["dx_std"] == 0).all() and (unc_reg["v_std"] == 0).all()


# ---------------------------------------------------------------- hybrid fallback rollout

class SoftRepulsiveVerletStepper:
    """Pure-PyTorch velocity Verlet reference stepper with gentle short-range repulsion."""

    def __init__(self, dt_fs=2.0, stride_fs=20.0, r_cut=2.5, k_rep=0.02):
        self.dt_fs = dt_fs
        self.n_sub = int(round(stride_fs / dt_fs))
        self.r_cut = r_cut
        self.k_rep = k_rep
        self.calls = 0

    def _accel(self, x, masses):
        dr = x[:, None, :] - x[None, :, :]
        d = torch.linalg.norm(dr, dim=-1).clamp_min(1e-6)
        eye = torch.eye(x.shape[0], dtype=torch.bool, device=x.device)
        active = (d < self.r_cut) & (~eye)
        mag = torch.where(active, self.k_rep * (self.r_cut - d), torch.zeros_like(d))
        f = (mag[:, :, None] * (dr / d[:, :, None])).sum(dim=1)
        return f / masses[:, None].double()

    def step(self, pos, vel, box=None, types=None, masses=None):
        self.calls += 1
        x = pos.double().clone()
        v = vel.double().clone()
        dt = self.dt_fs
        a = self._accel(x, masses)
        for _ in range(self.n_sub):
            v_half = v + 0.5 * dt * a
            x = x + dt * v_half
            a = self._accel(x, masses)
            v = v_half + 0.5 * dt * a
        v = remove_com(v[None], masses.double())[0]
        return x, v.to(vel.dtype)


class FaultInjectingPredictor:
    """Wraps a calm predictor and injects specific failure modes at chosen steps/systems."""

    def __init__(self, base):
        self.base = base
        self.call_idx = 0

    def predict(self, pos, vel, types, box, masses, noise=None, n_steps=2):
        self.call_idx += 1
        dx, v = self.base.predict(pos, vel, types, box, masses, noise=noise, n_steps=n_steps)
        if self.call_idx == 2:
            # Force atom 1 onto atom 0 in system 0 -> overlap (< min_dist)
            dx = dx.clone()
            dx[0, 1] = pos[0, -1, 0] - pos[0, -1, 1] + torch.tensor([0.1, 0.0, 0.0], dtype=dx.dtype)
        elif self.call_idx == 5 and pos.shape[0] > 1:
            # Force huge velocity spike in system 1 -> temperature (> t_max)
            v = v.clone()
            v[1, 0] = torch.tensor([0.8, 0.0, 0.0], dtype=v.dtype)
        return dx, v


def test_hybrid_rollout_heals_overlap_and_temperature_spikes():
    model = make_tiny_model(head="flow", dtype=F32)
    pos, vel, types, masses = lattice_inputs(b=2, k=3, spacing=3.2, dtype=F32, seed=4)
    faulty = FaultInjectingPredictor(model)

    # Without hybrid fallback, rollout records crashes on both systems!
    plain = rollout(
        model, pos, vel, types, None, masses, n_steps=6, sampling_steps=1,
        min_dist=1.5, t_max=800.0, check_every=1, predictor=faulty,
    )
    assert (plain["crash_step"] > 0).all()

    # With HybridRollout (n_fallback_strides=2), offending proposals are rejected and healed
    faulty2 = FaultInjectingPredictor(model)
    stepper = SoftRepulsiveVerletStepper(dt_fs=2.0, stride_fs=20.0)
    hybrid = HybridRollout(
        model, stepper, n_fallback_strides=2, min_dist=1.5, t_max=800.0, predictor=faulty2
    )
    res = hybrid.rollout(pos, vel, types, None, masses, n_steps=6, sampling_steps=1, check_every=1)
    assert res["crash_step"].tolist() == [-1, -1]
    assert [(s["step"], s["system"], s["reason"]) for s in res["switches"]] == [
        (2, 0, "overlap"),
        (5, 1, "temperature"),
    ]
    assert res["fallback_steps"] == 4  # 2 strides for sys 0 + 2 strides for sys 1
    assert res["model_steps"] == 8  # 2*6 - 4 = 8
    assert abs(res["fallback_fraction"] - 4 / 12) < 1e-12
    assert stepper.calls == 4


def test_hybrid_rollout_triggers_on_uncertainty_and_energy_jump():
    model = make_tiny_model(head="flow", dtype=F32)
    model.out_dx_std.fill_(0.25)  # large spread so uncertainty threshold triggers
    pos, vel, types, masses = lattice_inputs(b=1, k=3, spacing=3.5, dtype=F32, seed=11)
    stepper = SoftRepulsiveVerletStepper(dt_fs=5.0, stride_fs=20.0)

    # 1. Uncertainty-driven fallback
    hyb_u = HybridRollout(
        model, stepper, n_fallback_strides=3, min_dist=0.5, max_dx_std=1e-4, uncertainty_ensembles=3
    )
    res_u = hyb_u.rollout(
        pos, vel, types, None, masses, n_steps=3, sampling_steps=1,
        generator=torch.Generator().manual_seed(0),
    )
    assert res_u["switches"] == [{"step": 1, "system": 0, "reason": "uncertainty"}]
    # Steps 2 and 3 have fallback_remaining > 0 for all systems -> model skipped during cooldown!
    assert res_u["fallback_steps"] == 3 and res_u["model_steps"] == 0

    # 2. Energy-jump fallback
    calm = make_tiny_model(head="regression", dtype=F32)
    calm.out_dx_std.fill_(0.5)  # large deterministic step -> big harmonic energy jump
    x_ref = pos[:, -1].double()

    def harmonic_energy(x, box):
        return 500.0 * ((x - x_ref) ** 2).sum(dim=(1, 2))

    hyb_e = HybridRollout(
        calm, stepper, n_fallback_strides=1, min_dist=0.1,
        energy_fn=harmonic_energy, max_energy_jump=1.0,
    )
    res_e = hyb_e.rollout(pos, vel, types, None, masses, n_steps=2, sampling_steps=1)
    assert res_e["switches"][0]["reason"] == "energy_jump"
    assert res_e["fallback_steps"] >= 1


def test_openmm_stepper_end_to_end():
    openmm = pytest.importorskip("openmm")
    system = openmm.System()
    nb = openmm.NonbondedForce()
    nb.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(0.8)
    for _ in range(8):
        system.addParticle(39.948)
        nb.addParticle(0.0, 0.34, 0.996)
    system.addForce(nb)
    box_nm = 1.6
    system.setDefaultPeriodicBoxVectors(
        openmm.Vec3(box_nm, 0, 0), openmm.Vec3(0, box_nm, 0), openmm.Vec3(0, 0, box_nm)
    )

    stepper = OpenMMStepper(system, dt_fs=2.0, stride_fs=20.0, platform_name="CPU")
    pos, vel, types, masses = lattice_inputs(b=1, k=3, spacing=5.0, dtype=F32, seed=2)
    box = (torch.eye(3, dtype=F32) * 16.0)[None]
    x1, v1 = stepper.step(pos[0, -1].double(), vel[0, -1], box[0], types, masses)
    assert x1.dtype == F64 and v1.dtype == F32
    assert torch.isfinite(x1).all() and torch.isfinite(v1).all()
    assert torch.einsum("n,nd->d", masses, v1).abs().max() < 1e-6
