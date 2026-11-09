"""CPU checks for training resume: RNG, epoch position, step, optimizer, spike guard."""
import copy
import random
import sys

import numpy as np
import torch
import yaml

from equitraj.train import (
    EpochBatchSampler,
    SpikeGuard,
    apply_resume_rng,
    capture_rng_state,
    main,
    restore_rng_state,
)
from equitraj.trajfile import TrajWriter


def test_rng_state_roundtrip(tmp_path):
    torch.manual_seed(1)
    np.random.seed(2)
    random.seed(3)
    torch.rand(3)
    np.random.rand()
    random.random()
    state = capture_rng_state()
    if torch.cuda.is_available():
        assert "cuda" in state
    else:
        assert "cuda" not in state
    path = tmp_path / "rng.pt"
    torch.save({"rng": state}, path)
    torch.manual_seed(99)
    np.random.seed(99)
    random.seed(99)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restore_rng_state(loaded["rng"])
    a, b, c = torch.rand(4), np.random.rand(4), random.random()
    restore_rng_state(loaded["rng"])
    assert torch.equal(a, torch.rand(4))
    assert np.allclose(b, np.random.rand(4))
    assert c == random.random()


def test_spike_guard_state_roundtrip():
    guard = SpikeGuard(factor=5.0, window=20, min_history=2)
    assert guard.check(0.2, 2) and not guard.check(float("nan"), 2)
    guard.skipped = 7
    other = SpikeGuard(factor=5.0, window=20, min_history=2)
    other.load_state_dict(guard.state_dict())
    assert other.hist == guard.hist
    assert other.m == 2
    assert other.skipped == 7
    assert other.check(0.3, 2)
    assert other.hist[-1] == 0.3


def test_loader_iteration_does_not_change_global_rng():
    class Tiny(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return index

    data_gen = torch.Generator().manual_seed(0)
    loader_seed = torch.Generator().manual_seed(1)
    sampler = EpochBatchSampler(4, 2, data_gen)
    loader = torch.utils.data.DataLoader(
        Tiny(), batch_sampler=sampler, num_workers=0, generator=loader_seed)
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    first = list(next(iter(loader)))
    fresh = list(next(iter(EpochBatchSampler(4, 2, torch.Generator().manual_seed(0)))))
    assert first == fresh
    assert torch.equal(before, torch.get_rng_state())


def test_old_checkpoint_reseed_does_not_replay_opening_batch():
    n, batch_size, seed, step = 30, 2, 0, 4
    fresh = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()[:batch_size]
    gen = torch.Generator().manual_seed(seed)
    sampler = EpochBatchSampler(n, batch_size, gen)
    apply_resume_rng({}, seed, step, gen, sampler, n)
    resumed = list(next(iter(sampler)))
    expected = torch.randperm(n, generator=torch.Generator().manual_seed(seed + step)).tolist()[:batch_size]
    assert resumed == expected
    assert resumed != fresh


def test_sampler_resume_continues_the_same_epoch():
    n, batch_size = 20, 2
    gen = torch.Generator().manual_seed(0)
    sampler = EpochBatchSampler(n, batch_size, gen)
    seen = []
    for batch in sampler:
        seen.append(list(batch))
        sampler.next_batch += 1
        if sampler.next_batch == 3:
            break
    state = sampler.state_dict()
    redraw = torch.Generator()
    redraw.set_state(state["epoch_gen_state"].clone())
    perm = torch.randperm(n, generator=redraw).tolist()
    assert seen[0] == perm[:batch_size]
    assert seen[2] == perm[2 * batch_size:3 * batch_size]

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    gen2 = torch.Generator().manual_seed(99)
    sampler2 = EpochBatchSampler(n, batch_size, gen2)
    apply_resume_rng(
        {"rng": capture_rng_state(), "loader": state}, 0, 3, gen2, sampler2, n)
    resumed = list(next(iter(sampler2)))
    assert resumed == perm[3 * batch_size:4 * batch_size]
    assert resumed != seen[0]


def _opt_state_equal(left, right):
    assert set(left["state"]) == set(right["state"])
    for key in left["state"]:
        assert set(left["state"][key]) == set(right["state"][key])
        for name in left["state"][key]:
            a, b = left["state"][key][name], right["state"][key][name]
            if torch.is_tensor(a):
                assert torch.equal(a.cpu(), b.cpu())
            else:
                assert a == b


def _write_traj(path, n_frames):
    writer = TrajWriter(path, 2, [39.948, 39.948], ["Ar", "Ar"],
                        dict(complete=True, dt_fs=1.0, save_every=2, periodic=True))
    for t in range(n_frames):
        pos = np.zeros((2, 3), dtype=np.float32)
        pos[0] = (0.04 * t, 0.01 * t, 0.0)
        pos[1] = (0.22 + 0.03 * t, 0.08, 0.02 * (t % 5))
        vel = np.zeros((2, 3), dtype=np.float32)
        vel[0, 0] = 0.01
        vel[1, 0] = 0.008
        vel[1, 1] = 0.001 * ((t % 3) - 1)
        writer.add(pos, vel, np.zeros((2, 3)), 0.0, 0.1, np.eye(3) * 2.0, t * 0.002)
    writer.close()


def _config(data_dir, out_dir, steps):
    return {
        "name": "resume_test",
        "seed": 0,
        "data_dir": str(data_dir),
        "stride_steps": 2,
        "k": 2,
        "model": {"head": "regression", "lmax": 1, "channels": 4, "heads": 2, "n_rbf": 4,
                  "cutoff": 4.0, "n_spatial": 1, "n_fusion": 1},
        "train": {"batch_size": 2, "lr": 1.0e-3, "steps": steps, "warmup": 1, "noise_sigma": 0.01,
                  "unroll": [[0, 1]], "backprop": "truncated", "workers": 0, "ckpt_minutes": 10000,
                  "loss_spike_factor": 5, "loss_spike_window": 50},
        "eval": {"every": 1000, "val_batches": 1, "rollout_steps": 1, "rollout_starts": 1, "sampling_steps": 1},
        "out_dir": str(out_dir),
    }


def test_train_resume_restores_step_optimizer_guard_and_batch_order(tmp_path, monkeypatch):
    # test_model.py sets the process default to float64. Training data is float32.
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        _train_resume(tmp_path, monkeypatch)
    finally:
        torch.set_default_dtype(previous_dtype)


def _train_resume(tmp_path, monkeypatch):
    data_dir, out_dir = tmp_path / "data", tmp_path / "run"
    data_dir.mkdir()
    _write_traj(data_dir / "traj_000.h5", 24)
    _write_traj(data_dir / "traj_001.h5", 8)
    (data_dir / "config.yaml").write_text(yaml.safe_dump(
        {"split": {"train": [0], "val": [1], "test": []}}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(_config(data_dir, out_dir, steps=4)))

    seen = []
    real_iter = EpochBatchSampler.__iter__

    def record(self):
        for batch in real_iter(self):
            seen.append(list(batch))
            yield batch

    monkeypatch.setattr(EpochBatchSampler, "__iter__", record)
    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path), "--device", "cpu", "--no_tensorboard"])
    main()

    ckpt_path = out_dir / "ckpt.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt["step"] == 4
    assert ckpt["loader"]["next_batch"] == 4
    assert set(ckpt["rng"]) >= {"torch", "numpy", "python"}
    assert len(ckpt["spike_guard"]["hist"]) == 4
    assert ckpt["spike_guard"]["m"] == 1
    assert ckpt["opt"]["state"]
    fresh = torch.randperm(ckpt["loader"]["n"], generator=torch.Generator().manual_seed(0)).tolist()[:2]
    assert seen[0] == fresh

    redraw = torch.Generator()
    redraw.set_state(ckpt["loader"]["epoch_gen_state"].clone())
    perm = torch.randperm(ckpt["loader"]["n"], generator=redraw).tolist()
    expected = perm[8:10]
    assert expected != fresh
    saved_opt = copy.deepcopy(ckpt["opt"])
    saved_hist = list(ckpt["spike_guard"]["hist"])
    ckpt["spike_guard"]["skipped"] = 7
    torch.save(ckpt, ckpt_path)

    loaded = {}
    real_load = torch.optim.Optimizer.load_state_dict

    def spy_load(self, state_dict):
        loaded["opt"] = copy.deepcopy(state_dict)
        return real_load(self, state_dict)

    monkeypatch.setattr(torch.optim.Optimizer, "load_state_dict", spy_load)
    config_path.write_text(yaml.safe_dump(_config(data_dir, out_dir, steps=6)))
    seen.clear()
    main()

    _opt_state_equal(loaded["opt"], saved_opt)
    assert seen[0] == expected
    assert seen[0] != fresh
    resumed = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert resumed["step"] == 6
    assert resumed["spike_guard"]["skipped"] == 7
    assert resumed["spike_guard"]["hist"][:4] == saved_hist
    assert len(resumed["spike_guard"]["hist"]) == 6
    assert resumed["loader"]["next_batch"] == 6
