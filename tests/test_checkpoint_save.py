"""Checkpoint writes survive a missing output folder (Google Drive unmount). CPU only."""
import os
import shutil

import pytest
import torch

from equitraj.train import atomic_save


def test_atomic_save_creates_missing_parent_from_local_tmp(tmp_path, monkeypatch):
    dest = tmp_path / "drive" / "MyDrive" / "train_stable_ft" / "ckpt.pt"
    saved_to = []
    copied = []
    real_save = torch.save
    real_copy = shutil.copyfile

    def spy_save(obj, path, *args, **kwargs):
        saved_to.append(path)
        return real_save(obj, path, *args, **kwargs)

    def spy_copy(src, dst, *args, **kwargs):
        copied.append((src, dst))
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(torch, "save", spy_save)
    monkeypatch.setattr("equitraj.train.shutil.copyfile", spy_copy)

    payload = {"step": 11000, "w": torch.tensor([1.0, 2.0])}
    atomic_save(payload, str(dest), retries=3, backoff=0.0)

    assert len(saved_to) == 1
    assert saved_to[0].startswith("/tmp/")
    assert saved_to[0].endswith(".pt")
    assert not os.path.exists(saved_to[0])
    assert copied == [(saved_to[0], str(dest) + ".tmp")]
    assert dest.is_file()
    assert not os.path.exists(str(dest) + ".tmp")
    loaded = torch.load(dest, weights_only=False)
    assert loaded["step"] == 11000
    assert torch.equal(loaded["w"], payload["w"])


def test_atomic_save_retries_when_parent_is_missing(tmp_path, monkeypatch):
    dest = tmp_path / "runs" / "best.pt"
    real_makedirs = os.makedirs
    attempts = {"n": 0}
    sleeps = []

    def flaky_makedirs(path, exist_ok=False):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise FileNotFoundError(f"Parent directory {path} does not exist")
        return real_makedirs(path, exist_ok=exist_ok)

    monkeypatch.setattr("equitraj.train.os.makedirs", flaky_makedirs)
    monkeypatch.setattr("equitraj.train.time.sleep", lambda seconds: sleeps.append(seconds))

    atomic_save({"step": 3, "name": "best"}, str(dest), retries=5, backoff=0.25)

    assert attempts["n"] == 3
    assert sleeps == [0.25, 0.25]
    assert dest.is_file()
    assert not os.path.exists(str(dest) + ".tmp")
    assert torch.load(dest, weights_only=False)["step"] == 3


def test_atomic_save_replaces_existing_file(tmp_path):
    dest = tmp_path / "snapshots" / "step011000.pt"
    atomic_save({"step": 1}, str(dest), retries=2, backoff=0.0)
    atomic_save({"step": 11000}, str(dest), retries=2, backoff=0.0)
    assert torch.load(dest, weights_only=False)["step"] == 11000
    assert not os.path.exists(str(dest) + ".tmp")


def test_atomic_save_raises_drive_unmount_when_parent_stays_missing(tmp_path, monkeypatch):
    dest = tmp_path / "drive" / "MyDrive" / "train_stable_ft" / "snapshots" / "step011000.pt"

    def missing(path, exist_ok=False):
        raise FileNotFoundError(f"Parent directory {path} does not exist")

    monkeypatch.setattr("equitraj.train.os.makedirs", missing)
    monkeypatch.setattr("equitraj.train.time.sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="Drive unmount"):
        atomic_save({"step": 11000}, str(dest), retries=3, backoff=0.0)
    assert not dest.exists()
