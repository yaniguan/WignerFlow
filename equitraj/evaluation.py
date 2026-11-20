"""Shared checkpoint loading and evaluation helpers."""
import os
import numpy as np
import torch
from equitraj.dataset import TrajWindows
from equitraj.forcefield import EquiFF
from equitraj.layers import check_attention_version
from equitraj.model import EquiTraj

def first_zero(t, c):
    """First time where c crosses zero, linearly interpolated (None if it never does)."""
    i = np.argmax(c < 0)
    if c[i] >= 0:
        return None
    return t[i - 1] + (t[i] - t[i - 1]) * c[i - 1] / (c[i - 1] - c[i])


def load_model(run_dir, ckpt, device, split="test", data_dir=None):
    ckpt_path = os.path.join(run_dir, ckpt)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "cfg" in ck:
        cfg = dict(ck["cfg"])
        step = ck["step"]
    else:
        if data_dir is None:
            raise ValueError("Released checkpoints require --data_dir for ground-truth evaluation")
        check_attention_version(ck, source=ckpt_path)
        cfg = {"model": ck["model_cfg"], "k": ck["k"], "stride_steps": ck["stride_steps"]}
        step = ck["train_step"]
    if data_dir is not None:
        cfg["data_dir"] = data_dir
    ds = TrajWindows(cfg["data_dir"], split, cfg["stride_steps"], cfg["k"], m=1)
    if "stride_fs" in ck and not np.isclose(ds.stride_fs, ck["stride_fs"]):
        raise ValueError("Dataset stride does not match released checkpoint")
    if cfg.get("kind") == "ff":
        model = EquiFF(n_types=len(ck["elements"]), **cfg["model"])
    else:
        model = EquiTraj(n_types=len(ck["elements"]), k=cfg["k"], stride_fs=ds.stride_fs, **cfg["model"])
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, ds, step

