"""Shared checkpoint loading and evaluation helpers."""
import os
import numpy as np
import torch
from equitraj.dataset import TrajWindows
from equitraj.forcefield import EquiFF
from equitraj.layers import check_attention_version
from equitraj.model import make_model
from equitraj.pair_masks import exclusions_from_bonds


def first_zero(t, c):
    """First time where c crosses zero from above, linearly interpolated (None if it never does)."""
    if len(c) == 0 or c[0] < 0:
        return None
    if c[0] == 0:
        return float(t[0])
    i = int(np.argmax(c < 0))
    if c[i] >= 0:
        return None
    return float(t[i - 1] + (t[i] - t[i - 1]) * c[i - 1] / (c[i - 1] - c[i]))


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
    model_cfg = dict(cfg["model"])
    if model_cfg.get("n_ff_types") == "auto":
        model_cfg["n_ff_types"] = 0 if ds.ff_type_names is None else len(ds.ff_type_names)
    if cfg.get("kind") == "ff":
        model = EquiFF(n_types=len(ck["elements"]), **model_cfg)
    else:
        model = make_model(n_types=len(ck["elements"]), k=cfg["k"], stride_fs=ds.stride_fs, **model_cfg)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    if getattr(ds, "particle_metadata", None) is not None and hasattr(model, "set_chemistry"):
        model.set_chemistry(
            charges=ds.charges,
            ff_types=ds.ff_types,
            bonds=ds.bonds,
            n_atoms=len(ds.types),
        )
    if getattr(model, "prior", None) is not None and ds.bonds is not None and len(ds.bonds) > 0:
        model.prior.excluded_pairs = exclusions_from_bonds(ds.bonds, len(ds.types), device=device)
    return model, cfg, ds, step
