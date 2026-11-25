"""WignerFlow (`equitraj`): E(3)-equivariant autoregressive molecular dynamics without an integrator."""

import torch as _torch

# e3nn constants.pt pickles Python slices; allow them under torch>=2.6 weights-only load.
if hasattr(_torch.serialization, "add_safe_globals"):
    _torch.serialization.add_safe_globals([slice])

__version__ = "0.2.0"

__all__ = [
    "DistanceProjector",
    "EquiFF",
    "EquiTraj",
    "FastRollout",
    "PlainTraj",
    "TrajCastBaseline",
    "csvr",
    "energy_rescale",
    "evaluate_electrolyte_transport",
    "import_wmi_replica",
    "load_wmi_build_dir",
    "make_model",
    "rollout",
    "transport_calibration_loss",
]


def __getattr__(name):
    if name in ("EquiTraj", "make_model"):
        from equitraj import model as _m
        return getattr(_m, name)
    if name in ("PlainTraj", "TrajCastBaseline"):
        from equitraj import baselines as _b
        return getattr(_b, name)
    if name == "EquiFF":
        from equitraj.forcefield import EquiFF
        return EquiFF
    if name == "rollout":
        from equitraj.rollout import rollout
        return rollout
    if name == "FastRollout":
        from equitraj.fast import FastRollout
        return FastRollout
    if name == "DistanceProjector":
        from equitraj.constraints import DistanceProjector
        return DistanceProjector
    if name in ("csvr", "energy_rescale"):
        from equitraj import thermostat as _t
        return getattr(_t, name)
    if name in (
        "evaluate_electrolyte_transport",
        "import_wmi_replica",
        "load_wmi_build_dir",
        "transport_calibration_loss",
    ):
        from equitraj import electrolyte as _e
        return getattr(_e, name)
    raise AttributeError(f"module 'equitraj' has no attribute {name!r}")
