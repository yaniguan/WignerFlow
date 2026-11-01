import torch as _torch

# e3nn constants.pt pickles Python slices; allow them under torch>=2.6 weights-only load.
if hasattr(_torch.serialization, "add_safe_globals"):
    _torch.serialization.add_safe_globals([slice])

