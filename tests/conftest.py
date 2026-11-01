"""Allow e3nn's bundled constants.pt (contains Python slices) under torch>=2.6 weights-only loading."""
import torch

if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([slice])
