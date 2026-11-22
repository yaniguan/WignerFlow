"""Pytest session isolation for global PyTorch dtype, thread count, and warning caches."""

import pytest
import torch

from equitraj.layers import _WARNED_LEGACY_ATTENTION


@pytest.fixture(autouse=True)
def _restore_global_torch_state():
    prev_dtype = torch.get_default_dtype()
    prev_threads = torch.get_num_threads()
    prev_warned = set(_WARNED_LEGACY_ATTENTION)
    try:
        yield
    finally:
        torch.set_default_dtype(prev_dtype)
        torch.set_num_threads(prev_threads)
        _WARNED_LEGACY_ATTENTION.clear()
        _WARNED_LEGACY_ATTENTION.update(prev_warned)
