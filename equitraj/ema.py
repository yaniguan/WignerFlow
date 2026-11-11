"""Exponential Moving Average (EMA) shadow weights and TF32 matmul precision helpers.

Equivariant message-passing and flow-matching models benefit substantially from EMA shadow
weights during autoregressive rollout evaluation: averaging over noisy trajectory-unroll
updates damps high-frequency parameter oscillations that otherwise compound over 10^4 steps.
"""

from contextlib import contextmanager

import torch


class ExponentialMovingAverage:
    """Maintains an exponential moving average of trainable floating-point parameters.

    With `warmup=True`, the effective decay at update `u` (`1, 2, ...`) follows
        decay_eff = min(decay, (1 + u) / (10 + u)),
    preventing early random-initialization weights from dominating the first ~10^3 steps.
    Shadow tensors are kept in at least float32 so bfloat16/float16 training does not
    quantize small EMA increments.
    """

    def __init__(self, model, decay=0.999, warmup=True):
        if not 0.0 <= float(decay) <= 1.0:
            raise ValueError(f"decay must be in [0, 1], got {decay}")
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.num_updates = 0
        self.shadow = {}
        for name, param in self._iter_trainable(model):
            shadow_dtype = torch.float64 if param.dtype == torch.float64 else torch.float32
            self.shadow[name] = param.detach().to(dtype=shadow_dtype).clone()

    @staticmethod
    def _iter_trainable(model):
        for name, param in model.named_parameters():
            if param.requires_grad and param.is_floating_point():
                yield name, param

    def effective_decay(self):
        """Decay coefficient used on the most recent (or next) update."""
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters in-place from `model`'s current parameters."""
        self.num_updates += 1
        d = self.effective_decay()
        one_minus_d = 1.0 - d
        for name, param in self._iter_trainable(model):
            if name not in self.shadow:
                shadow_dtype = torch.float64 if param.dtype == torch.float64 else torch.float32
                self.shadow[name] = param.detach().to(dtype=shadow_dtype).clone()
            else:
                s = self.shadow[name]
                if s.device != param.device:
                    s = s.to(device=param.device)
                    self.shadow[name] = s
                s.mul_(d).add_(param.detach().to(dtype=s.dtype), alpha=one_minus_d)

    @torch.no_grad()
    def copy_to(self, model):
        """Overwrite `model`'s trainable parameters with the current EMA shadow weights."""
        for name, param in self._iter_trainable(model):
            if name in self.shadow:
                param.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))

    @contextmanager
    def average_parameters(self, model):
        """Temporarily swap `model` parameters with EMA shadow weights, restoring in `finally`."""
        backup = {name: param.detach().clone() for name, param in self._iter_trainable(model)}
        self.copy_to(model)
        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in self._iter_trainable(model):
                    if name in backup:
                        param.copy_(backup[name])

    def state_dict(self):
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": int(self.num_updates),
            "shadow": {k: v.detach().cpu().clone() for k, v in self.shadow.items()},
        }

    def load_state_dict(self, state):
        self.decay = float(state["decay"])
        self.warmup = bool(state.get("warmup", True))
        self.num_updates = int(state.get("num_updates", 0))
        loaded = state["shadow"]
        for k, v in loaded.items():
            if k in self.shadow:
                self.shadow[k] = v.detach().to(device=self.shadow[k].device, dtype=self.shadow[k].dtype).clone()
            else:
                self.shadow[k] = v.detach().clone()


def configure_matmul_precision(precision=None):
    """Configure float32 matmul / CUDA TensorFloat-32 precision (`'highest'`, `'high'`, `'medium'`).

    When `precision is None`, leaves PyTorch global flags unchanged and reports current settings.
    """
    valid = {None, "highest", "high", "medium"}
    if precision not in valid:
        raise ValueError(f"precision must be one of {sorted(x for x in valid if x is not None)}, got {precision!r}")
    if precision is not None:
        torch.set_float32_matmul_precision(precision)
    effective = torch.get_float32_matmul_precision()
    return {
        "matmul_precision": effective,
        "tf32_enabled": effective != "highest",
    }
