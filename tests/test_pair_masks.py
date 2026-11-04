import numpy as np
import pytest
import torch
from equitraj.prior import RepulsionPrior, overlap_penalty, estimate_rmin
from equitraj.monitor import StabilityMonitor
from equitraj.rollout import min_pair_distance, rollout


def inputs():
    x = torch.tensor([[[0., 0, 0], [.1, 0, 0], [3., 0, 0]]], requires_grad=True)
    mask = torch.zeros(3, 3, dtype=torch.bool)
    mask[0, 1] = mask[1, 0] = True
    return x, mask, torch.zeros(3, dtype=torch.long), torch.tensor([1., 2., 3.])


def test_prior_penalty_and_gradient():
    x, mask, types, masses = inputs()
    prior = RepulsionPrior(1, rmin=1., excluded_pairs=mask)
    corr = prior.correction(x, None, types, masses)
    assert torch.count_nonzero(corr) == 0
    assert overlap_penalty(x, None, types, prior.r0(), excluded_pairs=mask) == 0
    # A nonexcluded clash still produces differentiable momentum-conserving correction.
    x = x.detach().clone()
    x[:, 2, 0] = .4
    x.requires_grad_()
    corr = prior.correction(x, None, types, masses)
    torch.testing.assert_close((corr * masses[None, :, None]).sum(1), torch.zeros(1, 3), atol=1e-6, rtol=0)
    loss = overlap_penalty(x, None, types, prior.r0(), excluded_pairs=mask)
    loss.backward()
    assert loss > 0 and torch.isfinite(x.grad).all()


def test_monitor_reports_only_nonexcluded():
    x, mask, types, masses = inputs()
    assert min_pair_distance(x, None, excluded_pairs=mask).item() == pytest.approx(2.9)
    monitor = StabilityMonitor(types, masses, excluded_pairs=mask)
    assert not monitor.check(1, x.detach(), torch.zeros_like(x)).item()
    x = x.detach().clone()
    x[:, 2, 0] = .2
    assert monitor.check(2, x, torch.zeros_like(x)).item()
    info = monitor.diagnostics()["crashes"][0]
    assert [0, 1] not in [p[:2] for p in info["pairs"]]
    assert info["reason"] == "overlap"


def test_estimation_and_validation():
    x, mask, types, masses = inputs()
    class Dataset:
        pos = [x.detach().numpy()]
        boxes = [torch.eye(3)]
        periodic = False
        elements = ["X"]
    ds = Dataset()
    ds.types = types
    assert estimate_rmin(ds, percentile=0).item() == pytest.approx(.1)
    assert estimate_rmin(ds, percentile=0, excluded_pairs=mask).item() == pytest.approx(2.9)
    with pytest.raises(ValueError, match="symmetric"):
        min_pair_distance(x, None, excluded_pairs=torch.ones(3, 3, dtype=torch.bool).triu(1))
    with pytest.raises(ValueError, match="boolean"):
        min_pair_distance(x, None, excluded_pairs=torch.zeros(3, 3))


def test_rollout_exclusions_opt_in():
    x, mask, types, masses = inputs()
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
        def predict(self, x, v, *args, **kwargs):
            return torch.zeros_like(x[:, -1]), v[:, -1]
    args = (Model(), x.detach()[:, None], torch.zeros_like(x)[:, None], types, None, masses, 1)
    assert rollout(*args, check_every=1)["crash_step"].item() == 1
    assert rollout(*args, check_every=1, excluded_pairs=mask)["crash_step"].item() == -1


def test_empty_active_pairs_and_velocity_damping():
    x, mask, types, masses = inputs()
    all_excluded = torch.ones(3, 3, dtype=torch.bool)
    assert torch.isinf(min_pair_distance(x, None, excluded_pairs=all_excluded)).all()
    loss = overlap_penalty(x, None, types, torch.ones(1, 1), all_excluded)
    loss.backward()
    assert torch.count_nonzero(x.grad) == 0
    prior = RepulsionPrior(1, rmin=1, damp_velocity=1, excluded_pairs=mask)
    v = torch.zeros_like(x)
    v[:, 0, 0] = 1
    v[:, 1, 0] = -1
    torch.testing.assert_close(prior.damp(x, v, None, types, masses), v)
    # Exclusions do not add weights to existing state dicts.
    assert "excluded_pairs" not in prior.state_dict()
