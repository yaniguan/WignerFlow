import pytest
import torch
from equitraj.constraints import DistanceProjector


def test_unequal_mass_dimer():
    x = torch.tensor([[[0., 0, 0], [2., 0, 0]]], dtype=torch.float64)
    v = torch.tensor([[[1., 2, 0], [-2., 0, 0]]], dtype=torch.float64)
    m = torch.tensor([1., 3.], dtype=torch.float64)
    xp, vp, info = DistanceProjector([[0, 1, 1.]])(x, v, m)
    torch.testing.assert_close((xp * m[None, :, None]).sum(1), (x * m[None, :, None]).sum(1))
    torch.testing.assert_close((vp * m[None, :, None]).sum(1), (v * m[None, :, None]).sum(1))
    assert (xp[:, 0] - xp[:, 1]).norm().item() == pytest.approx(1)
    assert ((vp[:, 0] - vp[:, 1]) * (xp[:, 0] - xp[:, 1])).sum().abs() < 1e-10
    assert info["max_position_correction_A"] > 0
    assert x[0, 1, 0] == 2


def test_coupled_and_periodic():
    x = torch.tensor([[[0., 0, 0], [1.3, .1, 0], [2.7, 0, 0]]], dtype=torch.float64)
    v = torch.randn_like(x)
    m = torch.tensor([1., 2., 3.])
    xp, vp, info = DistanceProjector([[0, 1, 1.], [1, 2, 1.]])(x, v, m)
    assert info["distance_residual_A"] < 1e-8
    assert info["velocity_residual_A_per_fs"] < 1e-10
    box = torch.eye(3, dtype=x.dtype)[None] * 10
    x = torch.tensor([[[.1, 0, 0], [9.7, 0, 0]]], dtype=x.dtype)
    xp, _, _ = DistanceProjector([[0, 1, .5]])(x, torch.zeros_like(x), torch.ones(2), box)
    from equitraj.neighbors import minimum_image
    assert minimum_image((xp[:, 0] - xp[:, 1])[:, None], box).norm().item() == pytest.approx(.5)


def test_failures():
    x = torch.tensor([[[0., 0, 0], [2., 0, 0], [4., 0, 0]]])
    with pytest.raises(RuntimeError, match="converge"):
        DistanceProjector([[0, 1, 1], [1, 2, 1]], max_iterations=1)(x, x, torch.ones(3))
    with pytest.raises(ValueError, match="masses"):
        DistanceProjector([[0, 1, 1]])(x, x, torch.tensor([0., 1, 1]))
    with pytest.raises(ValueError):
        DistanceProjector([[0, 0, 1]])
    with pytest.raises(RuntimeError, match="coincident"):
        DistanceProjector([[0, 1, 1]])(x * 0, x * 0, torch.ones(3))


def test_rollout_hook_and_temperature_dof():
    from equitraj.rollout import rollout, temperature
    class Model(torch.nn.Module):
        stride_fs = 1
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(()))
        def predict(self, x, v, *args, **kwargs):
            return torch.zeros_like(x[:, -1]), v[:, -1]
    x = torch.tensor([[[[0., 0, 0], [2., 0, 0], [4., 0, 0]]]])
    v = torch.ones_like(x) * .01
    masses = torch.ones(3)
    result = rollout(Model(), x, v, torch.zeros(3, dtype=torch.long),
                     None, masses, 2, check_every=1,
                     constraint_projector=DistanceProjector([[0, 1, 1.]]))
    assert len(result["diagnostics"]["constraint_projection"]) == 2
    assert result["positions"][0, 0, 1, 0] - result["positions"][0, 0, 0, 0] == pytest.approx(1)
    expected = temperature(result["velocities"][0], masses, n_constraints=1).item()
    assert result["diagnostics"]["series"]["T"][0][0] == pytest.approx(expected)
