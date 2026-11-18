"""CPU inference API (equitraj/api.py) on the released argon checkpoint."""
import time

import pytest
import torch

from equitraj.api import MAX_ATOMS, ArgonPredictor, FrameSplitPredictor, InputError
from equitraj.bench import make_model


@pytest.fixture(autouse=True)
def float32_default():
    """Other test modules switch the default dtype to float64; reproduce that here."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


def test_api_ignores_float64_default(pred, ctx):
    torch.set_default_dtype(torch.float64)
    out = pred.predict(*ctx)
    assert torch.get_default_dtype() == torch.float64
    assert len(out["positions"]) == 1


@pytest.fixture(scope="module")
def pred():
    return ArgonPredictor(threads=1)


@pytest.fixture(scope="module")
def ctx(pred):
    d = pred.ck["demo"]
    return d["pos"], d["vel"], d["box"]


def test_frame_split_matches_equitraj_predict():
    torch.manual_seed(0)
    model = make_model(channels=16, heads=4)
    n = 32
    pos = torch.rand(1, 3, n, 3) * 13.0
    vel = torch.randn(1, 3, n, 3) * 0.003
    types, masses, box = torch.zeros(n, dtype=torch.long), torch.full((n,), 39.948), torch.eye(3)[None] * 13.0
    noise = model.com_free_noise(1, types, masses, like=vel[0], generator=torch.Generator().manual_seed(1))
    a = model.predict(pos, vel, types, box, masses, n_steps=3, noise=noise)
    b = FrameSplitPredictor(model).predict(pos, vel, types, box, masses, n_steps=3, noise=noise)
    for x, y in zip(a, b):
        torch.testing.assert_close(x, y, rtol=1e-5, atol=1e-6)


def test_single_step_on_bundled_context(pred, ctx):
    res = pred.predict(*ctx, n_steps=2, seed=3)
    n = ctx[0].shape[1]
    assert torch.tensor(res["positions"]).shape == (2, n, 3)
    assert torch.tensor(res["velocities"]).shape == (2, n, 3)
    assert res["time_fs"] == [50.0, 100.0] and res["crash_step"] is None and not res["warnings"]
    com_v = torch.tensor(res["velocities"]).mean(1)  # equal masses
    assert com_v.abs().max() < 1e-6
    assert all(60 < t < 130 for t in res["temperature_K"])


def test_seed_makes_requests_reproducible(pred, ctx):
    a, b, c = (pred.predict(*ctx, seed=s)["positions"] for s in (7, 7, 8))
    assert a == b and a != c


def test_thermostat_pulls_temperature_to_target(pred, ctx):
    hot, off, cold = (pred.predict(*ctx, n_steps=3, sampling_steps=1, thermostat_K=t) for t in (200.0, None, 30.0))
    assert hot["model"]["thermostat_K"] == 200.0 and off["model"]["thermostat_K"] is None
    assert hot["temperature_K"][-1] > off["temperature_K"][-1] > cold["temperature_K"][-1]


def test_box_as_lengths_equals_matrix(pred, ctx):
    pos, vel, box = ctx
    assert pred.predict(pos, vel, box)["positions"] == pred.predict(pos, vel, torch.diagonal(box))["positions"]


def test_density_outside_training_range_is_flagged(pred, ctx):
    pos, vel, box = ctx
    assert any("density" in w for w in pred.predict(pos, vel, box * 1.3)["warnings"])


@pytest.mark.parametrize("change, match", [
    (dict(species="Ne"), "argon only"),
    (dict(species=["Ar"] * 3), "argon only"),
    (dict(positions=lambda p: p[:2]), "shape"),
    (dict(positions=lambda p: p * float("nan")), "finite"),
    (dict(box=lambda b: b * 0.4), ">= 12"),
    (dict(box=lambda b: b + 0.1), "diagonal"),
    (dict(n_steps=0), "n_steps"),
    (dict(n_steps=21), "n_steps"),
    (dict(sampling_steps=11), "sampling_steps"),
    (dict(positions=lambda p: torch.zeros(3, MAX_ATOMS + 1, 3)), "shape|atoms"),
    (dict(positions="junk"), "numeric"),
    (dict(thermostat_K=0), "thermostat_K"),
    (dict(thermostat_K="hot"), "thermostat_K"),
])
def test_bad_requests_are_rejected(pred, ctx, change, match):
    pos, vel, box = ctx
    kw = dict(species="Ar", n_steps=1)
    args = {"positions": pos, "velocities": vel, "box": box}
    for key, value in change.items():
        if key in args:
            args[key] = value(args[key]) if callable(value) else value
        else:
            kw[key] = value
    with pytest.raises(InputError, match=match):
        pred.predict(args["positions"], args["velocities"], args["box"], **kw)


def test_one_step_cost(pred, ctx):
    t = time.process_time()
    pred.predict(*ctx, sampling_steps=1)
    assert time.process_time() - t < 10.0  # loose bound for shared runners; see scripts/bench_api.py


def test_fastapi_app(pred, ctx):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from equitraj.api import create_app

    client = TestClient(create_app(pred))
    pos, vel, box = (x.tolist() for x in ctx)
    ok = client.post("/predict", json={"positions": pos, "velocities": vel, "box": box})
    assert ok.status_code == 200 and ok.json()["time_fs"] == [50.0]
    bad = client.post("/predict", json={"positions": pos, "velocities": vel, "box": box, "species": "Xe"})
    assert bad.status_code == 422
