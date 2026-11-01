"""Small CPU coverage for the opt-in inference path."""
import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from equitraj.neighbors import VerletList, brute_force, cell_list
from equitraj.fast import FastRollout
from equitraj.model import EquiTraj
from equitraj.distill import distillation_loss


def pairs(result):
    return set(zip(result[0].tolist(), result[1].tolist()))


@pytest.mark.parametrize('periodic', ['none', 'orthogonal', 'triclinic', 'small'])
def test_neighbors(periodic):
    torch.manual_seed(10)
    box = torch.eye(3) * (7 if periodic == 'small' else 18)
    if periodic == 'triclinic': box[1, 0] = 2
    if periodic == 'none': box = None
    pos = torch.rand(45, 3) * (7 if periodic == 'small' else 18)
    cutoff = 2.5
    nl = VerletList(cutoff, skin=1)
    assert pairs(cell_list(pos, box, cutoff)) == pairs(brute_force(pos, box, cutoff))
    assert pairs(nl(pos, box)) == pairs(brute_force(pos, box, cutoff))
    first = nl.rebuilds
    pos = pos + torch.randn_like(pos) * .02
    assert pairs(nl(pos, box)) == pairs(brute_force(pos, box, cutoff))
    assert nl.rebuilds == first
    pos[0, 0] += .7
    assert pairs(nl(pos, box)) == pairs(brute_force(pos, box, cutoff))
    assert nl.rebuilds == first + 1


def test_verlet_interval_box_and_crossing():
    pos = torch.tensor([[.1, 0, 0], [17.9, 0, 0]])
    box = torch.eye(3) * 18
    nl = VerletList(2, rebuild_every=2)
    nl(pos, box); nl(pos, box)
    assert nl.rebuilds == 1
    nl(pos, box)
    assert nl.rebuilds == 2
    box = box * 1.1
    assert pairs(nl(pos, box)) == pairs(brute_force(pos, box, 2))
    assert nl.rebuilds == 3
    pos[0, 0] += 19.8
    assert pairs(nl(pos, box)) == pairs(brute_force(pos, box, 2))


def inputs():
    torch.manual_seed(3)
    m = EquiTraj(1, channels=8, heads=2, lmax=1, n_rbf=4, k=2,
                 n_spatial=1, n_fusion=1, n_denoiser=1, cutoff=3).float().eval()
    x = torch.randn(2, 2, 5, 3, dtype=torch.float32)
    v = torch.randn_like(x) * .01
    t = torch.zeros(5, dtype=torch.long)
    mass = torch.ones(5, dtype=torch.float32)
    return m, x, v, t, mass


@pytest.mark.parametrize('steps', [1, 2, 10])
def test_fast_matches_predict_and_rollout(steps):
    from equitraj.rollout import rollout
    m, x, v, t, mass = inputs()
    fast = FastRollout(m, sampling_steps=steps)
    noise = m.com_free_noise(2, t, mass, like=x)
    expected = m.predict(x, v, t, None, mass, n_steps=steps, noise=noise)
    actual = fast.predict(x, v, t, None, mass, noise=noise)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    torch.manual_seed(99)
    ref = rollout(m, x, v, t, None, mass, 2, sampling_steps=steps,
                  min_dist=0, t_max=None, check_every=2)
    torch.manual_seed(99)
    result = fast.rollout(x, v, t, None, mass, 2)
    torch.testing.assert_close(result['positions'], ref['positions'], atol=5e-5, rtol=5e-5)
    torch.testing.assert_close(result['velocities'], ref['velocities'], atol=5e-5, rtol=5e-5)


def test_bf16_cpu():
    m, x, v, t, mass = inputs()
    fast = FastRollout(m, precision='bf16', sampling_steps=1)
    result = fast.predict(x, v, t, None, mass)
    assert all(torch.isfinite(a).all() for a in result)


def test_fp8_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torchao', None)
    m, x, v, t, mass = inputs()
    with pytest.warns(RuntimeWarning, match='fp8 unavailable'):
        fast = FastRollout(m, precision='fp8', sampling_steps=1)
    assert fast.effective_precision == 'fp32'
    assert all(torch.isfinite(a).all() for a in fast.predict(x, v, t, None, mass))


def test_distillation_backward():
    teacher, x, v, t, mass = inputs()
    student = copy.deepcopy(teacher).train()
    loss = distillation_loss(student, teacher, x, v, t, None, mass, teacher_steps=2,
                             supervised_weight=.1)
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in student.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert all(p.grad is None for p in teacher.parameters())


def test_micro_cli(tmp_path):
    output = tmp_path / 'micro.jsonl'
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / 'scripts/bench_speed.py'), '--micro',
                    '--sizes', '32', '--steps', '2', '--cutoff', '2', '--out', str(output)],
                   cwd=root, check=True, capture_output=True, text=True, timeout=60)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 3
    assert all('error' not in r and r['ms_per_call'] > 0 for r in rows)


def test_padded_graph_matches_unpadded_cpu():
    m, x, v, t, mass = inputs()
    noise = m.com_free_noise(2, t, mass, like=x)
    expected = m.predict(x, v, t, None, mass, n_steps=1, noise=noise)
    fast = FastRollout(m, sampling_steps=1, cuda_graphs=True, edge_capacity=128)
    with pytest.warns(RuntimeWarning, match='require CUDA'):
        actual = fast.predict(x, v, t, None, mass, noise=noise)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    # Fusion reuses the newest spatial pairs without aging their Verlet cache twice.
    assert all(nl.age == 1 for nl in fast._lists.values())
