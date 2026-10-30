"""Synthetic evaluator checks, including survivor conditioning and cluster uncertainty."""
import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from equitraj import multiseed


def test_cluster_bootstrap_and_missing():
    rows = [dict(start=i, **{k: float(i) for k in multiseed.METRICS}) for i in range(3) for _ in range(2)]
    rows[0]["vacf_zero_rel_err"] = None
    result = multiseed.bootstrap_summary(rows, 4000, 42)
    assert result == multiseed.bootstrap_summary(rows, 4000, 42)
    assert result["rdf_l1"]["mean"] == 1
    assert result["rdf_l1"]["ci95"] == [0., 2.]
    assert result["vacf_zero_rel_err"]["n_valid"] == 5
    for r in rows:
        r["D_rel_err"] = None
    assert multiseed.bootstrap_summary(rows)["D_rel_err"]["mean"] is None


def test_physical_metrics_identical_synthetic_reference():
    rng = np.random.default_rng(4)
    x = rng.uniform(1, 5, (24, 4, 3)).astype('f4')
    v = rng.normal(0, .001, (24, 4, 3)).astype('f4')
    box = np.eye(3, dtype='f4')*10
    ds = SimpleNamespace(pos=[x], vel=[v], f=1, k=3, stride_fs=50.,
                         periodic=True, boxes=[torch.from_numpy(box)], types=torch.zeros(4), masses=torch.ones(4))
    ref = multiseed.reference(ds, 16)
    # Explicitly use the same horizons as physical_metrics, yielding zero error.
    half = len(x)//2
    ref['T'] = multiseed.temperature(torch.from_numpy(v[half:]), ds.masses).mean().item()
    ref['r'], ref['g'], _ = multiseed.analysis.rdf(x[half:], np.tile(box, (12,1,1)), np.arange(4), np.arange(4), ref['rmax'], 200)
    c = multiseed.analysis.vacf(v[half:], ref['lag'])
    ref['zero'] = multiseed.first_zero(np.arange(ref['lag'])*50, c/c[0])
    metrics = multiseed.physical_metrics(x, v, ds, box, ref)
    assert metrics['temperature_rel_err'] == pytest.approx(0)
    assert metrics['rdf_l1'] == pytest.approx(0)
    assert metrics['D_rel_err'] == pytest.approx(0)
    assert metrics['vacf_zero_rel_err'] in (None, 0.)


def test_evaluation_pairs_crash_exclusion_and_report(tmp_path, monkeypatch):
    x = np.zeros((30, 3, 3), dtype='f4')
    x[:, :, 0] = [1, 3, 5]
    v = np.ones_like(x)*.001
    ds = SimpleNamespace(pos=[x], vel=[v], f=1, k=3, stride_fs=50., traj_ids=[9],
                         periodic=True, boxes=[torch.eye(3)*12], types=torch.zeros(3, dtype=torch.long), masses=torch.ones(3))
    model = SimpleNamespace(head_type='flow')
    monkeypatch.setattr(multiseed, 'load_model', lambda *a: (model, {'k': 3}, ds, 40))
    seen = []
    def fake_rollout(*a, **kw):
        seen.append(torch.initial_seed())
        crashed = len(seen) % 2 == 0
        return dict(crash_step=torch.tensor([10 if crashed else -1]), wall_s=.01,
                    positions=torch.from_numpy(x[:16, None]), velocities=torch.from_numpy(v[:16, None]))
    monkeypatch.setattr(multiseed, 'rollout', fake_rollout)
    ck = tmp_path/'synthetic.pt'
    ck.write_bytes(b'synthetic')
    args = argparse.Namespace(device='cpu', split='test', data_dir='unused', steps=16, starts=2, seeds=2,
                              seed0=3, sampling_steps=2, bootstrap=100, bootstrap_seed=4)
    result = multiseed.evaluate_checkpoint(ck, args)
    assert seen == [3, 4, 3, 4]
    assert result['metrics']['crash_fraction']['mean'] == .5
    assert result['metrics']['steps_survived']['mean'] == 13
    assert result['metrics']['rdf_l1']['n_missing'] == 2
    assert all(r['temperature_rel_err'] is None for r in result['rollouts'] if r['crash_fraction'])
    multiseed.write_report(dict(method='synthetic test', checkpoints=[result], wall_s=.04), tmp_path)
    assert json.loads((tmp_path/'metrics.json').read_text())['checkpoints'][0]['train_step'] == 40
    assert 'Bootstrap 95% CI' in (tmp_path/'metrics.md').read_text()
