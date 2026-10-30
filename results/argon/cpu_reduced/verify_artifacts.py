"""Verify the exported final checkpoint and paired comparison artifacts without git."""
import hashlib
import json
from pathlib import Path

import torch
import yaml

base = Path('results/argon/cpu_reduced')
env = json.loads((base / 'environment_data.json').read_text())
checks = {}
releases = {}
for name in ['argon_flow_v0.pt', 'argon_flow_v1_cpu_reduced.pt']:
    p = Path('checkpoints') / name
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest == Path(str(p) + '.sha256').read_text().split()[0]
    releases[name] = torch.load(p, weights_only=True, map_location='cpu')
    checks[name] = {'sha256': digest, 'bytes': p.stat().st_size, 'weights_only_load': True}
assert checks['argon_flow_v0.pt']['sha256'] == env['source_checkpoint_sha256']
final = torch.load(base / 'train/snapshots/step001000.pt', weights_only=False, map_location='cpu')
v1 = releases['argon_flow_v1_cpu_reduced.pt']
assert v1['train_step'] == final['step'] == 1000
assert v1['model'].keys() == final['model'].keys()
assert all(torch.equal(v, final['model'][k]) for k, v in v1['model'].items())
assert v1['model_cfg'] == releases['argon_flow_v0.pt']['model_cfg']
assert v1['model_cfg'] == yaml.safe_load(Path('configs/argon/B_S3a_seed1.yaml').read_text())['model']
checks['final_snapshot_tensors_exactly_equal_export'] = True
checks['model_architecture_unchanged'] = True
for name in ['v0', 'v1']:
    m = json.loads((base / name / 'metrics.json').read_text())
    assert (m['steps'], m['n_rollouts'], m['seeds_per_start'], m['stride_fs'], m['thermostat_tau_fs']) == (100, 1, 1, 50.0, 0.0)
    json.dumps(m, allow_nan=False)
eq = json.loads((base / 'equivariance.json').read_text())
assert len(eq['measurements']) == 24
assert len(eq['summary']) == 4
assert eq['sampling_steps'] == 10
json.dumps(eq, allow_nan=False)
checks['equivariance_measurements'] = len(eq['measurements'])
checks['paired_evaluation_settings_match'] = True
checks['all_metrics_finite_or_explicit_null'] = True
source_paths = json.loads((base / 'source_sha256.json').read_text()).keys()
(base / 'source_sha256.json').write_text(json.dumps({p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in source_paths}, indent=2) + '\n')
(base / 'integrity.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))
