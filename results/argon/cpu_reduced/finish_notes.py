import json,hashlib,time
from pathlib import Path
base=Path('results/argon/cpu_reduced')
load=lambda p:json.loads(Path(p).read_text())
env=load(base/'environment_data.json');eq=load(base/'equivariance.json')
metrics={k:load(base/k/'metrics.json') for k in ['v0','v1']}
commands=[json.loads(x) for x in (base/'commands.jsonl').read_text().splitlines()]
ckpaths=['checkpoints/argon_flow_v0.pt','checkpoints/argon_flow_v1_cpu_reduced.pt']
files={str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in map(Path,ckpaths)}
for parent in ['data/argon',str(base)]:
 files[parent]={'files':sum(1 for p in Path(parent).rglob('*') if p.is_file()),'total_file_bytes':sum(p.stat().st_size for p in Path(parent).rglob('*') if p.is_file())}
summary=dict(checkpoints=files,metrics=metrics,equivariance_summary=eq['summary'],commands=commands,
 total_wall_since_initial_inspection_s=time.time()-1793330468,
 training=load(base/'train'/'done.json'),test_starts_saved_frame_indices=[0],
 note='v0 as-is under new attention. v1 final snapshot, no test-based checkpoint selection. One start, one stochastic seed, 100 model steps, 10 flow steps, no thermostat.')
(base/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
lines=['\n## Completed run: measured results\n']
lines += [f"Environment: Python {env['python']}, PyTorch {env['torch']}, OpenMM {env['openmm']}, h5py {env['h5py']}, NumPy {env['numpy']}; CPU affinity {env['cpu_affinity']}.\n"]
train=summary['training'];lines += [f"Training completed {train['step']} optimizer steps in {train['hours']*3600:.6f} s ({train['hours']*60:.6f} minutes), measured by the training loop. Fresh initialization, seed 1, 1,945,520 parameters; original model config unchanged. Batch 1, warmup 100, unroll schedule [[0,1],[600,2],[900,4]], 1,000 total updates. Validation every 250 steps: two batches, one start, ten rollout steps, two flow steps. The final `snapshots/step001000.pt` was exported; `best.pt` was not selected using test results.\n"]
progress=[json.loads(l) for l in (base/'training.log').read_text().splitlines() if l.startswith('{') and '"elapsed_h"' in l]
validation=[json.loads(l) for l in (base/'train'/'log.jsonl').read_text().splitlines()]
lines += ['| Step | Mean training loss (last 50) | Unroll length | Elapsed training seconds |','|---|---:|---:|---:|']
for r in progress:lines += [f"| {r['step']} | {r['loss']:.10g} | {r['m']} | {r['elapsed_h']*3600:.6f} |"]
lines += ['\n| Step | Validation loss | Crash fraction | Final T (K) | RDF L1 | Rollout wall (s) |','|---|---:|---:|---:|---:|---:|']
for r in validation:lines += [f"| {r['step']} | {r['val_loss']:.10g} | {r['val_crash_frac']} | {r.get('val_T_end','unavailable')} | {r.get('val_rdf_l1','unavailable')} | {r['val_rollout_wall_s']:.6f} |"]
gt=load(base/'ground_truth'/'metrics.json')
lines += [f"\nGround-truth analysis over all four trajectories: T = {gt['T_mean_K']:.10g} K; mean total-energy drift = {gt['energy_drift_kJmol_per_ns_per_dof']['mean']:.10g} kJ/mol/ns/DOF; maximum absolute per-trajectory drift = {gt['energy_drift_kJmol_per_ns_per_dof']['max_abs']:.10g} in the same units. RDF peak = {gt['rdf']['Ar-Ar']['first_peak_A']:.10g} Å, height = {gt['rdf']['Ar-Ar']['first_peak_height']:.10g}; VACF first zero = {gt['vacf']['first_zero_fs']:.10g} fs; diffusion = {gt['diffusion']['D_cm2_per_s']:.10g} cm²/s (fit 2–5 ps). This all-trajectory summary is distinct from the test-only reference used in the paired evaluation.\n"]
lines += ['| Command | Wall seconds | Exit |','|---|---:|---:|']
for row in commands:lines += [f"| {row['name']} | {row['wall_s']:.6f} | {row['exit_code']} |"]
lines += [f"\nElapsed wall time since initial inspection: {summary['total_wall_since_initial_inspection_s']:.6f} s. Timed jobs overlapped; their sum is not total elapsed time.\n"]
lines += ['The existing evaluator uses the held-out trajectory 3. Its starting window begins at saved-frame index 0 (model-subsampled index 0), with random seed 0 applied identically. It starts from three 50 fs context frames. Test ground-truth temperature is computed independently of the checkpoint reference payload.\n']
keys=[('train step',lambda m:m['train_step']),('ground-truth T (K)',lambda m:m['T_ground_truth_K']),('crashed / 1',lambda m:m['stability']['crashed']),('steps before crash',lambda m:m['stability']['steps_before_crash']),('second-half T (K)',lambda m:m.get('temperature',{}).get('T_model_second_half_K')),('temperature relative error',lambda m:m.get('temperature',{}).get('rel_err')),('RDF L1',lambda m:m.get('rdf',{}).get('l1')),('RDF peak position error (Å)',lambda m:m.get('rdf',{}).get('peak_pos_err_A')),('RDF peak height relative error',lambda m:m.get('rdf',{}).get('peak_height_rel_err')),('VACF first zero, model (fs)',lambda m:m.get('vacf',{}).get('first_zero_model_fs')),('VACF first zero, reference (fs)',lambda m:m.get('vacf',{}).get('first_zero_gt_fs')),('VACF first-zero relative error',lambda m:m.get('vacf',{}).get('first_zero_rel_err')),('D model (cm²/s)',lambda m:m.get('diffusion',{}).get('D_model_cm2_s')),('D reference (cm²/s)',lambda m:m.get('diffusion',{}).get('D_gt_cm2_s')),('D relative error',lambda m:m.get('diffusion',{}).get('rel_err')),('step-1 RMSD (Å)',lambda m:m['short_range_rmsd_A']['step_1']),('step-10 RMSD (Å)',lambda m:m['short_range_rmsd_A']['step_10']),('step-20 RMSD (Å)',lambda m:m['short_range_rmsd_A']['step_20']),('rollout wall (s)',lambda m:m.get('speed',{}).get('wall_s')),('all v0 criteria pass',lambda m:m.get('v0_verdict',{}).get('all_pass'))]
lines += ['| Metric | v0 as-is | v1 CPU reduced |','|---|---:|---:|']
def fmt(x):
 if x is None:return 'unavailable (see metrics JSON)'
 if isinstance(x,float):return f'{x:.10g}'
 return str(x)
for label,fn in keys:lines += [f"| {label} | {fmt(fn(metrics['v0']))} | {fmt(fn(metrics['v1']))} |"]
lines += ['\nMissing metrics are not replaced with invented values: the evaluator skips structural/dynamical statistics when all rollouts crash, and a VACF zero can be absent in the measured horizon.\n']
lines += ['### O(3) output errors\n',f"Three shared test windows × proper/improper rotation per precision per checkpoint; seed {eq['seed']}; flow sampling steps {eq['sampling_steps']}. Total script wall {eq['wall_s']:.6f} s. Maxima below include every tested component, atom, window and transform.\n",'| Checkpoint | Precision | Max Δx error (Å) | Max v_new error (Å/fs) | Max across outputs |','|---|---|---:|---:|---:|']
for row in eq['summary']:lines += [f"| {Path(row['checkpoint']).name} | {row['dtype']} | {row['max_abs_dx_error_A']:.12g} | {row['max_abs_velocity_error_A_per_fs']:.12g} | {row['max_abs_output_error']:.12g} |"]
lines += ['\n### Files and integrity\n','| File / directory | Bytes |','|---|---:|']
for path,obj in files.items():lines += [f"| `{path}` | {obj.get('bytes',obj.get('total_file_bytes'))} |"]
for row in env['data']:lines += [f"| `{row['path']}` | {row['bytes']} |"]
lines += ['\nDirectory sizes sum regular-file bytes (not disk blocks); result directory size is measured before writing the final summary itself. Checkpoint SHA-256:\n']
for path in ckpaths:lines += [f"- `{path}`: `{files[path]['sha256']}`"]
lines += ['\nThe v0 file hash was checked against the original inspection and its `.sha256` sidecar. Both exported payloads load with `weights_only=True`. Model/release tests: 32 passed; full output is in `tests.log`. Exact per-transform equivariance numbers, per-trajectory metadata, source hashes, validation logs, plots, and evaluation JSON are retained in `results/argon/cpu_reduced/`.\n']
with open('docs/RETRAIN_NOTES_oct29.md','a') as f:f.write('\n'.join(lines)+'\n')
